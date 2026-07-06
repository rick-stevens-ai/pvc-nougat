# Investigation: Porting `marker` to chiatta00 (Intel PVC / XPU)

Goal: run **marker** (PDF → Markdown) on the same `chiatta00` 8× Max 1550 (16-tile)
node we already use for Nougat, ideally reusing the `marker-xpu` conda env and the
1-rank-per-tile MPI pattern from `pvc-nougat`.

**Bottom line: feasible, and *much* less invasive than expected.** marker's GPU
work all runs through **surya-ocr**, which already has a clean, single-point device
abstraction (`settings.TORCH_DEVICE_MODEL`). There is **no hard CUDA dependency**
that blocks XPU. The port is ~3 small patches plus an MPI launcher, not a rewrite.

---

## 1. What we're starting from

| Item | State |
|---|---|
| marker source | `~/PDF-PVC/marker` (v1.10.2), git checkout, **not pip-installed** |
| surya-ocr | `0.17.1`, **already installed** in `marker-xpu` env |
| torch | `2.8.0+xpu` + `ipex 2.8.10.post0+xpu` (same stack proven with Nougat) |
| Device seen in plain shell | `cpu` — XPU only appears with the `chiatta-pvc` oneAPI module loaded |

marker is **architecturally different from Nougat**. Nougat is one
encoder-decoder model. marker is a *pipeline* of several surya models:

```
detection → layout → (foundation) recognition → table_rec → ocr_error
```

So "porting marker" really means "making **surya** run on XPU," plus wiring marker
on top. All the device logic lives in surya.

---

## 2. How surya selects the device (the one place that matters)

`surya/settings.py :: TORCH_DEVICE_MODEL`:

```python
if self.TORCH_DEVICE is not None:      # explicit override wins
    return self.TORCH_DEVICE
if torch.cuda.is_available(): return "cuda"
if torch.backends.mps.is_available(): return "mps"
try: import torch_xla ... return "xla"
return "cpu"
```

**There is no `xpu` branch.** On chiatta00 this silently returns `cpu` even when 16
PVC tiles are live — exactly what we saw (`device= cpu`). Two ways to fix:

- **Zero-code (recommended first):** set `TORCH_DEVICE=xpu` (env var
  `TORCH_DEVICE=xpu`, since surya/marker use `pydantic-settings`). The explicit
  override is honored everywhere downstream.
- **Patch:** add an `torch.xpu.is_available()` branch (nice-to-have, not required).

Everything else keys off `TORCH_DEVICE_MODEL`, so once it returns `xpu` the rest of
the pipeline follows.

---

## 3. The real blockers (all small) and their fixes

### 3a. dtype — **the one correctness issue to verify**
`surya/settings.py :: MODEL_DTYPE / MODEL_DTYPE_BFLOAT`:

```python
MODEL_DTYPE:        cpu→float32, xla→bfloat16, else→float16
MODEL_DTYPE_BFLOAT: cpu→float32, mps→bfloat16, else→bfloat16
```

For `xpu` we hit the `else` branches → weights load in **fp16 / bf16**. PVC handles
bf16 well (we measured 126 TFLOP/s bf16 with Nougat). **fp16** is the one to watch —
prefer bf16 on PVC. Easiest: force bf16 via the loader, or add an `xpu` dtype branch.

### 3b. batch sizes — **performance only, not a blocker**
`common/predictor.py :: default_batch_sizes` per model has keys `cpu/mps/cuda/xla`
but **no `xpu`**. `get_batch_size()` falls back to the **`cpu`** value when the key
is missing:

```
detection   cpu=8   cuda=36
layout      cpu=4   cuda=32
foundation  cpu=32  cuda=256
recognition cpu=32  cuda=256
table_rec   cpu=8   cuda=32
ocr_error   cpu=8   cuda=64
```

So without a patch, marker runs on XPU **at CPU-sized batches** → correct but slow.
Fix: add `"xpu": <cuda-ish>` to each `default_batch_sizes`, or set the per-model
`*_BATCH_SIZE` env vars. Start conservative (≈ cuda/2) per tile, tune like we did
for Nougat.

### 3c. flash-attention — **already handled, falls back to sdpa**
`is_flash_attn_2_supported()` returns `False` unless `"cuda" in str(device)`, so on
XPU surya cleanly selects **`sdpa`**. No action needed. (Same reason `torch.compile`
is off by default — and we already know `torch.compile` is broken in this env due to
the `triton._C.libtriton` import issue, see `docs/05`. Keep `COMPILE_* = False`.)

### 3d. `torch.cuda.empty_cache()` calls — **harmless no-op**
Three unguarded call sites (`detection`, `foundation` ×1, plus a bf16 check in the
loader guarded by `device == "cuda"`). Verified: `torch.cuda.empty_cache()` is a
**safe no-op when CUDA is absent**. It simply won't reclaim XPU memory — a minor
inefficiency, not a crash. Optional polish: swap for a device-aware
`empty_cache()` that calls `torch.xpu.empty_cache()`.

### 3e. `torch.cuda.get_device_capability()` — **only on the cuda path**
`common/util.py:239` is inside cuda-gated code; not reached on XPU.

---

## 4. Minimal port plan (in order)

1. **Install marker into `marker-xpu` (editable, no deps):**
   ```bash
   conda activate marker-xpu
   pip install -e ~/PDF-PVC/marker --no-deps
   ```
   (surya, torch, transformers, pdftext etc. are already present.)

2. **Smoke test on one tile, zero code changes:**
   ```bash
   module use ~/privatemodules; module load chiatta-pvc/2025.2
   conda activate marker-xpu
   export HF_HOME=/tmp/hf_cache
   export TORCH_DEVICE=xpu          # <-- forces surya onto PVC
   ZE_AFFINITY_MASK=0 marker_single <one.pdf> --output_dir /tmp/marker_out
   ```
   Confirms device=xpu, model download, and end-to-end on a single tile.

3. **Apply the 2 perf patches** (dtype→bf16 for xpu; add `xpu` batch-size keys).
   Keep them as a small patch set against surya so they survive reinstalls
   (mirror the `pvc-nougat/src` pattern — a thin `surya_xpu_patch.py` imported
   before model load, or a vendored settings override).

4. **MPI launcher** mirroring `nougat_mpi.py`: 1 rank/tile, `ZE_AFFINITY_MASK=rank`,
   sharded queue, per-rank JSONL + reconcile. marker's `models.create_model_dict()`
   is the per-rank model-load entrypoint.

5. **Benchmark** vs Nougat (`docs/04` methodology). marker does more work per page
   (layout + tables + OCR-error), so expect lower raw pages/s but richer output
   (tables, structure, equations).

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| fp16 numerics/perf on PVC | medium | force bf16 for xpu (3a) |
| sdpa attention slow/unsupported on XPU | low | known-good path with our torch+ipex; eager fallback exists |
| Per-tile model load OOM at 16× | medium | stagger loads (we already do for Nougat); tune batch sizes |
| marker pulls a dep that re-pins torch to CPU build | medium | install `--no-deps`; pin from `marker-xpu-freeze.txt` |
| `torch.compile` paths | n/a | leave all `COMPILE_*` off (broken in env, docs/05) |

## 6. Open questions to resolve next session

1. Does `TORCH_DEVICE=xpu` + bf16 actually run surya's foundation decoder on PVC
   end-to-end (step 2 smoke)? This is the single go/no-go test.
2. fp16 vs bf16 throughput on the foundation/recognition model.
3. Whether marker's `pdftext` page rasterization is a CPU bottleneck that caps
   16-tile scaling (it may be, since it's CPU-side pypdfium work).
