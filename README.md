# pvc-nougat

Running [Nougat](https://github.com/facebookresearch/nougat) PDF→Markdown OCR on
**Intel Data Center GPU Max 1550 (PVC)** nodes, **1 MPI rank per GPU tile**, with
robust hallucination/repetition control.

Developed and verified on `chiatta00` (8× Max 1550 = 16 tiles).

---

## TL;DR

```bash
conda activate marker-xpu
export HF_HOME=/tmp/hf_cache FI_PSM3_UUID=$(cat /proc/sys/kernel/random/uuid)
cd src
mpiexec -n 16 python nougat_mpi.py \
  --pdfdir ~/repl5x100_pdfs --outdir /tmp/nougat_run/mmd --tiles-per-node 16
```

This launches 16 ranks, pins each to one GPU tile, and converts a directory of
PDFs to `.mmd` markdown — resume-safe, lock-free, hallucination-hardened.

**Measured throughput (16 tiles, one node):** ~2.4 pages/s (~145 pages/min),
32 docs / 268 pages in 110.7s, 0 watchdog timeouts. See [`bench/`](bench/).

---

## What's in here

| Path | What it is |
|---|---|
| `src/nougat_mpi.py` | **The launcher.** 1 MPI rank per tile, lock-free sharded queue, per-rank JSONL + reconcile. |
| `src/nougat_infer.py` | **Hardened inference.** batch=1 + hard token cap + per-page watchdog + repetition flagging. |
| `src/reference_server_pipeline/` | The earlier persistent-server design (kept for reference / single-process use). |
| `docs/01-gpu-bringup.md` | xpu-smi / libmetee fix and how to verify the GPUs are actually working. |
| `docs/02-iaf-fabric-diagnosis.md` | The IAF rebind-loop investigation and why it does **not** block nougat. |
| `docs/03-hallucination-control.md` | Root cause of the 724s blowup and the three-layer fix. |
| `docs/04-throughput.md` | Benchmark methodology and numbers. |
| `docs/05-runbook.md` | Operational runbook: launch, tune, resume, troubleshoot. |
| `marker-xpu-freeze.txt` | Full `pip freeze` of the working `marker-xpu` conda env. |
| `bench/` | Benchmark result JSON. |

Start with [`docs/05-runbook.md`](docs/05-runbook.md) to run it, or
[`docs/03-hallucination-control.md`](docs/03-hallucination-control.md) for the
most important engineering finding.

---

## Environment

Working stack (conda env `marker-xpu`, Python 3.12):

```
torch        2.8.0+xpu
ipex         2.8.10.post0+xpu   (intel_extension_for_pytorch)
nougat       0.1.17
transformers 4.56.1
mpi4py       4.1.2              (built against Intel MPI 2021.15)
```

GPUs: 8× Intel Data Center GPU Max 1550, 2 tiles each → `torch.xpu.device_count() == 16`.

Full freeze: [`marker-xpu-freeze.txt`](marker-xpu-freeze.txt).

> Note: a harmless `cannot import name 'intel' from 'triton._C.libtriton'`
> warning appears at import. Nougat inference works regardless; it only blocks
> `torch.compile`. See `docs/05-runbook.md`.

---

## Key findings (one-liners)

1. **xpu-smi** needed a `libmetee.so.3.1.5 → libmetee.so.4.0.0` compat symlink and
   `LD_LIBRARY_PATH`/full path under sudo. (`docs/01`)
2. **The IAF fabric rebind loop is a red herring for nougat** — it's the Xe-Link
   layer, not L0 device enumeration. L0 sees all 16 tiles and compute runs at
   126 TFLOP/s bf16. (`docs/02`)
3. **The 724s-per-4-page blowup was batch-level hallucination contagion**, not
   CPU fallback. Nougat's stop criterion only halts when *all* pages in a batch
   settle, so one runaway page dragged the whole batch to `max_length`. Fix:
   batch=1 + hard token cap + watchdog → 724s → 18.8s on the same file. (`docs/03`)
