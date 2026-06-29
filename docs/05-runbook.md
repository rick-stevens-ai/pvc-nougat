# 05 — Runbook

## Prerequisites

- conda env `marker-xpu` (torch 2.8+xpu, ipex 2.8.10, nougat 0.1.17, mpi4py 4.1.2)
- Intel MPI on PATH (the env provides `mpiexec`)
- GPUs visible: `python -c "import torch,intel_extension_for_pytorch; print(torch.xpu.device_count())"` → 16

## Run (single node, 16 tiles)

```bash
conda activate marker-xpu
export HF_HOME=/tmp/hf_cache
export FI_PSM3_UUID=$(cat /proc/sys/kernel/random/uuid)   # silences Intel MPI warning
cd src
mpiexec -n 16 python nougat_mpi.py \
  --pdfdir /path/to/pdfs \
  --outdir /path/to/out_mmd \
  --tiles-per-node 16
```

Outputs:
- `<outdir>/<id>.mmd` — one markdown file per PDF
- `<outdir>/_results/rank_<r>.jsonl` — per-rank, per-doc status (append-only)
- `<outdir>/_summary.json` — totals + throughput (written by rank 0)
- `<outdir>/_manifest.json` — the sorted PDF list (written by rank 0)

## Tuning knobs (env vars)

| Var | Default | Effect |
|---|---|---|
| `NOUGAT_MAX_NEW_TOKENS` | 1536 | hard per-page token cap. Lower = faster on repetition-heavy input, risk clipping dense pages. |
| `NOUGAT_PAGE_TIMEOUT` | 90 | per-page wall watchdog (s). Set ≈ slowest legit page × 1.5. |
| `NOUGAT_PAGE_SEP` | `\n\n` | how per-page markdown is glued. |
| `TILES_PER_NODE` | 16 | tiles per node (also `--tiles-per-node`). |
| `HF_HOME` | — | HuggingFace cache; point at fast local disk (`/tmp`). |

Example — throughput mode:
```bash
NOUGAT_MAX_NEW_TOKENS=1024 mpiexec -n 16 python nougat_mpi.py ...
```

## Resume

Re-running the same command is **idempotent**: any PDF whose `.mmd` already
exists (size > 0) is skipped. Safe to relaunch after an interruption.

## Multi-node

Launch under your scheduler so `mpiexec` places 16 ranks per node. Each rank
pins its **local** tile (from `MPI_LOCALRANKID`/`PMI_LOCAL_RANK`), and work
sharding is **global** across all ranks (`index % world == rank`). No code
changes needed. Use a **shared** `--outdir` so resume + reconcile see all ranks.

> Lustre note: results are per-rank append-only JSONL (no shared SQLite writes),
> matching the lock-free contract that fixed the prior 3072-rank Lustre
> "locking protocol" cascade. See `src/reference_server_pipeline/` headers.

## Verify GPUs before a big run

```bash
python -c "import torch, intel_extension_for_pytorch; \
assert torch.xpu.is_available() and torch.xpu.device_count()>0; \
(torch.randn(8,8,device='xpu')@torch.randn(8,8,device='xpu')).sum().item()" && echo READY
```
Gate launches on **this** (L0 compute), not on a fabric/device-count query that
can return 0 during IAF churn (see `docs/02`).

`xpu-smi` (needs the libmetee fix from `docs/01`):
```bash
sudo bash -c 'LD_LIBRARY_PATH=/soft/tools/xpu-smi/1.2.22/lib64 /soft/tools/xpu-smi/1.2.22/bin/xpu-smi discovery'
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| A doc takes hundreds of s/page | Old batched path. Use `nougat_mpi.py` (batch=1 + caps). See `docs/03`. |
| `libmetee.so.3.1.5: cannot open` | Add compat symlink + `ldconfig`. `docs/01`. |
| "0 devices" from some probe | Probing fabric/sysman, not L0. Use the readiness probe above. `docs/02`. |
| `cannot import name 'intel' from triton._C.libtriton` | Cosmetic for nougat (inference works). Only blocks `torch.compile`. Reinstall a matching `pytorch-triton-xpu` if you need compile. |
| Intel MPI `FI_PSM3_UUID was not generated` warning | Harmless; set `FI_PSM3_UUID=$(uuidgen)` to silence. |
| One rank much slower (long wall) | Load imbalance with few docs/tile. Larger inputs amortize; or lower `NOUGAT_MAX_NEW_TOKENS`. |
| Page produces empty output, `timed_out=1` | Watchdog fired. Raise `NOUGAT_PAGE_TIMEOUT` or route the doc to marker fallback. |

## Reconcile-only (re-summarize existing results)

The summary is regenerated each run from the JSONL. To re-tally manually:
```bash
python - <<'PY'
import json, glob
tot={'docs':0,'pages':0,'chars':0,'repeated':0,'timed_out':0}
for f in glob.glob('OUTDIR/_results/rank_*.jsonl'):
    for l in open(f):
        j=json.loads(l)
        tot['docs']+=1; tot['pages']+=j.get('pages',0); tot['chars']+=j.get('chars',0)
        tot['repeated']+=j.get('repeated',0); tot['timed_out']+=j.get('timed_out',0)
print(tot)
PY
```
