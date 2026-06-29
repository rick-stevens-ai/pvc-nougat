# 04 — Throughput

All numbers on `chiatta00`, 8× Intel Data Center GPU Max 1550 (16 tiles),
conda env `marker-xpu`, nougat-small (`0.1.0-small`), bf16.

## Single-tile baseline

| Workload | Time | Notes |
|---|---|---|
| bf16 matmul 4096³ ×50 | 0.054s → **126 TFLOP/s** | raw GPU sanity check |
| nougat, one page | 3.0s | early-stop fired |
| nougat, 4-page PDF (batch=1) | 17.4s → **4.3s/page** | healthy |
| nougat, 4-page PDF (old batch=4 server) | 724s | hallucination contagion — see `docs/03` |

→ ~**0.2 pages/s/tile** typical (varies with page density and repetition rate).

## Full node: 16 MPI ranks / 16 tiles

Input: 32 PDFs (≤12 pages each), 268 pages total.
(`bench/mpi16_summary.json`, per-doc in `bench/mpi16_per_doc.csv`.)

```json
{
  "docs": 32, "pages": 268, "chars": 1058287,
  "repeated": 30, "timed_out": 0,
  "wall": 110.7, "ranks": 16
}
```

| Metric | Value |
|---|---|
| Docs converted | 32 / 32 ✅ |
| Pages | 268 |
| Wall (slowest rank) | 110.7s |
| **Throughput** | **2.42 pages/s ≈ 145 pages/min** (~17.3 docs/min) |
| Repetition-flagged pages | 30 (truncated cleanly, doc still produced) |
| Watchdog timeouts | 0 |
| Model load (per rank, one-time) | ~8.7s, all ranks in parallel |

### Why aggregate (2.42 pg/s) < 16 × single-tile (≈3.2 pg/s ideal)

- Small batch (32 docs / 16 tiles ≈ 2 docs each) → poor load balance; the
  slowest rank sets the wall. Larger inputs amortize this.
- This corpus is repetition-heavy (30 flagged pages) → those pages run to the
  hard token cap (1536) rather than stopping early.

On a large, well-balanced job expect throughput to approach the per-tile rate ×
16 minus tail effects.

## Methodology

- PDFs filtered to ≤12 pages via `pypdf` page count.
- Each rank pins `ZE_AFFINITY_MASK=local_rank`, loads the model once, processes
  its disjoint shard `(index % world == rank)`.
- Timing excludes the one-time model load (reported separately); wall is the
  slowest rank's processing time.
- Throughput = total pages / slowest-rank wall.

Reproduce:

```bash
conda activate marker-xpu
export HF_HOME=/tmp/hf_cache FI_PSM3_UUID=$(cat /proc/sys/kernel/random/uuid)
cd src
mpiexec -n 16 python nougat_mpi.py \
  --pdfdir /tmp/nougat_pool --outdir /tmp/nougat_mpi16 --tiles-per-node 16
cat /tmp/nougat_mpi16/_summary.json
```
