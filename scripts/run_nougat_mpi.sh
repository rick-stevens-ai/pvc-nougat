#!/bin/bash
# run_nougat_mpi.sh -- launch 1-rank-per-tile Nougat on a PVC node.
# Usage: ./run_nougat_mpi.sh <PDFDIR> <OUTDIR> [NRANKS]
set -eo pipefail   # NOTE: no -u (conda activate references unbound vars)

PDFDIR="${1:?usage: run_nougat_mpi.sh <PDFDIR> <OUTDIR> [NRANKS]}"
OUTDIR="${2:?usage: run_nougat_mpi.sh <PDFDIR> <OUTDIR> [NRANKS]}"
NRANKS="${3:-16}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- env ---
source ~/miniconda3/bin/activate 2>/dev/null || true
conda activate marker-xpu 2>/dev/null || true
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export FI_PSM3_UUID="${FI_PSM3_UUID:-$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo pvc-nougat)}"

# --- hallucination-control knobs (override as needed) ---
export NOUGAT_MAX_NEW_TOKENS="${NOUGAT_MAX_NEW_TOKENS:-1536}"
export NOUGAT_PAGE_TIMEOUT="${NOUGAT_PAGE_TIMEOUT:-90}"

echo "python : $(which python)"
echo "ranks  : $NRANKS   pdfdir: $PDFDIR   outdir: $OUTDIR"
echo "caps   : max_new_tokens=$NOUGAT_MAX_NEW_TOKENS page_timeout=$NOUGAT_PAGE_TIMEOUT"

mkdir -p "$OUTDIR"
exec mpiexec -n "$NRANKS" python "$HERE/src/nougat_mpi.py" \
  --pdfdir "$PDFDIR" --outdir "$OUTDIR" --tiles-per-node "$NRANKS"
