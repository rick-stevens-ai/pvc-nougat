#!/bin/bash
# chiatta00 small Nougat job stream -- 16 tiles, local nvme queue. No Lustre.
# Seeds from staged PDFs (skips already-.mmd), runs 16 lock-free workers, reconciles.
# NO set -u (conda refs unbound vars).
BASE=/tmp/nougat_run
cd "$BASE"
source ~/miniconda3/bin/activate 2>/dev/null
conda activate marker-xpu 2>/dev/null
export HF_HOME=/tmp/hf_cache
PY=$(which python)
echo "python: $PY"

PDFDIR="${PDFDIR:-$HOME/repl5x100_pdfs}"
OUTDIR="${OUTDIR:-/tmp/nougat_run/mmd}"
DB="$BASE/chiatta_nougat.sqlite"
RESULTS_DIR="$BASE/rank_results"
TIMEOUT="${TIMEOUT:-900}"
mkdir -p "$OUTDIR" "$RESULTS_DIR"
export RESULTS_DIR

# ---- seed queue (idempotent): one row per PDF lacking a .mmd ----
$PY - "$DB" "$PDFDIR" "$OUTDIR" <<'PY'
import sqlite3, sys, os, glob
db, pdfdir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
c = sqlite3.connect(db, timeout=120)
c.execute("PRAGMA journal_mode=DELETE")
c.execute("""CREATE TABLE IF NOT EXISTS jobs(
    osti_id TEXT PRIMARY KEY, pdf TEXT, out TEXT, status TEXT DEFAULT 'pending',
    tile INTEGER, pages INTEGER, chars INTEGER, sec REAL, err TEXT, ts REAL)""")
pdfs = []
for root,_,files in os.walk(pdfdir):
    for fn in files:
        if fn.lower().endswith(".pdf"):
            pdfs.append(os.path.join(root, fn))
added=0
for p in pdfs:
    oid = os.path.splitext(os.path.basename(p))[0]
    out = os.path.join(outdir, oid + ".mmd")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        continue  # already parsed
    c.execute("INSERT OR IGNORE INTO jobs(osti_id,pdf,out,status) VALUES(?,?,?,'pending')",(oid,p,out))
    added += c.total_changes and 1
# reap stale running
c.execute("UPDATE jobs SET status='pending' WHERE status='running'")
c.commit()
r = {k:v for k,v in c.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}
print(f"[seed] scanned {len(pdfs)} pdfs | queue status: {r}", flush=True)
c.close()
PY

# ---- launch 16 workers, one per tile ----
NSHARDS=16
echo "############ chiatta00 NOUGAT | $(date) | 16 tiles ############"
pids=()
for tile in $(seq 0 15); do
  ZE_AFFINITY_MASK=$tile NSHARDS=$NSHARDS SHARD=$tile \
    $PY nougat_worker_lockfree.py --db "$DB" --tile "$tile" \
      --server nougat_convert_server.py --timeout "$TIMEOUT" --python "$PY" \
      > "$RESULTS_DIR/tile_${tile}.log" 2>&1 &
  pids+=($!)
  sleep 1   # stagger model loads
done
echo "launched ${#pids[@]} workers: ${pids[*]}"
# wait for all
for p in "${pids[@]}"; do wait "$p"; done
echo "[run] all workers exited"

# ---- reconcile ----
$PY nougat_reconcile.py --db "$DB" --results-dir "$RESULTS_DIR"
$PY - "$DB" <<'PY'
import sqlite3, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
r = {k:v for k,v in c.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}
print("############ chiatta00 FINAL:", r, "############")
PY
echo "mmd produced:"; find "$OUTDIR" -name "*.mmd" -newermt "1 hour ago" | wc -l
echo "############ DONE $(date) ############"
