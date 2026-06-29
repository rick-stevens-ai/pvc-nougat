#!/bin/bash
# SMOKE: 1 tile, seed full queue but cap worker at --max 2 to verify convert works.
BASE=/tmp/nougat_run
cd "$BASE"
source ~/miniconda3/bin/activate 2>/dev/null
conda activate marker-xpu 2>/dev/null
export HF_HOME=/tmp/hf_cache
PY=$(which python)

PDFDIR=$HOME/repl5x100_pdfs
OUTDIR=/tmp/nougat_run/mmd
DB=$BASE/chiatta_nougat.sqlite
RESULTS_DIR=$BASE/rank_results
mkdir -p "$OUTDIR" "$RESULTS_DIR"
export RESULTS_DIR

# seed
$PY - "$DB" "$PDFDIR" "$OUTDIR" <<'PY'
import sqlite3, sys, os
db, pdfdir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
c = sqlite3.connect(db, timeout=120); c.execute("PRAGMA journal_mode=DELETE")
c.execute("""CREATE TABLE IF NOT EXISTS jobs(osti_id TEXT PRIMARY KEY, pdf TEXT, out TEXT,
   status TEXT DEFAULT 'pending', tile INTEGER, pages INTEGER, chars INTEGER, sec REAL, err TEXT, ts REAL)""")
pdfs=[]
for root,_,files in os.walk(pdfdir):
    for fn in files:
        if fn.lower().endswith(".pdf"): pdfs.append(os.path.join(root,fn))
for p in pdfs:
    oid=os.path.splitext(os.path.basename(p))[0]; out=os.path.join(outdir,oid+".mmd")
    if os.path.exists(out) and os.path.getsize(out)>0: continue
    c.execute("INSERT OR IGNORE INTO jobs(osti_id,pdf,out,status) VALUES(?,?,?,'pending')",(oid,p,out))
c.execute("UPDATE jobs SET status='pending' WHERE status='running'"); c.commit()
r={k:v for k,v in c.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}
print(f"[seed] {len(pdfs)} pdfs | {r}", flush=True); c.close()
PY

echo "=== SMOKE: 1 tile, --max 2 ==="
ZE_AFFINITY_MASK=0 NSHARDS=1 SHARD=0 timeout 600 \
  $PY nougat_worker_lockfree.py --db "$DB" --tile 0 \
    --server nougat_convert_server.py --timeout 600 --max 2 --python "$PY" 2>&1 | tail -25
echo "=== reconcile ==="
$PY nougat_reconcile.py --db "$DB" --results-dir "$RESULTS_DIR" 2>&1 | tail -5
echo "=== .mmd produced ==="
find "$OUTDIR" -name "*.mmd" -newermt "20 minutes ago" 2>/dev/null | head
find "$OUTDIR" -name "*.mmd" -newermt "20 minutes ago" 2>/dev/null | wc -l
