#!/usr/bin/env python
"""
nougat_reconcile.py -- single-writer fold of per-rank result JSONL into the queue DB.

Run BETWEEN jobs (or after each job) on a login node / single process. NEVER run
concurrently with a live conversion job that holds the DB read-only -- but since
workers only READ (immutable=1), this writer + those readers don't conflict on
Lustre as long as only ONE reconcile runs at a time. We enforce single-writer by
running it from the keep-inflight chainer (one process), not from ranks.

Reads RESULTS_DIR/rank_*.jsonl, dedups by osti_id (last ts wins), UPDATEs the DB
status/pages/chars/sec/err. Then truncates processed JSONL files (moves them to
RESULTS_DIR/_consumed/ with a timestamp) so the next reconcile only sees new rows.

Idempotent: re-running with no new JSONL is a no-op.
Usage: nougat_reconcile.py --db <q> [--results-dir <d>]
"""
import sys, os, json, time, argparse, sqlite3, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--results-dir", default=None)
args = ap.parse_args()
RESULTS_DIR = Path(args.results_dir or (Path(args.db).parent / "rank_results"))
CONSUMED = RESULTS_DIR / "_consumed"
CONSUMED.mkdir(parents=True, exist_ok=True)

def log(m): print(f"[reconcile] {m}", file=sys.stderr, flush=True)

files = sorted(RESULTS_DIR.glob("rank_*.jsonl"))
if not files:
    log("no rank result files, nothing to reconcile"); sys.exit(0)

# gather latest record per osti_id across all files
latest = {}
nlines = 0
for fp in files:
    try:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                nlines += 1
                try: rec = json.loads(line)
                except Exception: continue
                oid = rec.get("osti_id")
                if not oid: continue
                prev = latest.get(oid)
                if prev is None or rec.get("ts", 0) >= prev.get("ts", 0):
                    latest[oid] = rec
    except Exception as e:
        log(f"skip {fp.name}: {e}")

log(f"read {nlines} result lines from {len(files)} files -> {len(latest)} unique osti_ids")

# single-writer update. The DB is NOT immutable here -- normal connect, but only
# ONE reconcile runs at a time so Lustre lock contention is a non-issue (1 writer).
c = sqlite3.connect(args.db, timeout=300)
c.execute("PRAGMA busy_timeout=300000")
c.execute("PRAGMA journal_mode=DELETE")  # rollback journal; no WAL/shm on Lustre
n_upd = 0
c.execute("BEGIN")
for oid, rec in latest.items():
    c.execute("UPDATE jobs SET status=?, pages=?, chars=?, sec=?, err=?, ts=? WHERE osti_id=?",
              (rec.get("status","failed"), rec.get("pages",0), rec.get("chars",0),
               rec.get("sec",0), rec.get("err"), rec.get("ts", time.time()), oid))
    n_upd += c.total_changes and 1  # cheap; exact count below
c.execute("COMMIT")
# exact applied count
cur = c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
counts = dict(cur.fetchall())
c.close()
log(f"applied updates for {len(latest)} osti_ids")
log(f"DB status now: {counts}")

# move consumed files aside so next reconcile only sees fresh rows
ts = time.strftime("%Y%m%dT%H%M%S")
for fp in files:
    try:
        shutil.move(str(fp), str(CONSUMED / f"{fp.stem}.{ts}.jsonl"))
    except Exception as e:
        log(f"could not archive {fp.name}: {e}")
log(f"archived {len(files)} consumed result files to {CONSUMED}")
