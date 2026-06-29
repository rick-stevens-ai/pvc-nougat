#!/usr/bin/env python
"""
nougat_worker_lockfree.py -- one worker pinned to one PVC tile.

LOCK-FREE REWRITE (2026-06-28) for 3,072-rank Aurora prod on Lustre.

ROOT CAUSE of prior no-op jobs:
  sqlite3.OperationalError: "locking protocol" (566 ranks/job) -- SQLite POSIX
  byte-range locking is BROKEN on Lustre. WAL needs shm locking Lustre can't do.
  The old _retry() only retried "locked"/"busy", so "locking protocol" raised
  immediately -> rank exits rc=1 -> mpiexec Hydra proxy cascade kills the whole
  256-node job in ~8 min after ~100 docs. NOT a poison-PDF problem.

FIX: eliminate ALL concurrent SQLite writes during the run.
  1. ONE read-only snapshot of pending work at startup (immutable=1, NO locks).
  2. Static disjoint shard: this rank processes rows where (rowid % NSHARDS)==SHARD.
     Shards are disjoint by construction -> no two ranks ever touch the same row.
  3. Results written to a PER-RANK JSONL on Lustre (one file per global rank,
     append-only, no shared lock).
  4. A separate single-writer reconcile step (nougat_reconcile.py) folds all
     per-rank result files back into the DB AFTER the job (or between jobs).

Resume-safe: any row not yet reconciled stays 'pending' and is re-picked next
job (idempotent; convert overwrites .mmd). Orphans from dead ranks self-heal.

Env: NSHARDS, SHARD set by rank wrapper. RESULTS_DIR for per-rank JSONL.
Usage: nougat_worker_lockfree.py --db <q> --tile <N> --server <p> --timeout <s> [--max <n>]
"""
import sys, os, json, time, argparse, sqlite3, subprocess, threading
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--tile", type=int, required=True)
ap.add_argument("--server", required=True)
ap.add_argument("--timeout", type=float, default=300.0)
ap.add_argument("--max", type=int, default=0)
ap.add_argument("--python", default=sys.executable)
args = ap.parse_args()
TILE = args.tile
NSHARDS = int(os.environ.get("NSHARDS", "1"))
SHARD   = int(os.environ.get("SHARD", "0"))
RESULTS_DIR = os.environ.get("RESULTS_DIR",
                             str(Path(args.db).parent / "rank_results"))

def log(m):
    print(f"[worker tile={TILE} shard={SHARD}/{NSHARDS} pid={os.getpid()}] {m}",
          file=sys.stderr, flush=True)

# ---- 1) READ-ONLY snapshot of this shard's pending work. NO locks on Lustre. ----
def load_shard_jobs():
    """Open DB read-only & immutable (no lock files touched at all on Lustre).
    Returns list of (osti_id, pdf, out) for this rank's disjoint shard."""
    uri = f"file:{args.db}?immutable=1&mode=ro"
    # immutable=1 tells SQLite the file will not change -> it takes NO locks
    # whatsoever. Safe here: the DB is frozen for the duration of the job
    # (reconcile runs only after qsub completes). This is the Lustre-safe path.
    c = sqlite3.connect(uri, uri=True, timeout=120)
    try:
        rows = c.execute(
            "SELECT osti_id, pdf, out FROM jobs "
            "WHERE status='pending' AND (rowid % ?)=?",
            (NSHARDS, SHARD)
        ).fetchall()
    finally:
        c.close()
    return [{"osti_id": r[0], "pdf": r[1], "out": r[2]} for r in rows]

# ---- 3) per-rank result sink: append-only JSONL, no shared lock ----
class ResultSink:
    def __init__(self):
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        self.path = Path(RESULTS_DIR) / f"rank_{SHARD:05d}.jsonl"
        # append mode: a re-run of the same shard (resume) keeps prior results;
        # reconcile dedups by osti_id (last-wins).
        self.f = open(self.path, "a", buffering=1)  # line-buffered
        log(f"results -> {self.path}")
    def record(self, oid, ok, pages, chars, sec, err):
        rec = {"osti_id": oid, "status": "done" if ok else "failed",
               "pages": pages, "chars": chars, "sec": sec,
               "err": err, "ts": time.time(), "tile": TILE}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        try: os.fsync(self.f.fileno())
        except Exception: pass
    def close(self):
        try: self.f.close()
        except Exception: pass

class Server:
    def __init__(self): self.p=None; self.start()
    def start(self):
        env = dict(os.environ); env["ZE_AFFINITY_MASK"] = str(TILE)
        self.p = subprocess.Popen([args.python, args.server], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=sys.stderr, env=env, text=True, bufsize=1)
        line = self.p.stdout.readline()
        if not line: raise RuntimeError("server died before readiness")
        j = json.loads(line); assert j.get("ready"), f"bad first line {line!r}"
        log(f"server ready (model load {j.get('load_sec')}s)")
    def convert(self, pdf, out, timeout):
        if self.p.poll() is not None: return None, True
        try:
            self.p.stdin.write(json.dumps({"pdf": pdf, "out": out}) + "\n"); self.p.stdin.flush()
        except (BrokenPipeError, OSError): return None, True
        res = {"line": None}
        def rd():
            try: res["line"] = self.p.stdout.readline()
            except Exception: res["line"] = None
        th = threading.Thread(target=rd, daemon=True); th.start(); th.join(timeout)
        if th.is_alive():
            log(f"HARD TIMEOUT {timeout}s on {pdf} -- killing server"); self.kill(); return None, True
        line = res["line"]
        if not line: return None, True
        try: return json.loads(line), False
        except Exception as e: log(f"bad resp json {e} {line!r}"); return None, True
    def kill(self):
        if self.p and self.p.poll() is None:
            try: self.p.kill(); self.p.wait(timeout=10)
            except Exception: pass
    def respawn(self): self.kill(); log("respawning server..."); self.start()

# ---- main ----
jobs = load_shard_jobs()
log(f"shard has {len(jobs)} pending jobs")
if not jobs:
    log("shard empty, clean exit"); sys.exit(0)

sink = ResultSink()
srv = Server()
done = 0
for job in jobs:
    if args.max and done >= args.max:
        log(f"reached --max {args.max}, stopping"); break
    oid, pdf, out = job["osti_id"], job["pdf"], job["out"]
    resp, died = srv.convert(pdf, out, args.timeout)
    if died or resp is None:
        sink.record(oid, False, 0, 0, args.timeout, "server died/timeout (native crash or hang)")
        log(f"[FAIL] {oid} server died/timeout -> respawn")
        try: srv.respawn()
        except Exception as e: log(f"respawn FAILED {e} -- exiting"); break
    else:
        sink.record(oid, resp["ok"], resp.get("pages",0), resp.get("chars",0), resp.get("sec",0), resp.get("err"))
        log(f"[{'OK' if resp['ok'] else 'FAIL'}] {oid} pages={resp.get('pages',0)} chars={resp.get('chars',0)} {resp.get('sec',0)}s")
    done += 1
srv.kill()
sink.close()
log(f"worker done, processed {done} jobs")
