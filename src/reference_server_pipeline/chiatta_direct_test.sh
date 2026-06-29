#!/bin/bash
# Direct single-PDF conversion test on chiatta00, tile 0, 180s cap, full stderr.
# Bypasses the queue/worker -- just: does ONE nougat convert actually return on PVC?
source ~/miniconda3/bin/activate 2>/dev/null
conda activate marker-xpu 2>/dev/null
export HF_HOME=/tmp/hf_cache
PDF=$(find ~/repl5x100_pdfs/OSTI-100 -name "*.pdf" 2>/dev/null | head -1)
[ -z "$PDF" ] && PDF=$(find ~/repl5x100_pdfs -name "*.pdf" 2>/dev/null | head -1)
echo "testing: $PDF"
cd /tmp/nougat_run
# feed one request to the convert server directly, 180s cap
ZE_AFFINITY_MASK=0 timeout 200 python - "$PDF" <<'PY' 2>&1 | tail -40
import sys, os, time, subprocess, json
pdf = sys.argv[1]
out = "/tmp/nougat_run/_directtest.mmd"
# launch the convert server, send one request, watch for response
env = dict(os.environ); env["ZE_AFFINITY_MASK"]="0"; env["NOUGAT_CHUNK_PAGES"]="4"
p = subprocess.Popen(["python","nougat_convert_server.py"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=sys.stderr, env=env, text=True, bufsize=1)
ready = p.stdout.readline()
print("READY LINE:", ready.strip(), flush=True)
t0=time.time()
p.stdin.write(json.dumps({"pdf":pdf,"out":out})+"\n"); p.stdin.flush()
print(f"[{time.time()-t0:.1f}s] request sent, waiting for result...", flush=True)
# read with our own timeout
import threading
res={"line":None}
def rd(): res["line"]=p.stdout.readline()
th=threading.Thread(target=rd,daemon=True); th.start(); th.join(170)
if th.is_alive():
    print(f"[{time.time()-t0:.1f}s] STILL HANGING after 170s -- conversion does NOT return on PVC", flush=True)
    p.kill()
else:
    print(f"[{time.time()-t0:.1f}s] RESULT:", res["line"], flush=True)
PY
echo "=== did it write mmd? ==="
ls -la /tmp/nougat_run/_directtest.mmd 2>/dev/null && head -c 300 /tmp/nougat_run/_directtest.mmd 2>/dev/null
