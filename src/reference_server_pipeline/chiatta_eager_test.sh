#!/bin/bash
# Test: does nougat convert work if we FORCE eager mode (no triton/inductor)?
source ~/miniconda3/bin/activate 2>/dev/null
conda activate marker-xpu 2>/dev/null
export HF_HOME=/tmp/hf_cache
# kill any compile/triton routing
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_DISABLE_TRITON=1
PDF=$(find ~/repl5x100_pdfs/OSTI-100 -name "*.pdf" 2>/dev/null | head -1)
echo "testing (eager-forced): $PDF"
cd /tmp/nougat_run
ZE_AFFINITY_MASK=0 timeout 240 python - "$PDF" <<'PY' 2>&1 | tail -30
import sys, os, time, subprocess, json, threading
pdf = sys.argv[1]; out = "/tmp/nougat_run/_eagertest.mmd"
env = dict(os.environ); env["ZE_AFFINITY_MASK"]="0"; env["NOUGAT_CHUNK_PAGES"]="4"
p = subprocess.Popen(["python","nougat_convert_server.py"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=sys.stderr, env=env, text=True, bufsize=1)
ready=p.stdout.readline(); print("READY:", ready.strip(), flush=True)
t0=time.time(); p.stdin.write(json.dumps({"pdf":pdf,"out":out})+"\n"); p.stdin.flush()
print(f"[{time.time()-t0:.1f}s] sent, waiting...", flush=True)
res={"line":None}
def rd(): res["line"]=p.stdout.readline()
th=threading.Thread(target=rd,daemon=True); th.start(); th.join(210)
if th.is_alive():
    print(f"[{time.time()-t0:.1f}s] STILL HANGING -- eager mode did NOT fix it", flush=True); p.kill()
else:
    print(f"[{time.time()-t0:.1f}s] RESULT:", res["line"], flush=True)
PY
echo "=== mmd written? ==="
ls -la /tmp/nougat_run/_eagertest.mmd 2>/dev/null && echo "--- first 200 chars ---" && head -c 200 /tmp/nougat_run/_eagertest.mmd
