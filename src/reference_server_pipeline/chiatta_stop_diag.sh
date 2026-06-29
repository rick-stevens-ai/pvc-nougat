#!/bin/bash
# Stop the broken stream (all converts hanging to 900s timeout) and reset the
# queue rows it marked failed back to pending (they're not really failed -- the
# ENV is broken, not the PDFs).
pkill -f nougat_worker_lockfree 2>/dev/null
pkill -f nougat_convert_server 2>/dev/null
pkill -f chiatta_nougat_run 2>/dev/null
sleep 3
echo "workers remaining: $(pgrep -f nougat_worker_lockfree | grep -v pgrep | wc -l)"
# reset failed->pending (env bug, not PDF bug). Keep any genuine 'done' (none yet).
source ~/miniconda3/bin/activate 2>/dev/null; conda activate marker-xpu 2>/dev/null
python - /tmp/nougat_run/chiatta_nougat.sqlite <<'PY'
import sqlite3,sys
c=sqlite3.connect(sys.argv[1],timeout=60); c.execute("PRAGMA journal_mode=DELETE")
n=c.execute("UPDATE jobs SET status='pending', err=NULL WHERE status IN ('failed','running')").rowcount
c.commit()
print("reset", n, "rows ->", {k:v for k,v in c.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")})
c.close()
PY
echo "=== now diagnose triton/inference: what xpu backend does nougat decode use? ==="
python <<'PY' 2>&1 | tail -30
import torch
print("torch:", torch.__version__)
print("xpu available:", torch.xpu.is_available(), "count:", torch.xpu.device_count())
# does a trivial xpu matmul work? (basic op, no triton)
import time
t0=time.time()
a=torch.randn(512,512,device="xpu:0"); b=torch.randn(512,512,device="xpu:0")
c=(a@b); torch.xpu.synchronize()
print(f"basic xpu matmul OK in {time.time()-t0:.2f}s, sum={c.sum().item():.1f}")
# does triton actually work on xpu? this is what nougat decode may need
try:
    import triton
    print("triton:", triton.__version__)
    from triton._C.libtriton import intel
    print("triton intel backend: OK")
except Exception as e:
    print("triton intel backend BROKEN:", repr(e))
PY
