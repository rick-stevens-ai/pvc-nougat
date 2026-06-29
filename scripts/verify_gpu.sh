#!/bin/bash
# verify_gpu.sh -- confirm GPUs are actually usable for compute (not CPU fallback).
# Use this as the readiness gate, NOT a fabric/device-count query (see docs/02).
source ~/miniconda3/bin/activate 2>/dev/null || true
conda activate marker-xpu 2>/dev/null || true

python - <<'PY'
import sys
try:
    import torch, intel_extension_for_pytorch as ipex, time
    assert torch.xpu.is_available(), "xpu not available"
    n = torch.xpu.device_count()
    x = torch.randn(4096,4096, device="xpu", dtype=torch.bfloat16)
    torch.xpu.synchronize(); t=time.time()
    for _ in range(50): y=x@x
    torch.xpu.synchronize()
    tflops = 50*2*4096**3/(time.time()-t)/1e12
    print(f"READY  xpu devices={n}  bf16={tflops:.0f} TFLOP/s  name={torch.xpu.get_device_name(0)}")
    sys.exit(0)
except Exception as e:
    print("NOT READY:", type(e).__name__, e); sys.exit(1)
PY
