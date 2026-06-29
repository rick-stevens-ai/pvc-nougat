# 01 — GPU bring-up: getting `xpu-smi` and Level Zero working

## Symptom chain

```text
$ sudo xpu-smi discovery
sudo: xpu-smi: command not found

$ sudo /soft/tools/xpu-smi/1.2.22/bin/xpu-smi
... error while loading shared libraries: libxpum.so.1: cannot open shared object file
... then: libmetee.so.3.1.5: cannot open shared object file
```

Three distinct problems, peeled one at a time.

### 1. `command not found` under sudo
`sudo` uses a restricted `secure_path` that excludes `/soft/tools/...`.
**Fix:** call the full path.

### 2. `libxpum.so.1` not found
`sudo` strips the environment, so the dynamic linker can't find xpu-smi's own
libs in `/soft/tools/xpu-smi/1.2.22/lib64`.
**Fix:** pass the library path into the privileged process. The robust form that
survives sudo env-stripping and paste mangling is a quoted `bash -c`:

```bash
sudo bash -c 'LD_LIBRARY_PATH=/soft/tools/xpu-smi/1.2.22/lib64 /soft/tools/xpu-smi/1.2.22/bin/xpu-smi discovery'
```

> Pitfall we hit: `sudo VAR=value cmd` and multi-line backslash pastes both
> failed (one sudo policy rejected inline `VAR=value`; pastes injected stray
> characters → `sudo:  : command not found`). Single-line `bash -c '...'` is
> paste-proof.

### 3. `libmetee.so.3.1.5` not found  ← the real root cause
xpu-smi 1.2.22's `libxpum.so.1` was linked against **libmetee v3**, but the
system only ships **v4** (`/usr/lib64/libmetee.so.4.0.0`, plus 4.2.1 under
`/usr/local/intel-gpu-umd/lib64`). v3 is installed nowhere.

We verified the 5 metee symbols libxpum actually imports
(`TeeConnect/Disconnect/Init/Read/Write`) are **all present in v4**, so a
compatibility symlink is safe:

```bash
sudo ln -s /usr/lib64/libmetee.so.4.0.0 /usr/lib64/libmetee.so.3.1.5
sudo ldconfig
```

(The symlink already exists on `chiatta00` as of this work.)

## Result

```text
$ sudo bash -c 'LD_LIBRARY_PATH=/soft/tools/xpu-smi/1.2.22/lib64 /soft/tools/xpu-smi/1.2.22/bin/xpu-smi discovery'
| 0 | Intel(R) Data Center GPU Max 1550 ... /dev/dri/card1 |
... 8 GPUs total ...
```

## Optional: make it permanent / cleaner

Register the xpu-smi lib dir system-wide so you don't need `LD_LIBRARY_PATH`:

```bash
echo /soft/tools/xpu-smi/1.2.22/lib64 | sudo tee /etc/ld.so.conf.d/xpu-smi.conf
sudo ldconfig
# then just:
sudo /soft/tools/xpu-smi/1.2.22/bin/xpu-smi discovery
```

Or a user shell function (`~/.bashrc`):

```bash
xpu-smi-sudo() { sudo bash -c 'LD_LIBRARY_PATH=/soft/tools/xpu-smi/1.2.22/lib64 /soft/tools/xpu-smi/1.2.22/bin/xpu-smi "$@"' _ "$@"; }
```

---

## Verifying the GPU is *actually* used (not CPU fallback)

xpu-smi listing devices is necessary but not sufficient. Confirm L0 compute:

```bash
conda activate marker-xpu
python - <<'PY'
import torch, intel_extension_for_pytorch as ipex, time
print("xpu.device_count:", torch.xpu.device_count())          # -> 16
x = torch.randn(4096,4096, device="xpu", dtype=torch.bfloat16)
torch.xpu.synchronize(); t=time.time()
for _ in range(50): y = x@x
torch.xpu.synchronize()
flops = 50*2*4096**3
print(f"{flops/(time.time()-t)/1e12:.1f} TFLOP/s bf16")        # -> ~126 TFLOP/s
PY
```

`~126 TFLOP/s bf16` confirms real GPU execution. With `ZE_AFFINITY_MASK=0` the
count drops to 1 (the pinned tile) and a matmul still runs — exactly how the MPI
workers pin tiles.

**Readiness probe for an agent/launcher** (gate on compute, not on fabric):

```bash
python -c "import torch, intel_extension_for_pytorch; \
assert torch.xpu.is_available() and torch.xpu.device_count()>0; \
(torch.randn(8,8,device='xpu')@torch.randn(8,8,device='xpu')).sum().item()" && echo READY
```
