# 02 — The IAF fabric rebind loop (and why nougat doesn't care)

## The scary-looking symptom

`dmesg` shows the Intel Accelerator Fabric (IAF) endpoints re-binding in a
**tight loop** — every ~20–160 seconds:

```text
[1733871.x] mei_iaf i915.mei-gscfi.34304-...: bound i915.iaf.5
[1733871.x] mei_iaf ...: bound i915.iaf.2
...
[1734008.x] mei_iaf ...: bound i915.iaf.5     <- re-bound 137s later
[1734031.x] mei_iaf ...: bound i915.iaf.5     <- and again 23s later
```

Initial hypothesis was: "the fabric binds at the kernel level but L0 reports 0
devices even right after a fresh bind — GPUs stuck in a low-power state L0 won't
enumerate."

## What the data actually showed

| Layer | State | Evidence |
|---|---|---|
| i915 GPU + PCI | ✅ healthy | all 8 BDFs `runtime_status: active`, `control=on`; no GPU HANG/reset/wedged in dmesg |
| L0 compute (`torch.xpu`) | ✅ healthy | `device_count() == 16`; bf16 matmul `~126 TFLOP/s`; masked `ZE_AFFINITY_MASK=0` → 1 device + working matmul |
| DRM device nodes | ✅ present | `/dev/dri/card0–8`, `renderD128–135` all exist |
| IAF / Xe-Link fabric (`i915.iaf.*`) | 🔁 rebind loop | the dmesg churn above |

**Conclusion: the rebind loop is in the *fabric* (Xe-Link scale-up) layer, which
is separate from L0 device enumeration.** The GPUs enumerate and compute fine
*during* the churn.

## Why this matters (and doesn't) for nougat

- The IAF/Xe-Link fabric is for **GPU↔GPU collective communication** (oneCCL).
- **Nougat is single-GPU-per-tile.** Each MPI rank pins one tile via
  `ZE_AFFINITY_MASK` and never does cross-tile collectives.
- Therefore the rebind loop **cannot block nougat**. Verified by running real
  conversions to completion during the churn.

## The "0 devices" report — what it really was

Whatever was reporting 0 devices was querying the **fabric/sysman fabric-port**
layer (the IAF), **or** running without the conda env's L0 runtime on its path,
**or** in a context where `ZE_AFFINITY_MASK` masked everything out. It was *not*
the L0 compute path, which always saw 16 (or 1 when masked).

### Lesson for launchers/agents
**Gate readiness on the L0 compute probe, not on a fabric/device-count query
that can return 0.** See the readiness one-liner in `01-gpu-bringup.md`.

## Still worth reporting to admins

The rebind loop indicates the `mei_iaf` firmware / Xe-Link fabric is unstable.
It won't hurt single-tile nougat, but **multi-tile collective workloads
(oneCCL over Xe-Link) would be affected.** File it as a fabric-stability issue,
not a "GPUs are dead" issue.

## Handy probes used

```bash
cat /proc/uptime
dmesg | grep -iE 'iaf|fabric|mei' | tail -30
lsmod | grep -iE 'mei|iaf|i915'
ls -la /dev/dri/
ls -la /sys/bus/auxiliary/devices/ | grep iaf
for d in /sys/bus/pci/drivers/i915/*/; do
  echo "$(basename $d): $(cat $d/power/runtime_status) control=$(cat $d/power/control)"
done
```
