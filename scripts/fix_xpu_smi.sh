#!/bin/bash
# fix_xpu_smi.sh -- one-time host fix so xpu-smi runs.
# Creates the libmetee v3->v4 compat symlink and registers the xpu-smi lib dir.
# Requires sudo. Safe to re-run (idempotent). See docs/01-gpu-bringup.md.
set -e

XPU_LIB=/soft/tools/xpu-smi/1.2.22/lib64
METEE_V4=/usr/lib64/libmetee.so.4.0.0
METEE_V3=/usr/lib64/libmetee.so.3.1.5

echo "1) libmetee v3 -> v4 compat symlink"
if [ -e "$METEE_V3" ]; then
  echo "   already exists: $(readlink -f $METEE_V3)"
else
  sudo ln -s "$METEE_V4" "$METEE_V3"
  echo "   created $METEE_V3 -> $METEE_V4"
fi

echo "2) register xpu-smi lib dir with ldconfig"
echo "$XPU_LIB" | sudo tee /etc/ld.so.conf.d/xpu-smi.conf >/dev/null
sudo ldconfig
echo "   done"

echo "3) verify"
sudo /soft/tools/xpu-smi/1.2.22/bin/xpu-smi discovery | grep -c 'Device Name' \
  && echo "   xpu-smi discovery OK" \
  || echo "   xpu-smi still failing -- check docs/01"
