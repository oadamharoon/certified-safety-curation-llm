#!/bin/bash
# Finish E23's XSTest scoring (WildGuard, ~14 GB, cannot fit beside the CDT cells): pause the xargs so
# no new cell starts, wait for the running cells, score, resume. ~15 min of CDT time.
PY="${PY:-python}"
L="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; W="$(dirname "$L")"; cd $L
PY1=${PY}; say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
cdt_running () { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_cdt.py && return 0; done; return 1; }
export PYTHONNOUSERSITE=1
kill -STOP 3769291 && say "xargs SIGSTOP"; while cdt_running; do sleep 60; done; say "GPU free"
$PY1 scripts/e6_utility.py > logs/e23_e6b.log 2>&1 && say "E23 XSTEST DONE" || say "E23 XSTEST FAILED"
kill -CONT 3769291 && say "xargs SIGCONT"; echo "[$(date +%m/%d-%H:%M)] E23 xstest scoring done, xargs SIGCONT" >> $W/runs/logs/v2f2/progress.log
