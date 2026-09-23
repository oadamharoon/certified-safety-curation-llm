#!/bin/bash
# E17 then E18 after the running CDT cells finish (E17's SFT OOM'd beside them); the CDT xargs
# (3769291) is stopped meanwhile and continued afterwards. E17's 0.5B condition also goes through
# the harm sets (inside run_e18.sh via E15_ALL).
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$W"
while pgrep -f "examples/train/train_cdt.py" > /dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done
bash scripts/run_e17.sh > logs/run_e17.out 2>&1
bash scripts/run_e18.sh > logs/run_e18.out 2>&1
kill -CONT 3769291; echo "[$(date +%m/%d-%H:%M)] E17+E18 done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
echo "E17 WAIT DONE"
