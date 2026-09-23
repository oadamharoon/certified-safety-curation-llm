#!/bin/bash
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$W"
bash scripts/run_e19_e20.sh > logs/run_e19_e20.out 2>&1
kill -CONT 3769291; echo "[$(date +%m/%d-%H:%M)] E17-E20 done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
echo "E19E20 WAIT DONE"
