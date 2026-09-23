#!/bin/bash
until grep -q "E3 SWEEP2 DONE" $W/logs/e3/run_sweep2.out 2>/dev/null; do sleep 120; done
cd $W && bash scripts/run_e3_sweep2.sh
echo E3 SWEEP3 DONE
