#!/bin/bash
# waits for E4 to release the GPU, then runs the two BeaverTails-free judges on everything
PY="${PY:-python}"
# Run this after e4 (the refusal dose sweep): both want the GPU to itself.
cd $W
env PYTHONNOUSERSITE=1 ${PY} scripts/e1_panel2.py > logs/e1_panel2.log 2>&1; grep -v -i "warn\|Fetching\|Loading\|legacy" logs/e1_panel2.log; echo PANEL2 DONE
env PYTHONNOUSERSITE=1 ${PY} scripts/e1_pool_refusal.py > logs/e1_pool_refusal.log 2>&1; grep -v -i "warn\|Fetching\|Loading\|legacy" logs/e1_pool_refusal.log; echo POOLREF DONE
