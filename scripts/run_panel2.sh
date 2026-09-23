#!/bin/bash
# waits for E4 to release the GPU, then runs the two BeaverTails-free judges on everything
PY="${PY:-python}"
until grep -q "E4 DONE" /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/tasks/bjoh7zo9w.output 2>/dev/null; do sleep 30; done
cd $W
env PYTHONNOUSERSITE=1 ${PY} scripts/e1_panel2.py > logs/e1_panel2.log 2>&1; grep -v -i "warn\|Fetching\|Loading\|legacy" logs/e1_panel2.log; echo PANEL2 DONE
env PYTHONNOUSERSITE=1 ${PY} scripts/e1_pool_refusal.py > logs/e1_pool_refusal.log 2>&1; grep -v -i "warn\|Fetching\|Loading\|legacy" logs/e1_pool_refusal.log; echo POOLREF DONE
