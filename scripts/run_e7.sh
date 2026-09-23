#!/bin/bash
PY="${PY:-python}"
until grep -q "E8 DONE" $W/logs/run_e8.out 2>/dev/null; do sleep 120; done
cd $W
env PYTHONNOUSERSITE=1 ${PY} scripts/e7_quality.py > logs/e7_quality.log 2>&1 && echo E7 DONE || { echo "E7 FAILED"; tail -n 5 logs/e7_quality.log; }
