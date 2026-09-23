#!/bin/bash
PY="${PY:-python}"
until grep -q "E7 DONE\|E7 FAILED" $W/logs/run_e7.out 2>/dev/null; do sleep 120; done
cd $W
env PYTHONNOUSERSITE=1 ${PY} scripts/e12_categories.py > logs/e12_categories.log 2>&1 && echo E12 DONE || { echo "E12 FAILED"; tail -n 5 logs/e12_categories.log; }
