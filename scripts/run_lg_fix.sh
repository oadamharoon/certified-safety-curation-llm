#!/bin/bash
PY="${PY:-python}"
until grep -q "WG DONE" /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/tasks/b6awzhuku.output 2>/dev/null; do sleep 30; done
cd $W
env PYTHONNOUSERSITE=1 ${PY} scripts/debug_llamaguard.py 2>&1 | grep -v -i "warn\|Loading\|Fetching"; echo LGDEBUG DONE
