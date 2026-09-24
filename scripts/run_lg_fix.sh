#!/bin/bash
PY="${PY:-python}"
# Run this after the Llama Guard judging pass: both want the GPU to itself.
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$W"
env PYTHONNOUSERSITE=1 ${PY} scripts/debug_llamaguard.py 2>&1 | grep -v -i "warn\|Loading\|Fetching"; echo LGDEBUG DONE
