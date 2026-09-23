#!/bin/bash
# LLM-data variant step 1 grid: pool contamination x judge strength.
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}
for J in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  for U in nat 0.25 0.10; do
    tag=$(basename $J)_$U
    if [ $U = nat ]; then env PYTHONNOUSERSITE=1 LLM_JUDGE=$J $PY $W/scripts/step1_certify_gate.py > $W/logs/certify_$tag.log 2>&1
    else env PYTHONNOUSERSITE=1 LLM_JUDGE=$J LLM_POOL_UNSAFE=$U $PY $W/scripts/step1_certify_gate.py > $W/logs/certify_$tag.log 2>&1; fi
    echo "== $tag =="; grep -v -i "warn\|token\|scored\|deprecated" $W/logs/certify_$tag.log
  done
done
echo LLM GRID DONE
