#!/bin/bash
# Overnight: anti-curation at 8B (lambda .25), then E3 at lambda .10 and .30. Detached.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
# anti arm at lambda .25: bottom by the Prompting judge's harm log-odds (highest 'Yes'), size = certlg s0 (2176)
env PYTHONNOUSERSITE=1 $PY1 - <<'PY'
import json, numpy as np
D="$W/selections_e3/lambda0.25"
pool=json.load(open(f"{D}/pool.json")); S=json.load(open(f"{D}/scores.json")); summ=json.load(open(f"{D}/summary.json"))
unsafe=np.array([not r["is_safe"] for r in pool]); pr=np.array(S["prompting_yes_logodds"]); req=np.array(S["harmful_request"]); demo=req&~unsafe
for s in range(3):
    idx=[int(i) for i in np.argsort(-pr)[:2176]]
    meta={"rule":"top 2176 by Prompting 'Yes' log-odds (judge's rejects)","n":2176,"unsafe_frac":float(unsafe[idx].mean()),"safety_demo_share":float(demo[idx].mean())}
    json.dump({"idx":idx,"meta":meta},open(f"{D}/anti_s{s}.json","w")); summ.setdefault("anti",{})[str(s)]=meta
json.dump(summ,open(f"{D}/summary.json","w"),indent=1); print("anti (8B): unsafe %.3f demos %.3f"%(meta["unsafe_frac"],meta["safety_demo_share"]))
PY
for s in 0 1 2; do env PYTHONNOUSERSITE=1 LAMBDA=0.25 ARM=anti SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_anti_s${s}_0.25.log 2>&1 && echo "[$(date +%H:%M)] done anti s$s"; done
env PYTHONNOUSERSITE=1 LAMBDA=0.25 $PY2 scripts/e3_eval.py gen > logs/e3/gen_0.25b.log 2>&1; env PYTHONNOUSERSITE=1 LAMBDA=0.25 $PY1 scripts/e3_eval.py judge > logs/e3/judge_0.25b.log 2>&1
grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_0.25b.log | tail -n 10
sed -i 's/for a in \["base", "full", "prompting", "random", "certlg", "certstrat2", "oracle"\]/for a in ["base", "full", "prompting", "anti", "random", "certlg", "certstrat2", "oracle"]/' scripts/e3_eval.py
for L in 0.10 0.30; do
  export LAMBDA=$L
  env PYTHONNOUSERSITE=1 $PY1 scripts/e3_select.py > logs/e3/select_$L.log 2>&1 || { echo "SELECT $L FAILED"; continue; }
  for s in 0 1 2; do for arm in full prompting random certlg certstrat2 oracle; do
    env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_$L.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s lambda $L" || echo "FAIL $arm s$s $L"
  done; done
  env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_$L.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_$L.log 2>&1
  grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_$L.log | tail -n 10
done
echo E3 SWEEP DONE
