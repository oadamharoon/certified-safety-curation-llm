"""Prose-number audit for the LLM paper, the port of iclr2027/scripts/audit_prose_numbers.py.

1. ARM-SCOPED EXISTENCE. Every decimal (and 2-5 digit integer) in a prose sentence that names an arm
   must be a value of that arm at the stated precision, where an arm's values are: its rows in every
   generated table (data/tables/*.tex), its per-seed and seed-mean metrics in every results/*.json
   whose keys start with the arm's result prefix, and its selection metadata. A sentence naming
   several arms is checked against their union; one naming none against every value anywhere.
2. AGREEMENT BUCKETS by (arms named, judge named, statistic keyword): more than one distinct value
   in a bucket is listed for review.
3. COUNT PHRASES ("four of the five", "2/3") listed for review.
Exit 1 when any orphan remains. Usage: python paper/scripts/audit_prose.py [--verbose]
"""
import glob, json, os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); W = os.path.dirname(BASE)
# prose name -> result-key prefixes (0.5B: results/models names; 8B: e3 arms) and table row labels
ARMS = {
    "no filter": ["full", "e3:full", "p2_full", "hs:full"], "full pool": ["full", "e3:full", "p2_full", "hs:full"],
    "fixed cutoff": ["heur", "lgfilter", "e3:lgfilter", "hs:lgfilter"], "top 75": ["heur"], "Prompting": ["e3:prompting", "hs:prompting", "hs:certprompt"],
    "judge's rejects": ["anti_qwen", "e3:anti"], "rejects": ["anti_qwen", "e3:anti"], "random": ["random1400", "e3:random", "p2_random"],
    "certified": ["cert400", "certlg400", "cert200", "certbd400", "certsaft05", "u0.25|", "u0.10|", "nat|", "hs:certlg", "hs:certlg400", "hs:cert400", "hs:certprompt", "e3:certlg", "e3:certsaft", "e3:certseal", "e3:certseal_long", "e3:certprompt", "p2_cert400", "p2_certlg400"],
    "stratum": ["certstrat2_lg800", "certstrat_lg", "certstrat_lg800", "certstrat_qwen", "e3:certstrat2", "p2_certstrat2"], "stratified": ["certstrat2_lg800", "certstrat_lg", "certstrat_lg800", "certstrat_qwen", "certstrat2_qwen800", "certstrat_saft05", "e3:certstrat_prompt", "hs:certstrat2", "hs:certstrat2_lg800", "e3:certstrat2", "e3:certstrat_saft", "e3:certstrat_seal", "p2_certstrat2"],
    "labeled": ["labels", "e3:labels", "hs:labels"], "oracle": ["oracle", "e3:oracle", "p2_oracle", "hs:oracle"], "SAFT": ["e3:saft", "e3:saft15", "e3:certsaft", "saft05", "certsaft05", "e3:certstrat_saft", "hs:saft", "hs:saft15"], "Prompting judge": ["e3:certprompt", "e3:prompting", "tension8b"], "SEAL": ["e3:seal", "e3:seal_long", "e3:certseal", "e3:certseal_long", "hs:seal", "hs:seal_long"],
    "base": ["base", "e3:base"], "decile": ["decile"], "dose": ["safedemo"], "safety-demonstration share": ["safedemo"], "harmful examples remain": ["e3:harmtop8", "e3:harmmid8", "e3:harmbot8", "e3:safedemo8_0.37"], "at its size": ["e3:oracle_matched", "e3:certlg", "e3:oracle"],
}
JUDGES = ["beaver-dam", "MD-Judge", "WildGuard", "Llama Guard 3", "XSTest", "ROUGE"]
STATS = ["harmful", "harm", "demonstration", "certif", "labels", "AUROC", "AUC", "kappa", "agreement", "margin", "rate", "refusal", "helpfulness", "threshold", "score"]
CONST = {"0.10", "0.25", "0.30", "0.15", "0.20", "0.85", "0.80", "0.05", "0.50", "0.90", "0.95"}
PROTO = re.compile(r"epochs?|batch|learning rate|tokens?|characters?|rank|dropout|max(imum)? length|percent of the pool|quantiles?|\\times|10\^|pools? of|N = |examples per|draws|prompts|labels|n = ")


def nums(o, acc):
    if isinstance(o, bool): return
    if isinstance(o, (int, float)): acc.append(float(o))
    elif isinstance(o, dict): [nums(v, acc) for v in o.values()]
    elif isinstance(o, list): [nums(v, acc) for v in o]


def collect():
    """arm prefix -> values; '' -> everything."""
    V = {"": []}
    for f in glob.glob(f"{W}/results/*.json") + glob.glob(f"{W}/results/gen_scores/*.json") + glob.glob(f"{W}/selections_e3/*/summary.json"):
        try: d = json.load(open(f))
        except Exception: continue
        is_e3 = "e3_eval" in f or "selections_e3" in f
        nums(d, V[""])
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str):
                        key = ("e3:" if is_e3 else "") + re.sub(r"_s\d+$", "", k)
                        acc = []; nums(v, acc)
                        if acc: V.setdefault(key, []).extend(acc)
                        # per-model seed means for dict-of-seeds layouts
                        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                            keys = set.intersection(*[set(x) for x in v.values()])
                            for kk in keys:
                                xs = [x[kk] for x in v.values() if isinstance(x.get(kk), (int, float)) and not isinstance(x.get(kk), bool)]
                                if len(xs) == len(v): V.setdefault(key, []).append(sum(xs) / len(xs))
                    walk(v)
            elif isinstance(o, list): [walk(v) for v in o]
        walk(d)
    # seed means across models named <arm>_s<k> in flat per-model dicts
    for f in glob.glob(f"{W}/results/*.json"):
        try: d = json.load(open(f))
        except Exception: continue
        src = d.get("per_model", d) if isinstance(d, dict) else {}
        if not isinstance(src, dict): continue
        groups = {}
        for k, v in src.items():
            if isinstance(k, str) and re.search(r"_s\d+$", k) and isinstance(v, dict):
                groups.setdefault(("e3:" if "e3_eval" in f else "") + re.sub(r"_s\d+$", "", k), []).append(v)
        for arm, vs in groups.items():
            for kk in set.intersection(*[set(x) for x in vs]):
                xs = [x[kk] for x in vs if isinstance(x.get(kk), (int, float)) and not isinstance(x.get(kk), bool)]
                if len(xs) == len(vs): V.setdefault(arm, []).append(sum(xs) / len(xs)); V[""].append(sum(xs) / len(xs))
    hs = f"{W}/results/harmsets.json"
    if os.path.exists(hs):
        for k, v in json.load(open(hs)).items():
            acc = []; nums(v, acc); V.setdefault("hs:" + re.sub(r"_s\d+$", "", k.split("/")[1]), []).extend(acc)
    for f in glob.glob(f"{BASE}/data/tables/*.tex"):
        for line in open(f):
            vals = [float(x) for x in re.findall(r"-?\d+\.\d+|\b\d{2,5}\b", re.sub(r"\\[a-zA-Z]+", " ", line))]
            if not vals: continue
            V[""].extend(vals); head = line.split("&")[0].lower()
            for name, prefixes in ARMS.items():
                if name.lower() in head:
                    for p in prefixes: V.setdefault(p, []).extend(vals)
    return V


def supports(lit, vals):
    k = len(lit.split(".")[1]) if "." in lit else 0; x = float(lit)
    return any(abs(round(v, k) - x) < 1e-9 or abs(round(abs(v), k) - x) < 1e-9 or (x <= 100 and abs(round(v * 100, k) - x) < 1e-9) for v in vals)   # percent literals may carry decimals (27.6 percent = .276)


def sentences(tex):
    body = tex[tex.index("\\begin{abstract}"):tex.index("\\bibliography")]
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
    body = re.sub(r"\\(cite|ref|eqref|citep|citet|label|tabinput|includegraphics)\{[^}]*\}", " ", body)
    body = re.sub(r"\$\\pm\$|\\pm", " +- ", body); body = re.sub(r"\\[a-zA-Z]+\*?", " ", body).replace("{", " ").replace("}", " ")
    ln = 0
    for para in body.split("\n"):
        ln += 1
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z(])", para):
            if re.search(r"\d", sent): yield sent.strip()


def main():
    verbose = "--verbose" in sys.argv
    if not os.path.exists(f"{BASE}/paper.tex"):
        print("paper/paper.tex is not in this archive; the prose audit runs in the full repository.")
        return 0
    tex = open(f"{BASE}/paper.tex").read(); V = collect()
    orphans, buckets = [], {}
    for sent in sentences(tex):
        lits = []
        for m in re.finditer(r"(?<![\w.\-])(\d+\.\d+|\d{2,5})(?![\w.%])", sent):
            ctx = sent[max(0, m.start() - 60):m.end() + 30]
            if PROTO.search(ctx) or m.group(1) in CONST: continue
            if "." not in m.group(1) and (1900 < float(m.group(1)) < 2100): continue
            lits.append(m.group(1))
        if not lits: continue
        arms = [a for a in ARMS if a.lower() in sent.lower()]
        pool = []
        for a in arms:
            for p in ARMS[a]:
                for key, vals in V.items():
                    if key == p or key.startswith(p + "_") or (p.endswith("|") and key.startswith(p)) or (p == "decile" and key.startswith("decile")) or (p == "safedemo" and key.startswith("safedemo")): pool += vals
        if not arms: pool = V[""]
        judge = next((j for j in JUDGES if j.lower() in sent.lower()), ""); stat = next((s for s in STATS if s.lower() in sent.lower()), "")
        for lit in lits:
            if not supports(lit, pool):
                orphans.append((lit + ("" if not supports(lit, V[""]) else " (exists elsewhere)"), arms, sent[:120]))
            elif arms:
                buckets.setdefault((tuple(arms), judge, stat), {}).setdefault(lit, 0)
                buckets[(tuple(arms), judge, stat)][lit] += 1
    print(f"== 1. ARM-SCOPED ORPHANS: {len(orphans)}\n")
    for lit, arms, sent in orphans: print(f"  {lit:<8} [{', '.join(arms) or 'no arm'}]  ...{sent}...")
    dis = {k: v for k, v in buckets.items() if len(v) > 1}
    print(f"\n== 2. AGREEMENT BUCKETS with >1 distinct value: {len(dis)} (review)\n")
    for (arms, judge, stat), vals in sorted(dis.items()):
        if len(vals) > 8 and not verbose: continue
        print(f"  [{', '.join(arms)} | {judge or '-'} | {stat or '-'}]  " + "  ".join(sorted(vals)))
    print("\n== 3. COUNT PHRASES\n")
    words = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|all|none"
    for sent in sentences(tex):
        for m in re.finditer(rf"\b((?:{words})|\d+) (?:of|in) (?:the |its |every )?((?:{words})|\d+)\b[^.;]{{0,50}}|\b\d/\d\b[^.;]{{0,40}}", sent, re.I):
            print("  " + m.group(0)[:80])
    print(f"\nGATE: {'CLEAN' if not orphans else f'{len(orphans)} orphan(s)'}")
    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main())
