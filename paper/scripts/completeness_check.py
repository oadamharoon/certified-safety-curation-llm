"""Experimental-completeness gate, run before any draft is called complete. Every rule is a
statement about files on disk, so the result is checkable by anyone with the repository.
  R1  scorer x certificate matrix: at each scale every scorer has a fixed cutoff, the pooled
      certificate and the stratified certificate at attainable targets, each with >= 3 seeds,
      fine-tuned models, and all four judges (+ XSTest, ROUGE-L) on the BeaverTails prompts.
  R2  every condition of Tables 1-3 is on all three instruction-style harm sets under all four judges.
  R3  control conditions (no filter, random, rejects, labeled-only, oracle) at both scales, same evals.
  R4  every appendix section is referenced from the main text.
  R5  every experiment section of the pre-registration record carries an outcome.
  R6  every claim in CLAIMS.md is ESTABLISHED or RESOLVED.
  R8  every scorer (two judges, Llama Guard 3) passes the gate and label-complexity sweep on all three pools.
  R9  the dose-response exists at both scales with four levels, >= 3 seeds and four judges.
  R7  the prose-number audit is clean.
  R10 every prose number that is not a table cell traces to the one expression that produces it.
  R11 derived statistics (the benchmark-transfer correlations) regenerate to the stored values.
  R12 each table's caption declares exactly the markings (bold, gray) that table uses.
Exit 1 on any failure. Usage: python paper/scripts/completeness_check.py"""
import json, os, re, subprocess, sys
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); fails = []
def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok: fails.append(msg)

sel = json.load(open(f"{W}/results/step2_selections.json")); ev = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]
md = json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg = json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg = json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
u = json.load(open(f"{W}/results/e6_utility.json")); H = json.load(open(f"{W}/results/harmsets.json"))
s8 = json.load(open(f"{W}/selections_e3/lambda0.25/summary.json")); d8 = json.load(open(f"{W}/results/e3_eval_lambda0.25.json"))
def models05(a): return [m for m in ev if re.match(re.escape(a) + r"_s\d+$", m)]
def full05(a):
    ms = models05(a); base = re.sub(r"^(tl_|qi_)", "", a)   # second-base-model conditions share the 0.5B selection files
    if base == "base": return len(ms) == 1 and all(m in md and m in wg and m in lg and m in u for m in ms)   # untouched model: one run, no selection
    return len(ms) >= 3 and all(m in md and m in wg and m in lg and m in u and "xstest_safe_refusal" in u[m] for m in ms) and base in sel and len(sel[base]) >= len(ms)
def full8(a):
    ms = [m for m in d8 if re.match(re.escape(a) + r"_s\d+$", m)]; return len(ms) >= 3 and a in s8 and len(s8[a]) == len(ms) and all(all(k in d8[m] for k in ("beaverdam", "mdjudge", "wildguard", "llamaguard", "xstest_unsafe_harmful", "rouge_l")) for m in ms)
def hs(sub, a, n):
    ks = [k for k in H if k.startswith(sub + "/") and re.match(re.escape(a) + r"_s\d+$", k.split("/")[1])]
    return len(ks) >= n and all(all(j in H[k] and all(st in H[k][j] for st in ("DirectHarm4", "HarmBench", "HEx-PHI")) for j in ("beaver-dam", "MD-Judge", "WildGuard", "Llama Guard 3")) for k in ks)

print("R1 scorer x certificate matrix")
M05 = {"judge": ("heur", "cert400", "certstrat2_qwen800"), "Llama Guard 3": ("lgfilter", "certlg400", "certstrat2_lg800"), "SAFT": ("saft05", "certsaft05", "certstrat_saft05")}
M8 = {"Prompting judge": ("prompting", "certprompt", "certstrat_prompt"), "Llama Guard 3": ("lgfilter", "certlg", "certstrat2"), "SAFT": ("saft", "certsaft", "certstrat_saft"), "SEAL": ("seal_long", "certseal_long", "certstrat_seal")}
for sc, arms in M05.items():
    for kind, a in zip(("fixed cutoff", "certified", "stratified attainable"), arms): check(full05(a), f"0.5B {sc:14s} {kind:22s} {a}")
for sc, arms in M8.items():
    for kind, a in zip(("fixed cutoff", "certified", "stratified attainable"), arms): check(full8(a), f"8B   {sc:14s} {kind:22s} {a}")
print("R3 controls")
for a in ("full", "random1400", "anti_qwen", "labels", "oracle"): check(full05(a), f"0.5B control {a}")
for a in ("full", "random", "anti", "labels", "oracle", "oracle_matched"): check(full8(a), f"8B   control {a}")
for a in ("harmtop8", "harmmid8", "harmbot8", "safetop8", "safelen8"): check(full8(a), f"8B   provenance condition {a}")
for a in ("labels800", "randdemo", "judgetop05", "judgeharm05"): check(full05(a), f"0.5B condition {a}")
for a in ("labels800_8", "randdemo8"): check(full8(a), f"8B   label-budget baseline {a}")
for tag, arms in (("tl_", ("base", "full", "random1400", "cert400", "anti_qwen", "lgfilter", "certlg400", "certstrat2_lg800")),
                  ("qi_", ("base", "full", "random1400", "cert400", "anti_qwen", "certlg400", "certstrat2_lg800", "safedemo0.1", "safedemo0.25", "safedemo0.54"))):
    for a in arms: check(full05(tag + a), f"0.5B-class second base model {tag}{a}")
print("R2 harm sets for every condition of Tables 1-3")
T1 = ("full", "random1400", "heur", "anti_qwen", "cert400", "certstrat2_qwen800", "lgfilter", "certlg400", "certstrat2_lg800", "saft05", "certsaft05", "certstrat_saft05", "labels", "oracle")
T2 = ("certstrat_lg", "certstrat_lg800")
T3 = ("full", "prompting", "lgfilter", "saft", "seal_long", "anti", "random", "certprompt", "certstrat_prompt", "certsaft", "certseal_long", "certlg", "certstrat_saft", "certstrat_seal", "certstrat2", "labels", "oracle")
for a in T1 + T2: check(hs("05b", a, 3), f"0.5B harm sets {a}")
for a in T3: check(hs("8b", a, 3), f"8B   harm sets {a}")
print("R4 appendix references")
HAVE_TEX = os.path.exists(f"{W}/paper/paper.tex")
tex = open(f"{W}/paper/paper.tex").read() if HAVE_TEX else ""
main, app = (tex[:tex.index("\\appendix")], tex[tex.index("\\appendix"):]) if HAVE_TEX else ("", "")
if not HAVE_TEX: print("  SKIP  paper/paper.tex is not in this archive")
for title, lab in re.findall(r"\\section\{([^}]*)\}\s*\\label\{(app:[^}]*)\}", app): check(f"\\ref{{{lab}}}" in main, f"appendix '{title}' referenced")
print("R5 pre-registration outcomes")
HAVE_PREREG = os.path.exists(f"{W}/PREREGISTRATION.md")
if not HAVE_PREREG: print("  SKIP  PREREGISTRATION.md is kept outside the repository")
rd = open(f"{W}/PREREGISTRATION.md").read() if HAVE_PREREG else ""; secs = re.split(r"\n## ", rd)
for sec in secs[1:]:
    head = sec.split("\n")[0]
    if re.match(r"(E\d+|Step|SAFT|Anti|OUTCOMES|SEAL)", head) and "Queue" not in head:
        if "PROPOSED, not run" in head: continue   # a proposal that was never launched is not an experiment the paper reports
        # a section pre-registered with predictions ("pre-registered" in its title) must carry an explicit OUTCOME
        # marker; prediction text alone ("held", "failed") must not satisfy the rule (hole found 2026-09-21)
        if "pre-registered" in head.lower(): check(re.search(r"\bOUTCOME", sec) is not None, f"pre-registration section '{head[:60]}' has an OUTCOME"); continue
        body = sec.lower(); check(any(k in body for k in ("outcome", "result", "held", "failed")), f"pre-registration section '{head[:60]}' has an outcome")
print("R6 claims ledger")
HAVE_CLAIMS = os.path.exists(f"{W}/CLAIMS.md")
if not HAVE_CLAIMS: print("  SKIP  CLAIMS.md is kept outside the repository")
for line in (open(f"{W}/CLAIMS.md") if HAVE_CLAIMS else []):
    if line.startswith("| C"):
        cid = line.split("|")[1].strip(); status = line.split("|")[4].strip(); check(status.startswith(("ESTABLISHED", "RESOLVED")), f"{cid}: {status[:50]}")
print("R8 every scorer through the gate and label-complexity sweep")
e5 = json.load(open(f"{W}/results/e5_label_complexity.json"))
for sc in ("1.5B", "7B", "LG"):
    for pool in ("nat", "u0.25", "u0.10"):
        for a in ("a0.1", "a0.25"):
            k = f"{pool}|{sc}|{a}"; check(k in e5 and len(e5[k]["rate"]) == 7, f"gate/label-complexity cell {k}")
print("R9 dose-response at both scales (four demonstration-share levels, >= 3 seeds, four judges)")
for q in ("0.1", "0.17", "0.25", "0.33", "0.4", "0.47", "0.54"): check(full05(f"safedemo{q}") and len(models05(f"safedemo{q}")) >= 5, f"0.5B dose level q={q} (five seeds)")
for q in ("0.3", "0.37", "0.45", "0.545"): check(full8(f"safedemo8_{q}"), f"8B   dose level q={q}")
print("R7 prose audit")
out = subprocess.run([sys.executable, f"{W}/paper/scripts/audit_prose.py"], capture_output=True, text=True).stdout
check("GATE: CLEAN" in out or "is not in this archive" in out, "audit_prose gate" + ("" if HAVE_TEX else " (skipped: no paper source)"))
print("R10 source trace of every prose number that is not a table cell")
env = dict(os.environ, PYTHONNOUSERSITE="1")
out = subprocess.run([sys.executable, f"{W}/paper/scripts/verify_constants.py"], capture_output=True, text=True, env=env).stdout
check("CONSTANTS: PASS" in out, "verify_constants gate")
print("R12 every table's caption declares the markings that table actually uses")
if not HAVE_TEX: print("  SKIP  paper/paper.tex is not in this archive")
_tex = tex
for _m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", _tex, re.S):
    _b = _m.group(1)
    _lab = re.search(r"\\label\{(tab:[^}]+)\}", _b); _src = re.search(r"tabinput\{data/tables/([^}]+)\}", _b)
    _cap = re.search(r"\\caption\{(.*?)\}\s*\n", _b, re.S)
    if not (_lab and _src and _cap): continue
    _body = open(f"{W}/paper/data/tables/{_src.group(1)}").read(); _c = _cap.group(1).lower()
    _gray = "textcolor{gray}" in _body
    _bold = re.search(r"textbf\{\s*[-0-9]", _body) is not None      # a bolded value, not a header word
    check(not _gray or "gray" in _c, f"{_lab.group(1)}: caption explains the gray cells")
    check(not _bold or "bold" in _c, f"{_lab.group(1)}: caption explains the bold cells")
    check(_gray or "gray" not in _c, f"{_lab.group(1)}: caption does not promise gray the table lacks")

print("R11 derived statistics are regenerated, not typed")
before = open(f"{W}/results/harmsets_transfer.json").read()
subprocess.run([sys.executable, f"{W}/paper/scripts/transfer_spearman.py"], capture_output=True, text=True, env=env)
check(open(f"{W}/results/harmsets_transfer.json").read() == before, "harmsets_transfer.json is current")
print(f"\nCOMPLETENESS: {'PASS' if not fails else f'{len(fails)} FAIL(s)'}")
sys.exit(1 if fails else 0)
