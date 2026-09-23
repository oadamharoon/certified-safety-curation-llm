"""Source trace for the audit: every numeric literal in the paper's prose, with the result files and
keys whose values match it at the stated precision (so a reader can see where each number comes
from). Prints one line per (sentence, literal). Usage: python paper/scripts/trace_numbers.py"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_prose as A

def sources():
    S = {}   # value key -> list of "file:path" strings
    def walk(o, path, f):
        if isinstance(o, bool): return
        if isinstance(o, (int, float)): S.setdefault(float(o), []).append(f"{f}:{path}")
        elif isinstance(o, dict):
            for k, v in o.items(): walk(v, f"{path}/{k}", f)
        elif isinstance(o, list):
            for i, v in enumerate(o[:60]): walk(v, f"{path}[{i}]", f)
    for f in glob.glob(f"{A.W}/results/*.json") + glob.glob(f"{A.W}/selections_e3/*/summary.json"):
        try: walk(json.load(open(f)), "", os.path.relpath(f, A.W))
        except Exception: pass
    for f in glob.glob(f"{A.BASE}/data/tables/*.tex"):
        for ln, line in enumerate(open(f)):
            for x in re.findall(r"-?\d+\.\d+|\b\d{2,5}\b", re.sub(r"\\[a-zA-Z]+", " ", line)):
                S.setdefault(float(x), []).append(f"tables/{os.path.basename(f)}:{ln+1}")
    return S

def main():
    tex = open(f"{A.BASE}/paper.tex").read(); S = sources()
    keys = sorted(S)
    for sent in A.sentences(tex):
        lits = []
        for m in re.finditer(r"(?<![\w.\-])(\d+\.\d+|\d{2,5})(?![\w.%])", sent):
            ctx = sent[max(0, m.start() - 60):m.end() + 30]
            if A.PROTO.search(ctx) or m.group(1) in A.CONST: continue
            if "." not in m.group(1) and (1900 < float(m.group(1)) < 2100): continue
            lits.append(m.group(1))
        if not lits: continue
        print("\n>> " + sent[:150].replace("\n", " "))
        for lit in lits:
            k = len(lit.split(".")[1]) if "." in lit else 0; x = float(lit)
            hits = []
            for v in keys:
                if abs(round(v, k) - x) < 1e-9 or abs(round(abs(v), k) - x) < 1e-9 or (k == 0 and x <= 100 and abs(round(v * 100) - x) < 1e-9):
                    hits += S[v]
            tab = [h for h in hits if h.startswith("tables/")]; res = [h for h in hits if not h.startswith("tables/")]
            print(f"   {lit:>8}  tables:{len(tab)} results:{len(res)}  e.g. {(tab[:2] + res[:3])[:4]}")

if __name__ == "__main__":
    main()
