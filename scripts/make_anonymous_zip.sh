#!/bin/bash
# Build the anonymous code archive for the ICLR supplementary upload.
#
# It is the repository's tracked codebase (`git ls-files`) minus the LaTeX paper, which is not code
# and is submitted separately, with the authorship block stripped from README.md. A scrub step fails
# the build if any author name, institution, e-mail or account string survives anywhere.
#
# Usage: bash scripts/make_anonymous_zip.sh [outdir]      default outdir: the repository's parent
set -euo pipefail
ROOT="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${1:-$(dirname "$ROOT")}"
NAME="certified-safety-curation-llm-code"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
cd "$ROOT"
command -v git >/dev/null || { echo "git is required"; exit 1; }
git rev-parse --verify HEAD >/dev/null 2>&1 || { echo "commit the repository first"; exit 1; }

mkdir -p "$DEST"
git ls-files -z | while IFS= read -r -d '' f; do
  # The LaTeX paper is not code and is submitted separately. Match only files directly in paper/,
  # since a shell glob's * spans slashes and would otherwise swallow paper/data/tables/*.tex.
  if [ "$(dirname "$f")" = "paper" ]; then
    case "$f" in *.tex|*.bib|*.sty|*.bst) continue ;; esac
  fi
  case "$f" in
    scripts/make_anonymous_zip.sh) continue ;;                     # this builder greps for the names
    README.md) continue ;;                                         # handled below
  esac
  mkdir -p "$DEST/$(dirname "$f")"; cp "$f" "$DEST/$f"
done

# README without the authorship block
python - "$ROOT/README.md" "$DEST/README.md" <<'EOF'
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
t = re.sub(r"<!-- AUTHORSHIP:.*?<!-- END AUTHORSHIP -->\n+", "", t, flags=re.S)
open(sys.argv[2], "w", encoding="utf-8").write(t)
EOF

# Scrub: nothing identifying may survive anywhere in the archive.
python - "$DEST" <<'EOF' || exit 1
import os, re, sys
stage = sys.argv[1]
# Our names, institution and accounts. E-mail-shaped strings that occur inside the redistributed
# dataset text (BeaverTails and PKU-SafeRLHF prompts) are third-party dataset content, not ours,
# so only our own domain and handles are treated as identifying.
PAT = re.compile(r"haroon|fleming|iastate|iowa state|ames,\s*ia|oadamharoon|aharoon", re.I)
bad, n = [], 0
for root, _, files in os.walk(stage):
    for f in files:
        p = os.path.join(root, f); n += 1
        try: txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        bad += [(os.path.relpath(p, stage), m.group(0)) for m in PAT.finditer(txt)]
if bad:
    print("SCRUB FAILED -- identifying strings in the archive:")
    for rel, s in bad[:25]: print("   ", rel, "->", s)
    sys.exit(1)
print("scrub clean: no author, institution, e-mail or account string in %d files" % n)
EOF

mkdir -p "$OUT"; rm -f "$OUT/$NAME.zip"
( cd "$STAGE" && zip -qr "$OUT/$NAME.zip" "$NAME" )
echo "archive: $OUT/$NAME.zip  ($(du -sh "$OUT/$NAME.zip" | cut -f1), $(find "$DEST" -type f | wc -l) files)"
