"""Cheap LaTeX sanity checks (no TeX installation needed): environment balance, braces, bib keys, refs."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
tables = "\n".join(f.read_text(encoding="utf-8") for f in sorted((ROOT / "paper").glob("tables*.tex")))
bib = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
text = main + "\n" + tables

ok = True
b, e = Counter(re.findall(r"\\begin\{(\w+\*?)\}", text)), Counter(re.findall(r"\\end\{(\w+\*?)\}", text))
if b != e:
    ok = False
    print("environment mismatch:", b - e, e - b)
body = re.sub(r"\\verb.*|\\begin\{verbatim\}.*?\\end\{verbatim\}", "", text, flags=re.S)
body = re.sub(r"\\[{}]", "", body)
if body.count("{") != body.count("}"):
    ok = False
    print("brace imbalance:", body.count("{") - body.count("}"))
keys = set()
for c in re.findall(r"\\cite[tp]?\{([^}]+)\}", text):
    keys |= {k.strip() for k in c.split(",")}
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
if keys - bibkeys:
    ok = False
    print("missing bib entries:", sorted(keys - bibkeys))
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
if refs - labels:
    ok = False
    print("undefined refs:", sorted(refs - labels))
print("unused bib entries:", sorted(bibkeys - keys))
print("TBD markers:", len(re.findall(r"\\TBD\{", main)))
print("OK" if ok else "PROBLEMS FOUND")
sys.exit(0 if ok else 1)
