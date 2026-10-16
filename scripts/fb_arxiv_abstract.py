"""Print the abstract as plain text for the arXiv metadata form (macros expanded, no LaTeX).

arXiv's abstract field takes plain text up to 1,920 characters; this expands the paper's own macros
so the form abstract cannot drift from the PDF.  Run: python scripts/fb_arxiv_abstract.py
"""
import re
from pathlib import Path

MS = Path(__file__).resolve().parents[1] / "manuscript" / "tmlr"
LIMIT = 1920

macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (MS / "macros.tex").read_text(encoding="utf-8")))
text = (MS / "00_abstract.tex").read_text(encoding="utf-8")
text = re.sub(r"\\(begin|end)\{abstract\}", "", text)
text = re.sub(r"\\(\w+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), text)
text = re.sub(r"\\(\w+)", lambda m: macros.get(m.group(1), m.group(0)), text)
text = text.replace("{,}", ",").replace("\\%", "%").replace("\\varepsilon", "epsilon")
text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text).replace("$", "")
paras = [" ".join(p.split()) for p in text.strip().split("\n\n")]
out = "\n".join(paras)
assert "\\" not in out, out
print(out)
print(f"\n[{len(out)} characters; arXiv limit {LIMIT}]")
assert len(out) <= LIMIT
