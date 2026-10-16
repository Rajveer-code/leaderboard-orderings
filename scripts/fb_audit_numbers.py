"""Final number audit (gate 11): no number in the paper is typed by hand.

1. Regeneration: every generated artefact (macros, tables, figures) is rebuilt from the result files
   and must be byte-identical to the committed copy. Slow generators (certification, taxonomy) are
   rebuilt only with --full.
2. The rendered PDF: every number in its prose must be the value of a macro, a number printed by a
   generated table or figure, or a structural reference (sections, figures, tables, equations,
   pages, statements). Everything else is listed; the audit passes when the list is empty.

Run: python scripts/fb_audit_numbers.py [--full]
Out: results/analysis/number_audit.json; exit status 1 on any failure
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "manuscript" / "tmlr"
OUT = ROOT / "results" / "analysis" / "number_audit.json"
FAST = ["fb_macros_aistats.py", "fb_tables_tmlr.py", "fb_provenance_table.py", "fb_figures_aistats.py"]
SLOW = ["fb_certify.py", "fb_extraction_taxonomy.py"]
GENERATED = ["manuscript/tmlr/macros.tex", "manuscript/tmlr/tables.tex", "manuscript/tmlr/shared_table.tex",
             "manuscript/tmlr/sensitivity_table.tex", "manuscript/tmlr/framework_table.tex",
             "manuscript/tmlr/provenance_table.tex", "manuscript/tmlr/taxonomy_table.tex",
             "manuscript/tmlr/gallery_table.tex", "manuscript/tmlr/figures"]
NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
# structural references and conventions that are not results
STRUCTURAL = [r"(?:Section|Sections|Appendix|Appendices|Table|Tables|Figure|Figures|Proposition|Lemma|"
              r"Assumption|Definition|Equation|Step|step)\s+[A-J]?\d+(?:\.\d+)?", r"\(\d+\)", r"^\d+(?:\.\d+)?\s",
              r"^\d+(?:\.\d+)?$", r"Level[- ]\d", r"\b(?:000|111|101|0000|1111)\b"]


def macro_values():
    vals = set()
    for line in (TEX / "macros.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\\w+\}\{(.*)\}$", line)
        if m:
            vals.update(NUM.findall(m.group(1).replace("{,}", ",")))
    return vals


def constant_values():
    """The guard's allowed constants (mathematical values, cell labels, the quoted example) and the
    years of the cited works, which appear in author-year citations."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from fb_check_manuscript import ALLOWED_DIGITS
    years = set(re.findall(r"year\s*=\s*[{\"]?(\d{4})", (TEX / "references.bib").read_text(encoding="utf-8")))
    return set(ALLOWED_DIGITS) | years


def generated_values():
    vals = set()
    for p in TEX.glob("*table*.tex"):
        vals.update(NUM.findall(p.read_text(encoding="utf-8").replace("{,}", ",")))
    import fitz
    for p in (TEX / "figures").glob("*.pdf"):
        for page in fitz.open(p):
            vals.update(NUM.findall(page.get_text()))
    return vals


def regeneration(full):
    steps = (SLOW if full else []) + FAST
    for s in steps:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            return {"ok": False, "failed_step": s, "stderr": r.stderr[-500:]}
    diff = subprocess.run(["git", "status", "--porcelain", "--", *GENERATED], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip().splitlines()
    return {"ok": not diff, "steps": steps, "changed": diff}


BUILDS = ["main", "arxiv", "dmlr"]  # anonymous review, arXiv preprint, DMLR submission


def pdf_prose(name="main"):
    import fitz
    doc = fitz.open(TEX / f"{name}.pdf")
    allowed = macro_values() | generated_values() | constant_values()
    unexplained, in_refs = [], False
    for i, page in enumerate(doc, 1):
        for line in page.get_text().splitlines():
            s = line.strip()
            if s == "References":
                in_refs = True
            if in_refs and s == "Reproducibility":
                in_refs = False
            if in_refs or s == str(i) or s.startswith("©2026"):  # bibliography, page number, DMLR footer
                continue
            masked = s
            for pat in STRUCTURAL:
                masked = re.sub(pat, " ", masked)
            masked = re.sub(r"[A-Za-z]+[-_]?[\w.-]*\d[\w.-]*", " ", masked)  # names such as Qwen2-72B
            for tok in NUM.findall(masked):
                if tok not in allowed:
                    unexplained.append({"pdf": name, "page": i, "token": tok, "line": s[:120]})
    return {"ok": not unexplained, "unexplained": unexplained}


def main():
    full = "--full" in sys.argv
    prose = [pdf_prose(b) for b in BUILDS if (TEX / f"{b}.pdf").exists()]
    out = {"regeneration": regeneration(full),
           "pdf_prose": {"ok": all(x["ok"] for x in prose), "unexplained": [u for x in prose for u in x["unexplained"]]}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("regeneration:", "OK" if out["regeneration"]["ok"] else out["regeneration"])
    u = out["pdf_prose"]["unexplained"]
    print(f"pdf prose: {len(u)} unexplained number(s)")
    for x in u[:40]:
        print(f"   {x['pdf']} p{x['page']}: {x['token']!r} in: {x['line']}")
    return 0 if out["regeneration"]["ok"] and out["pdf_prose"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
