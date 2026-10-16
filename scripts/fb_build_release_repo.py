"""Build the public release repository (github.com/Rajveer-code/leaderboard-orderings).

Same file set as the review supplement (scripts/fb_build_anonymous_supplement.py), so it holds
the code, data, results and manuscript and none of the internal planning notes, but with the
author's name kept and the author-visible builds (arXiv, DMLR) included. Local home-directory
paths are replaced so no machine path is published. The folder is written fresh each run; it
does not touch this repository's history.

Usage: python scripts/fb_build_release_repo.py [out_dir]
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_build_anonymous_supplement as supp  # noqa: E402 -- reuse its manifest, not copy it

ROOT = supp.ROOT
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "leaderboard-orderings"
NAMED_BUILDS = {"manuscript/tmlr/arxiv.tex", "manuscript/tmlr/arxiv.pdf", "manuscript/tmlr/dmlr.tex",
                "manuscript/tmlr/dmlr.pdf", "manuscript/tmlr/dmlr2e.sty",
                "manuscript/tmlr/dmlr_data_release.tex", "scripts/fb_build_arxiv_tarball.py",
                "scripts/fb_arxiv_abstract.py", "scripts/fb_build_release_repo.py"}
EXCLUDE = (supp.EXCLUDE - NAMED_BUILDS) | {"scripts/fb_build_anonymous_supplement.py"}
HOME = re.compile(r"[A-Za-z]:(\\+|/)Users(\\+|/)[^\\/\"'\s]+")
PRIVATE = [r"Users[\\/]+Asus", r"LABELING_MASTER_PROMPT", r"MASTER_PROMPT\.md"]

README = """# Which Leaderboard Orderings Survive a Human Reader?

Code, data and results for the paper by Rajveer Singh Pall (Gyan Ganga Institute of Technology
and Sciences, Jabalpur).

A leaderboard score for a free-text benchmark depends on the model and on the *verifier*, the
program that finds the answer in each response. On the Open LLM Leaderboard's own records,
changing the verifier on unchanged responses reversed 15.9% of the pairwise orderings among
3,366 models. On the archived MATH-Hard board (27 models, 35,748 responses, three verifiers and a
400-response human audit), this repository bounds which orderings a human reader supports: the
audit alone certifies none, and under a stated bound on how much models differ where the verifiers
agree, 174 of 351 are certified at the reference tolerance and 208 when models are taken as alike.

## Contents

- `manuscript/tmlr/` -- the paper: `arxiv.tex` (preprint), `dmlr.tex` (DMLR submission),
  `main.tex` (anonymous review build). `macros.tex` and the `*_table.tex` files are generated.
- `scripts/`, `src/`, `tests/` -- the pipeline and its checks (`python -m pytest -q`).
- `results/` -- every result file the paper's numbers are read from; `CLAIM_LEDGER_AISTATS.md`
  maps each macro to its source.
- `results/flipbudget/census_k3.json.gz` -- every verifier's verdict on all 35,748 responses,
  identified by model and item.
- `results/analysis/mathhard_l1.csv` (and `_l2`, `_l3` for the overlaps) -- the 400-item audit's
  human labels, the only labels any result uses.
- `labeling/` -- labelling sheets, the offline labelling app and the rulebooks, including those of
  larger audits that were designed but not labelled and enter no result.
- `TC_EXPANSION_PREREGISTRATION.md` -- the analysis plan with its dated amendments. Despite the
  file name it was never lodged with an external registry, and it was first written after the
  400-item audit's labels existed (paper, Appendix Table 6).

## Reproduce

    pip install -e .
    python scripts/fb_certify.py
    python scripts/fb_figures_aistats.py
    python scripts/fb_test_census.py
    python scripts/fb_macros_aistats.py
    python scripts/fb_tables_tmlr.py
    python scripts/fb_check_manuscript.py
    cd manuscript/tmlr && latexmk -pdf arxiv.tex

## Data sources and licences

Problems and reference solutions: MATH-Hard (`lighteval/MATH-Hard`, MIT licence). Model responses:
the Open LLM Leaderboard's public per-model details datasets on the Hugging Face Hub, whose cards
declare no licence; this repository copies only the responses shown to labellers, for research
use, and refers to the rest by identifier. Code, verdicts and labels: Apache 2.0 (`LICENSE`).

The version behind the DMLR submission is tagged `dmlr-submission`.
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    files = [p for p in supp.tracked()
             if (p in supp.INCLUDE_FILES or any(p.startswith(d + "/") for d in supp.INCLUDE_DIRS)
                 or p in NAMED_BUILDS)
             and p not in EXCLUDE and not p.endswith(supp.EXCLUDE_SUFFIXES)]
    for p in files:
        src, dst = ROOT / p, OUT / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in supp.TEXT_SUFFIXES:
            try:
                dst.write_text(HOME.sub("<home>", src.read_text(encoding="utf-8")), encoding="utf-8")
                continue
            except UnicodeDecodeError:
                pass
        shutil.copy2(src, dst)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitignore").write_text((ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8")

    hits = []
    for f in OUT.rglob("*"):
        if f.is_file() and f.suffix in supp.TEXT_SUFFIXES and f.name != Path(__file__).name:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            hits += [f"{f.relative_to(OUT)}: {pat}" for pat in PRIVATE if re.search(pat, text)]
    print(f"built {OUT}: {len(files) + 2} files; {len(hits)} private strings")
    for h in hits[:20]:
        print("  ", h)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
