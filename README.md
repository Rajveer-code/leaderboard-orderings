# Which Leaderboard Orderings Survive a Human Reader?

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
