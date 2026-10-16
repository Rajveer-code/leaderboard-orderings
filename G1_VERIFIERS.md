# G1 — verifier provenance, census, and audit mapping

Date: 2026-09-19. Every number below is read from a committed output; its source is named
next to it. The paper takes its numbers from the macro pipeline, not from this file.

## 1. Provenance of the leaderboard's per-item `exact_match` (G1.1)

Source: `results/flipbudget/verifier_provenance.json` (`scripts/fb_verifier_provenance.py`).

**Conclusion (conclusive, documentary + re-execution):** the per-item `exact_match` in the
Open LLM Leaderboard v2 MATH-Hard details files is the leaderboard's **original 2024 scorer**.
That scorer extracts the answer with the Minerva regex ("Final Answer: The final answer is …
I hope it is correct."), applies `normalize_final_answer` to the prediction, and scores
`is_equiv(prediction, doc["answer"])` with sympy (5 s deadline). The paper can name it that way.

Evidence:
- **Documentary.** For each model, the results JSON as first uploaded records the
  `process_results` source that ran. It is byte-identical to upstream lm-eval commit
  `3c8db1bb7b` (2024-07-03, "Adds Open LLM Leaderboard Taks") for all 25 models that have an
  original file. Recorded harness hashes: `8f1dc26`, `60f0679`. These are HF fork commits
  that are no longer resolvable on GitHub, so the upstream copy is the public reference
  (`provenance/verifier/`).
- **Timestamps.** All 27 models' math samples files were last committed in June 2024 and
  never modified. The Feb 2025 Math-Verify rescoring rewrote only the results JSON
  (2025-02-13 commits): scores changed and the task version went 1.0 → 3.0, but the recorded
  config was not updated. Two models (zephyr-7b-alpha, Mistral-7B-v0.1) have no rescored
  results file at the evaluated timestamp.
- **Re-execution.** Re-running the recorded scorer on the published responses reproduces
  the published field on 35,748 / 35,748 items. Per-subject means equal the original
  aggregates in 175 / 175 model-subjects that have an original file.

**Consequence for the third verifier (needs a decision, §5).** Math-Verify 0.9.0, called as
upstream lm-eval `61ee867816` calls it, reproduces the leaderboard's **current (Feb 2025)
aggregates** exactly in 168 of the 175 rescored model-subjects, and each of the other 7
differs by one item. So `mv` is, to within those items, the leaderboard's own current
verifier. It is not a separately implemented third party. The rescoring published no
per-item verdicts, so item-level identity cannot be checked.

## 2. Coverage (G1.2)

Source: `results/flipbudget/coverage_table.json`; rule `G1_DATA_RULES.md` (commit 71ed270,
made before the census). 27/27 models kept, 0 missing rows, 1,324 common items. The 135
rows previously counted as missing were falcon-40b's precalculus file, which was absent
from the local cache but present on the Hub (downloaded; revision recorded in `sources`).

## 3. Frozen verifiers and shared components (G1.3)

Source: `results/flipbudget/verifier_config.json`; pins in `pyproject.toml [verify]`.

| Component | `em` (leaderboard 2024) | `sb` (project, boxed) | `mv` (Math-Verify, leaderboard 2025) |
|---|---|---|---|
| Answer extraction | Minerva "Final Answer" regex | last `\boxed{}`/`\fbox` (lm-eval `last_boxed_only_string`) | Math-Verify LaTeX + expression extraction, whole response |
| Prediction normalization | `normalize_final_answer` | `normalize_final_answer` | latex2sympy2_extended normalization |
| Gold | `doc["answer"]` (dataset field, already normalized) | `doc["answer"]`, normalized again | parsed from `doc["solution"]` |
| Equivalence | sympy `parse_latex`, `simplify(diff)==0`; no string shortcut | exact string, else sympy `parse_latex` + `simplify` | `math_verify.verify` (sympy-based; rounding 6, precision 15) |
| Deadline | 5 s | 1 s | 5 s per parse / verify |
| Shared with others | normalizer, `parse_latex`, sympy (with `sb`) | normalizer, `parse_latex`, sympy (with `em`) | sympy core and antlr4 runtime only |
| Independent parts | extraction regex | boxed extraction, string shortcut | extraction, parser (latex2sympy2), gold source, comparison logic |

Paper wording (rule 6): `mv` is "a separately implemented verifier with a distinct
extraction and parsing stack". It shares sympy with both others, and it is the verifier
behind the leaderboard's current scores. Neither `mv` nor the others are statistically
independent.

## 4. The Windows `is_equiv` defect (found in G1)

`is_equiv` guards its sympy call with `signal.SIGALRM`, which does not exist on Windows. It
swallows the resulting error and returns False. The thread-pool wrappers in
`fb_e4_mathhard.py` and knowledgeshift `scripts/19` still call the original function, so they
do not fix this (checked: `is_equiv(\frac{1}{2}, \frac{2}{4})` → False). As a result,
**every `score_boxed` verdict and every human-reference match computed before 2026-09-19 is
normalized string equality, not the harness's sympy equivalence.** Historical results stay
internally consistent (all sides share the defect), which is why earlier cross-checks passed.

`src/flipbudget/verifiers.py` runs the original code with the signal guard replaced by
timing. A call slower than its original deadline gets the original timeout verdict. Timeouts
in the census: `em_rec` 2, `sb` 4, `mv` 6 of 35,748 (3 killed by the 60 s watchdog).

Impact on the 17-model K=2 anchor (`results/flipbudget/g1_semantics_impact.json`, same code):

| Semantics | sb≠em rows | audit-only | combined | sampling ceiling |
|---|---|---|---|---|
| string (historical) | 817 | 101 | 84 | 118 |
| sympy (faithful) | 704 | 104 | 89 | 117 |

## 5. Audit mapping (G1.5) and regression anchor (G1.6)

Source: `results/flipbudget/audit_k3_map.json`. All 400 audit items map to census rows.
Human reference Y* under three routines: string vs sympy differ on 1 item; string vs
Math-Verify on 5 (two prose transcriptions that Math-Verify extracts, one Math-Verify false
positive on "base 10", one pmatrix transcription, one sympy-only equivalence).

**The dataset's `doc["answer"]` is itself pre-normalized** (`\infty` → `\iny`,
`\left(` → `\le(`, `12, 10, 6` → `12106`). String and sympy references re-normalize, so this
is harmless for them. The Math-Verify reference therefore parses gold from the solution.

The existing audit covers only K=3 cells 000, 011, 111 (`em, sb_sympy, mv`). Cells 001, 010,
100, 101, 110 have no human labels. This is an input to G3.

Anchor: `tests/test_regression_anchor.py` rebuilds the old inputs from the new census and
reproduces 817 disagreements, cells 142/32/1/145 (141/16/0/0 correct), and
84 / 101 / 118 under both indeterminate rules. It passes locally. CI:
`.github/workflows/tests.yml` (not yet pushed).

## Decisions (author, 2026-09-19): all three recommendations adopted

1. Primary `sb` semantics: **sympy** (faithful). `sb_string` is kept only as the
   historical reconstruction and for the regression anchor.
2. `mv` is described as **the leaderboard's own 2025 verifier** (Math-Verify). The setting
   is "the leaderboard's 2024 and 2025 verifiers plus a boxed verifier". Independence
   wording follows the shared-component table.
3. Primary human reference: **harness routine (sympy)**. Math-Verify reference is a
   sensitivity analysis.

## Decisions requested (kept for the record)

1. **Primary `sb` semantics:** faithful sympy (recommended; it is the harness as written)
   or historical string (reproduces earlier numbers). Every earlier headline built on
   `score_boxed` (3.63%, 8 reversals, 84) must be recomputed under the chosen semantics.
2. **Framing of `mv`:** it is the leaderboard's own current verifier. Recommended: say so,
   and treat it as a strength ("the leaderboard's 2024 and 2025 verifiers plus a boxed
   verifier"). Plan rule 6's "separately implemented" is accurate only for the parsing stack.
3. **Primary human reference:** the harness routine (sympy) is recommended. Math-Verify on
   free-text transcriptions misfires (§5); report it as a sensitivity.
