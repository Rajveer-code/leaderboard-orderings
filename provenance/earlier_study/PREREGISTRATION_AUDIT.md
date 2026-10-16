# PLAN_TMLR T4.1 — Pre-registration: the 400-item human MATH-Hard audit

> **Terminology note (added 2026-09-08, after this document was frozen; the commitments below are
> unchanged).** This file's title and the word "pre-registration" throughout describe what the
> design was called internally when it was written. The manuscript deliberately does **not** use
> that word for it, and says so explicitly: the design was *pre-specified and frozen before
> labelling*, committed to version control in its own commit before any item was drawn or labelled,
> but it was **not** lodged with an external registry such as OSF or AsPredicted. "Pre-registered"
> conventionally implies the latter, so the paper claims only what the mechanism actually supports.
> This note is appended rather than applied by rewriting the text below, because the value of this
> document is that its contents are fixed at a verifiable point in the commit history — silently
> editing a frozen commitment would defeat its purpose.

Committed before any item is drawn, before any labeller sees any response, and before
`scripts/63_make_mathhard_sheets.py` (T4.2, not yet written) runs. Follows the convention of the
project's own `PREREGISTRATION.md` (frozen 2026-08-06): **never rewrite the hypothesis, the strata,
or the sample size after seeing results.**

## Why this exists

The paper's headline (§ identification, 90.2% / 92.6% joint-95% of pairwise MATH-Hard orderings
not identified) rests on margins φ̄ and ψ measured by **two LLM adjudicators and zero humans**
(`results/analysis/margins_math_hard.json`). The existing human channel (48 items, 3 labellers,
Fleiss κ = 0.979, `docs/…`) is MMLU-only. This is the single strongest reviewer attack the paper
does not yet close. This document commits the design for closing it before any labelling starts.

## Target population and frame

`scripts/62_mathhard_frame.py` computed, from `results/cache_l1` (the raw Open LLM Leaderboard
generations already used for the paper's 28-model result), every `(model, item)` pair's:

- **stratum**, via the harness's own boxed extractor (`scripts/19::score_boxed`): `credited` (boxed
  found, scored correct — the φ̄/false-credit cell), `wrong_parsed` (boxed found, scored incorrect —
  the ψ/wrong-span cell), or `unparsed` (no boxed answer found — the U/n cell; humans do not
  estimate U/n itself, which is exact and observable, but a small check here verifies the
  extractor's own ⊥-detection is accurate rather than a bug that misses a real boxed answer);
- **subject**, from the cache filename (`math_<subject>_hard`), MATH's own 7-way taxonomy;
- **length quartile**, of the response's character count, computed within each `(stratum, subject)`
  cell so one subject's typically longer proofs cannot dominate every cell's top bin.

Population: **35,613** `(model, item)` pairs with a resolvable stratum, subject, and file (28
models attempted; models whose cached files did not resolve to a MATH-subject file are excluded
from the frame, same as they are from every other MATH-Hard analysis in this paper).

| Stratum | Population | Purpose |
|---|---|---|
| `credited` | 2,118 | estimate φ̄ (false-credit rate) with a human primary channel |
| `wrong_parsed` | 16,714 | estimate ψ (wrong-span rate) with a human primary channel |
| `unparsed` | 16,781 | confirm the boxed extractor's own ⊥-detection is accurate (smaller check, not a margin estimate) |

This population-count computation is descriptive bookkeeping over data already on disk — it does
not draw, preview, or show any individual item, so computing it before this document is committed
does not leak anything about what a labeller will see. What must not happen before this commit,
and has not: drawing the actual 400 item_ids.

## Sample size and allocation

**400 primary items total**: 175 credited, 175 wrong_parsed, 50 unparsed. Credited and
wrong_parsed are weighted evenly and over-weighted relative to unparsed — deliberately, matching
the asymmetry already in the LLM-adjudicated design (585 checked credited, 546 wrong_parsed, 0
unparsed): these two strata are where the paper's estimated margins live, and the unparsed check is
confirmatory rather than a margin estimate.

Within each stratum, allocation across the 28 `(subject × length-quartile)` cells is
**proportional** to each cell's population share, rounded to the nearest integer with leftover
units (from rounding) assigned to the cells with the largest fractional remainder — a standard,
auditable rounding rule, not a discretionary one. Exact per-cell targets and population counts:
`results/analysis/mathhard_frame.json`.

**Design weights**: each sampled item's design weight is `population_in_cell / sample_target_in_cell`
(the inverse of its inclusion probability), so a design-weighted estimate of φ̄ or ψ over the sample
generalizes to the full population rather than only to the sample's own composition.

## Labelling protocol

Reuses the existing blind protocol (`scripts/23_make_labelling_sheet.py`'s pattern, extended for
MATH-Hard by `scripts/63_make_mathhard_sheets.py`, not yet written): the labeller sees the
question and the model's response. They do **not** see the gold answer, the boxed extractor's
output, or which stratum an item belongs to — strata are interleaved so a labeller cannot infer the
condition from position, matching the MMLU protocol's own rationale (Appendix D). The question
asked is **"which final answer does this response commit to?"**, never "is this correct?" — the
same distinction that makes the existing estimate usable rather than a measure of the extractor's
compliance with itself.

**Carrying forward a known failure mode:** the previous labelling round found 3 of 48 sheets
corrupted by a task-collision (a different task's cached generation shown under the wrong label,
`labelling_sheet_audit.json`). `scripts/63` must include a positive-control self-test asserting
every sheet's `(item_id, response_text)` pair matches its declared `(model, item_id)` source before
any sheet ships, so this cannot recur silently. This is a verification requirement on T4.2, stated
here so it is committed before, not discovered after.

## Overlap design (independent double/triple labelling)

| Labels | Who | Purpose |
|---|---|---|
| 400 primary | author | scale — the estimator's full sample |
| 150 of those 400 | independent labeller 2 | reliability (inter-rater agreement) |
| 75 of those 150 | independent labeller 3 | adjudication of systematic disagreement |

The 150- and 75-item overlap subsets are drawn from the 400 **now**, before any labelling begins,
using a fixed seed, so which items are double- and triple-labelled is itself part of the frozen
design rather than a choice made after seeing who is available or what early labels look like.
They are stratified proportionally across the three top strata (credited/wrong_parsed/unparsed) so
the overlap set is not accidentally concentrated in one.

**Labeller availability is not yet confirmed.** The previous round used three labellers (l1/l2/l3)
so the people plausibly exist, but this must be confirmed by the author before labelling starts. If
no second/third human is available, the design **degrades to author-plus-LLM-readers**, and that
degradation must be stated as an explicit limitation in the manuscript — not silently absorbed as
if it were the originally intended design. This is a decision only the author can make; it is not
something an AI assistant can substitute for without corrupting the exact thing being measured (an
independent human judgment of what a response commits to, from someone who did not build the
extractor being audited).

## Estimator and reporting plan

- **Primary estimate**: design-weighted φ̄ and ψ (Horvitz–Thompson-style: weighted mean of the
  per-item false-credit / wrong-span indicator, weights as defined above), with conservative
  one-sided 95% bounds. Exact method (Clopper–Pearson within stratum combined via the known
  weights, vs. a weighted bootstrap) to be finalized in `scripts/64_score_mathhard_human.py` and
  reported alongside the point estimate — not decided here in a way that could be tuned after
  seeing the labels, but the two candidate methods are both conservative one-sided procedures
  consistent with every other bound in this paper, and the choice between them will be justified by
  which one's coverage assumptions the finite stratum sizes actually satisfy.
- **Inter-rater agreement**: raw agreement and prevalence-adjusted statistics on the 150/75 overlap
  sets, not Cohen's/Fleiss' κ alone — this repository's own `interrater_agreement.json` and
  supporting docs already establish why κ collapses under this task's prevalence.
- **Exclusion rules, decided now**: an indeterminate response (the labeller cannot tell what it
  commits to even blind) is recorded as its own category, not silently merged into either credited
  or wrong_parsed. Truncated responses are labelled as whatever they commit to up to the truncation
  point, consistent with how `commit(r)` is defined in the paper's theory section. A response that
  contradicts itself (states two different final answers) is recorded as indeterminate.
- **Sensitivity reporting, decided now**: alongside the primary weighted estimate, report
  conservative alternatives treating indeterminate responses as (a) wrong, (b) missing/excluded,
  (c) their own category — so a reader who distrusts the primary treatment of indeterminate cases
  can read the bound that matches their own judgment, the same pattern §identification already uses
  for the zero-margin robustness row.
- **Downstream propagation, decided now**: once labels land, re-run `scripts/29` (pair
  identification), `scripts/47` (joint identification), `scripts/48` (ranking geometry), and
  `scripts/51` (margin contour) with the human-anchored margins substituted for the LLM-adjudicated
  ones, and report both versions side by side in the manuscript rather than silently replacing one
  with the other.

## What would make this null, and that this is still a finding

If human margins come out **larger** than the LLM adjudicators' (5.5% MMLU-transferred, 1.1%/0.9%
MATH-Hard-native), that widens the identified-set gap and makes *more* pairs unidentified —
strengthens the paper. If they come out **smaller**, more pairs become identified — a real finding,
reported honestly, not something to explain away. The zero-margin robustness row (87.8% not
identified, §identification) is the reported floor either way and does not depend on this audit at
all.

---

*Committed 2026-09-06, before `scripts/63_make_mathhard_sheets.py` exists and before any item is
drawn. The frame computation (`scripts/62_mathhard_frame.py`, committed alongside this document)
counted population sizes from data already on disk; it drew no sample and showed no item to anyone.*
