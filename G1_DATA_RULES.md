# G1 data rules (prespecified)

Fixed 2026-09-19, before the K=3 census or any certification count on the 27-model
roster was computed. Changes after this commit must be logged as amendments below,
with date and reason.

## Missingness rule (G1.2)

1. Item universe: the 1,324 MATH-Hard (Level 5) items, identified by `doc_hash`.
   Model set: the 27-model roster of `results/flipbudget/e4_mathhard_per_model.json`
   (primary, 351 pairs). The 17-model set is used only for the regression anchor.
2. Source: for each (model, subject), the latest-timestamp `samples_leaderboard_math_*`
   file (the scripts/19 rule). A file absent from `knowledgeshift/results/cache_l1` is
   downloaded from the same Hugging Face details repo with its revision recorded. It is
   not missing if it exists on the Hub.
3. A (model, item) row is **scored** iff the record exists, the response is a string,
   gold exists, and the published `exact_match` is 0 or 1. A verifier that fails to
   parse or times out gives a **verdict of 0**, not missingness. These events are counted
   per verifier and reported.
4. `missing_rate = 1 - n_scored / 1324`. A model with `missing_rate > 5%` is dropped from
   all analyses and the drop is reported. No other reason drops a model.
5. Primary analysis: the **common-item subset** (items scored for every included model).
   Sensitivities: (a) all scored items per model; (b) missing rows counted as incorrect
   under all three verifiers.

## Amendments

None.
