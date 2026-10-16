# A real, platform-specific bug in the MATH-Hard comparator, found via the adversarial suite

Found while running `scripts/fb_adversarial_suite.py` (items 13-14):
6 of 8 initial "mismatches" against known-ground-truth constructed cases
turned out to share one root cause, not be independent test-writing errors.
Verified with dedicated scripts before trusting or reporting anything:
`scripts/fb_verify_is_equiv_bug.py`, `scripts/fb_verify_original_audit_bug.py`.
Results: `results/flipbudget/is_equiv_bug_verification.json`,
`results/flipbudget/original_audit_bug_verification.json`.

## The bug

The vendored comparator's `is_equiv` (`vendor/leaderboard_math/utils.py`,
copied character-for-character from Lewkowycz et al. 2022 appendix D, the
same code used across this entire project) wraps its symbolic-equivalence
check in a `timeout()` context manager that calls
`signal.signal(signal.SIGALRM, ...)`. **`SIGALRM` does not exist on
Windows** — confirmed directly: `hasattr(signal, 'SIGALRM')` is `False` on
this machine. `is_equiv`'s own broad `except Exception: return False`
silently swallows the resulting `AttributeError`, so **`is_equiv` returns
`False` for every call, unconditionally, including for genuinely equivalent
expressions.**

**The existing "fix" in three scripts
(`fb_tc_real_analysis.py`, `tc_correlated_error.py`,
`ee_format_association.py`) does not actually fix this.** All three detect
the missing `SIGALRM` and wrap `is_equiv` in a `ThreadPoolExecutor` with its
own timeout — but the wrapped call is still the *original* `is_equiv`, which
still tries `signal.signal(signal.SIGALRM, ...)` internally and still hits
its own exception handler before the wrapper's timeout logic is ever
reached. The wrapper is real insurance against a *different* problem
(`signal.alarm` also fails if called from a non-main thread, which the
wrapper itself does) — it just doesn't fix the one that actually fires here.

**Confirmed directly, not inferred**: `matches_gold_broken(r"\frac{2}{4}",
r"\frac{1}{2}")` → `False` (wrong; these are equal); the same call through a
comparator with the `signal.alarm` dependency removed entirely
(`scripts/fb_math_comparator_fixed.py`) → `True` (correct).

## Impact — checked directly against every real dataset in this project, not assumed

**T-C's own 69-row real audit (this session's headline human-audit result):
UNCHANGED.** Re-ran all 69 usable rows through the fixed comparator: **0
rows change.** Every real transcription that was correct also happened to be
an exact string match to gold (post-normalization); none of the 69 relied
on the broken symbolic-equivalence fallback to reach its verdict. **The
0/69 scorer-wrong result, and everything built on it
(`TC_REAL_ANALYSIS.md`, `TC_ALPHA_BETA_UPDATE.md`, the consolidated verdict)
stands exactly as reported.**

**The original 400-item companion-project audit (`TIER1_VERDICT.md`'s
basis): one real, small correction found.** `uid=efcbea8134ac`,
`google/gemma-7b`, answer `\frac{411}{333}` vs. gold `\frac{137}{111}` —
these ARE equal (`411/3=137`, `333/3=111`) and the broken comparator missed
it. **Total real scorer-wrong events on the 400-item audit: 6, not 5.**

**This is not retroactively applied to `EE_FEATURE_STABILITY_CHECK.md`'s
AUC=0.830 or `TIER1_VERDICT.md`'s "5 positive events" language — those
numbers are not re-derived here.** A single additional event moves 5→6 out
of 350 usable records; the "underpowered, cannot trust the point estimate"
characterization is extremely unlikely to change at that scale (the
original CI already crossed chance at n=5; n=6 will not resolve that), but
**re-running E-E's logistic regression and bootstrap with the corrected
6-event dataset is a real, small, not-yet-done follow-up**, flagged
explicitly rather than silently left stale or silently "corrected" without
re-running the actual fit.

## Genuine remaining comparator limitations, found by the same suite (unaffected by this fix)

Two of the original eight adversarial mismatches persist even with the
`SIGALRM` bug fixed — real, separate findings about the comparator itself,
not artifacts of this bug:

- **Reordered interval/set-union notation is not recognized as equivalent**
  (`(-\infty,-7)\cup(-7,3)\cup(3,\infty)` vs. the same three intervals in a
  different order) — `sympy`'s `parse_latex`+subtract+`simplify` pipeline is
  built for scalar algebraic equivalence, not set-valued comparison. A real,
  structural limitation for any MATH-Hard item whose gold answer is a union
  of intervals, worth naming as a specific mechanism (item 10) rather than
  folding into a generic "formatting" bucket.
- **A bare numeral is treated as equivalent to the same numeral with a unit
  word appended** (`"9"` vs. `"9 meters"`) — traced to `REMOVED_EXPRESSIONS`
  in the vendor file, which deliberately strips common unit words
  (`feet`, `inches`, `km`, ... — confirmed present in the source). This
  looks like **intentional behavior** matching the MATH dataset's own
  unit-free-gold convention, not a bug — the adversarial suite's own
  expectation for this case was likely the wrong ground truth, not the
  comparator.

## What was fixed and what was not

Fixed: a reusable, correct comparator
(`scripts/fb_math_comparator_fixed.py`, `is_equiv_fixed`/
`matches_gold_fixed`) with no `signal.alarm` dependency anywhere in its call
path, self-checked against 4 known cases. **Any new analysis on this
machine should import this, not the vendor's `is_equiv` directly or through
the existing three scripts' non-working patch.**

Not fixed (by design, not oversight): the three existing scripts
(`fb_tc_real_analysis.py`, `tc_correlated_error.py`,
`ee_format_association.py`) are left exactly as they are — they already
produced results that have now been individually re-verified against this
bug (T-C: unchanged; original audit: one event found). Patching them
in place would risk quietly changing already-reported, already-verified
numbers without the explicit before/after record this document provides.
