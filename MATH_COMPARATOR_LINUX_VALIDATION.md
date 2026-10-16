# is_equiv Linux validation (item 3) — RESOLVED, not merely "WSL unavailable"

WSL and Docker both confirmed unavailable on the dev machine (checked directly: `docker --version` → not found; `wsl --list` → not installed). Per instruction, tried the next option instead of stopping: **GitHub Actions on `ubuntu-latest`**. Workflow: `.github/workflows/is_equiv_linux_validation.yml`. Run: [34764225948](https://github.com/Rajveer-code/flipbudget/actions/runs/34764225948) — **success**.

## What it found

Real Ubuntu runner (`Linux-6.17.0-1022-azure-x86_64`, Python 3.11.16, `signal.SIGALRM` confirmed present) ran the exact same `scripts/fb_verify_is_equiv_bug.py` used to diagnose the Windows bug:

| Check | Result |
|---|---|
| 3 constructed known-equivalent cases (unsimplified fraction, commutative reorder, sign placement) | **Original, unmodified vendor `is_equiv` gets all 3 right on Linux** ("bug confirmed 0/3" — meaning broken==fixed==expected for all 3) |
| All 69 real T-C rows, original vs. fixed comparator | **0 changed** — identical to the Windows-computed result |
| Scorer-wrong events under the fixed comparator | **0/69**, matching the Windows figure exactly |

## What this confirms

1. **The bug is genuinely Windows-specific**, not a defect in the vendor `is_equiv` logic itself — on a platform where `SIGALRM` actually exists, the original, unpatched code already works correctly. This directly validates the root-cause diagnosis in `MATH_COMPARATOR_BUG.md`.
2. **The fix (`fb_math_comparator_fixed.py`) changes nothing on a platform where the original already worked** — it is a compatibility fix, not a behavior change, confirmed rather than assumed.
3. **A second, independent bug was found and fixed in the process of building this validation**: `fb_tc_real_analysis.py`'s `load_l1/l2/l3` hardcoded absolute `<home>\Downloads\...` paths — the CI run failed on this before the path fix, which is direct proof the project was not actually reproducible outside the author's own machine until now. Fixed (see `fix: T-C loader hardcoded absolute Downloads paths` commit); re-verified locally that the fix changes only the file path, not the data (identical 69/0/51/100% numbers before and after).

## Provenance

`results/flipbudget/is_equiv_bug_verification.json` (Windows) and `results/flipbudget/is_equiv_bug_verification_ubuntu_ci.json` (Linux CI, downloaded from the workflow artifact) are both preserved, tagged by platform, neither overwriting the other.
