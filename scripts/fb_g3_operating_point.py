"""G3.1 -- apply the plan's default operating-point rule, deterministically, before any draw.

Rule (PLAN_AISTATS2027.md G3.1 default): reserve the concordant budget for the models that appear
most in fragile orderings, each concordant cell sized for 80% power at rho = 0.20 under the valid
falsification rule (fb_g3_frontier.py); put the remainder on discordant strata in proportion to
stratum size. F = the largest number of fragile models for which every per-cell n0 fits both the
budget and the unlabeled capacity of every chosen model.
Run: python scripts/fb_g3_operating_point.py   Out: results/flipbudget/g3_operating_point.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fb_g3_frontier as G  # noqa: E402

OUT = ROOT / "results" / "flipbudget" / "g3_operating_point.json"
RHO, POWER = 0.20, 0.80


def n0_for(inp, F, c):
    ub = G.pooled_error_ub(inp, F, c)
    return next((n for n in range(1, 1000) if G.power(n, F, ub, RHO) >= POWER), None), ub


def main():
    G._init()
    inp, cap, order = G.CTX["inp"], G.CTX["cap"], G.CTX["order"]
    best = None
    for F in range(1, len(order) + 1):
        per_cell = {c: n0_for(inp, F, c) for c in inp.conc_cells}
        need = {c: v[0] for c, v in per_cell.items()}
        fits = all(need[c] is not None and all(cap[(m, c)] >= need[c] for m in order[:F]) for c in need)
        if not fits or F * sum(need.values()) > G.BUDGET:
            break
        best = (F, need, {c: v[1] for c, v in per_cell.items()})
    F, need, ub = best
    op = {"rule": "PLAN G3.1 default (see docstring)", "rho": RHO, "power": POWER, "eps0": G.EPS0,
          "alpha_falsification": G.ALPHA_F, "budget": G.BUDGET, "F": F,
          "falsification_models": order[:F], "n0_by_cell": need,
          "pooled_error_ub_existing": ub,
          "reject_if_errors_at_least": {c: G.reject_threshold(need[c], F, ub[c]) for c in need},
          "labels_falsification": F * sum(need.values()),
          "labels_certification": G.BUDGET - F * sum(need.values()),
          "cannot_detect": f"per-model concordant error below {RHO} at {POWER:.0%} power in the {F} screened "
                           f"models; any departure in the other {len(order) - F} models (unscreened)"}
    json.dump(op, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in op.items() if k != "falsification_models"}, indent=1))
    print("screened:", op["falsification_models"])


if __name__ == "__main__":
    main()
