"""Minimax-efficiency allocation of the 458-row audit across the 4 joint-scorer
cells, for the corrected (predictive-value) targets. Efficiency of a target =
Var(Neyman-optimal for that target) / Var(candidate). Maximise the minimum."""
import json, itertools, numpy as np
d = json.load(open("results/flipbudget/core_estimand_audit.json"))["design_458"]
N, W, P = d["population_cells_N"], d["W"], d["planning_priors"]
C = ["both_credited", "both_not_credited", "disagree_sb_credited_em_not", "disagree_em_credited_sb_not"]
em1, sb1 = [C[0], C[3]], [C[0], C[2]]; em0, sb0 = [C[1], C[2]], [C[1], C[3]]
def dom(cells): t = sum(W[c] for c in cells); return {c: 1 / t for c in cells}
T = {"A_true": {c: 1.0 for c in C}, "q_em": dom(em1), "q_sb": dom(sb1), "r_em": dom(em0), "r_sb": dom(sb0),
     "disagreement_gap": {C[3]: 1.0, C[2]: -1.0}}
def var(n, k): return sum((k.get(c, 0) * W[c]) ** 2 * P[c] * (1 - P[c]) / n[c] * (1 - n[c] / N[c]) for c in C if k.get(c))
best_v = {t: d["targets"][{"A_true": "A_true (pooled)", "q_em": "q_em = P(Y*=0|em=1)", "q_sb": "q_sb = P(Y*=0|sb=1)",
          "r_em": "r_em = P(Y*=1|em=0)", "r_sb": "r_sb = P(Y*=1|sb=0)", "disagreement_gap": "a_em - a_sb bias gap"}[t]]["se_neyman"] ** 2 for t in T}
def eff(n): return {t: best_v[t] / var(n, T[t]) for t in T}
cur = dict(zip(C, [200, 198, 30, 30]))
best = None
for bc in range(20, 400, 4):
    for ds in range(10, min(291, 458 - bc), 4):
        for de in range(10, min(493, 458 - bc - ds), 4):
            bn = 458 - bc - ds - de
            if bn < 10: continue
            n = dict(zip(C, [bc, bn, ds, de])); e = min(eff(n).values())
            if best is None or e > best[0]: best = (e, n)
out = {"current": cur, "current_eff": eff(cur), "minimax": best[1], "minimax_eff": eff(best[1])}
json.dump(out, open("results/flipbudget/redesign_458.json", "w"), indent=1)
for k in ("current", "minimax"):
    print(k, out[k], {t: round(v, 3) for t, v in out[k + "_eff"].items()})
