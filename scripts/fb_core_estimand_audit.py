"""Core-estimand audit: what the scorer-stratified audit actually identifies.

Notation (kept distinct on purpose):
  Y*    latent (human-reference) correctness, Yh  scorer output.
  A     = P(Y*=1)                       true accuracy (target)
  a     = P(Yh=1)                       scorer accuracy (census over the benchmark)
  alpha = P(Yh=1 | Y*=0), beta = P(Yh=0 | Y*=1)   error rates (condition on truth)
  q     = P(Y*=0 | Yh=1), r    = P(Y*=1 | Yh=0)   predictive-value complements
                                                  (condition on scorer output)
An audit that samples within Yh strata estimates q and r directly, never
alpha/beta. Old pipeline plugged q-hat, r-hat into g(a, alpha, beta).

Sections: (1) algebra, (2) simulation, (3) real-data re-analysis,
(4) 458-row design. Output: results/flipbudget/core_estimand_audit.json.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import sympy as sp

sys.path.insert(0, "scripts")
from fb_e4_mathhard import load_data, matches_gold, is_indeterminate  # noqa: E402

Z = 1.96
SEED = 42
OUT = "results/flipbudget/core_estimand_audit.json"


# ---------------------------------------------------------------- maps
def g_old(a, x, y):
    """Misclassification inversion (a - alpha)/(1 - alpha - beta)."""
    return (a - x) / (1 - x - y)


def pv_map(a, q, r):
    """Law of total probability conditioned on scorer output."""
    return a * (1 - q) + (1 - a) * r


def wilson(x, n, z=Z):
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pv_interval(a, box):
    qlo, qhi, rlo, rhi = box
    return pv_map(a, qhi, rlo), pv_map(a, qlo, rhi)


def old_interval(a, box):
    """Exactly the old pipeline's corner envelope; None = empty/degenerate."""
    vals = [g_old(a, x, y) for x in box[:2] for y in box[2:] if x + y < 0.999]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if hi < 0 or lo > 1:
        return None
    return max(lo, 0.0), min(hi, 1.0)


# ---------------------------------------------------------------- 1 algebra
def algebra():
    A, al, be = sp.symbols("A alpha beta", positive=True)
    a = A * (1 - be) + (1 - A) * al
    q = (1 - A) * al / a                   # Bayes
    r = A * be / (1 - a)
    ident = sp.simplify(a * (1 - q) + (1 - a) * r - A)
    back_alpha = sp.simplify(a * q / (1 - A) - al)
    back_beta = sp.simplify((1 - a) * r / A - be)
    # what the old pipeline computes when fed the true q, r:
    old_given_pv = sp.simplify(g_old(a, q, r) - A)
    return {
        "pv_identity_residual": str(ident),
        "alpha_from_q_residual": str(back_alpha),
        "beta_from_r_residual": str(back_beta),
        "old_map_fed_q_r_minus_A": str(sp.factor(old_given_pv)),
    }


# ---------------------------------------------------------------- 2 simulation
def simulate(n_rep=4000, seed=SEED):
    """Finite benchmark, differential per-model error, audit SRS within Yh
    strata. Compares old map vs PV map (bias, coverage of a 95% Sidak
    projection set for A)."""
    rng = np.random.default_rng(seed)
    scenarios = {
        "typical":        dict(N=1324, A=0.20, alpha=0.01, beta=0.02, n1=15, n0=15),
        "low_acc":        dict(N=1324, A=0.02, alpha=0.01, beta=0.02, n1=6, n0=15),
        "zero_acc_model": dict(N=1324, A=0.00, alpha=0.01, beta=0.00, n1=6, n0=15),
        "high_error":     dict(N=1324, A=0.40, alpha=0.10, beta=0.15, n1=40, n0=40),
        "perfect_scorer": dict(N=1324, A=0.20, alpha=0.00, beta=0.00, n1=11, n0=11),
    }
    lvl = 0.95 ** 0.5                      # two independent params -> Sidak
    z_s = float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + lvl / 2))
    out = {}
    for name, s in scenarios.items():
        ys = (rng.random(s["N"]) < s["A"]).astype(int)
        res = {"old_pt": [], "pv_pt": [], "pv_cover": 0, "old_cover": 0,
               "old_empty": 0, "pv_width": [], "old_width": []}
        for _ in range(n_rep):
            flip = np.where(ys == 1, rng.random(s["N"]) < s["beta"],
                            rng.random(s["N"]) < s["alpha"])
            yh = np.where(flip, 1 - ys, ys)
            A_true, a = ys.mean(), yh.mean()
            i1, i0 = np.flatnonzero(yh == 1), np.flatnonzero(yh == 0)
            s1 = rng.choice(i1, min(s["n1"], len(i1)), replace=False) if len(i1) else i1
            s0 = rng.choice(i0, min(s["n0"], len(i0)), replace=False)
            x_q = int((ys[s1] == 0).sum()); x_r = int((ys[s0] == 1).sum())
            q_hat = x_q / len(s1) if len(s1) else 0.0
            r_hat = x_r / len(s0)
            res["pv_pt"].append(pv_map(a, q_hat, r_hat) - A_true)
            if q_hat + r_hat < 0.999:
                res["old_pt"].append(g_old(a, q_hat, r_hat) - A_true)
            box = wilson(x_q, len(s1), z_s) + wilson(x_r, len(s0), z_s)
            lo, hi = pv_interval(a, box)
            res["pv_cover"] += lo - 1e-12 <= A_true <= hi + 1e-12
            res["pv_width"].append(hi - lo)
            oi = old_interval(a, box)
            if oi is None:
                res["old_empty"] += 1
            else:
                res["old_cover"] += oi[0] - 1e-12 <= A_true <= oi[1] + 1e-12
                res["old_width"].append(oi[1] - oi[0])
        out[name] = {
            **s,
            "pv_bias": float(np.mean(res["pv_pt"])),
            "old_bias": float(np.mean(res["old_pt"])) if res["old_pt"] else None,
            "pv_coverage": res["pv_cover"] / n_rep,
            "old_coverage_among_nonempty": (res["old_cover"] / (n_rep - res["old_empty"])
                                            if n_rep > res["old_empty"] else None),
            "old_empty_rate": res["old_empty"] / n_rep,
            "pv_median_width": float(np.median(res["pv_width"])),
            "old_median_width": float(np.median(res["old_width"])) if res["old_width"] else None,
        }
    return out


# ---------------------------------------------------------------- 3 real data
def load_real():
    key, answers = load_data()
    frame = json.load(open("results/analysis/mathhard_frame.json"))
    pop = json.load(open("results/analysis/tc_expansion_population.json"))["population"]

    def w(item):
        t = frame["targets_by_stratum"][item["stratum"]]
        cell = f'{item["subject"]}|q{item["quartile"]}'
        n = t["sample_target_by_cell"].get(cell)
        return t["population_by_cell"][cell] / n if n else None

    audit = defaultdict(lambda: {"cred": [], "wrong": [], "indet_cred": 0, "indet_wrong": 0})
    for it in key:
        ans = answers.get(it["uid"])
        side = "cred" if it["stratum"] == "credited" else "wrong"
        if is_indeterminate(ans):
            audit[it["model"]]["indet_" + side] += 1
            continue
        human_ok = matches_gold(ans, it["gold"])
        err = (not human_ok) if side == "cred" else human_ok
        audit[it["model"]][side].append((w(it), int(err)))

    a_sb, a_em, n_bench = defaultdict(list), defaultdict(list), {}
    strat_check = {}
    pop_idx = {}
    for row in pop:
        a_sb[row["model"]].append(row["score_boxed_stratum"] == "credited")
        a_em[row["model"]].append(row["exact_match"] == 1)
        pop_idx[(row["model"], row["item_id"])] = row["score_boxed_stratum"]
    agree = [(pop_idx[(it["model"], it["item_id"])] == "credited") == (it["stratum"] == "credited")
             for it in key if (it["model"], it["item_id"]) in pop_idx]
    strat_check = {"audit_items_found_in_population": len(agree),
                   "stratum_consistent": int(sum(agree))}
    models = {}
    for m in a_sb:
        d = audit.get(m)
        if d is None:
            continue
        models[m] = {"a_sb": float(np.mean(a_sb[m])), "a_em": float(np.mean(a_em[m])),
                     "N": len(a_sb[m]),
                     "x_q": sum(e for _, e in d["cred"]), "n_q": len(d["cred"]),
                     "x_r": sum(e for _, e in d["wrong"]), "n_r": len(d["wrong"]),
                     "indet_cred": d["indet_cred"], "indet_wrong": d["indet_wrong"],
                     "q_hajek": (sum(wt * e for wt, e in d["cred"]) / sum(wt for wt, _ in d["cred"])
                                 if d["cred"] else None)}
    return models, strat_check


def pair_table(models, acc_key, mapper, level=None):
    """level=None: per-parameter 95% Wilson (old convention).
    level=0.95: Sidak projection -- each of the 4 independent parameters at
    0.95**(1/4), so the projected set covers Delta with >=95% (given a)."""
    from scipy.stats import norm
    z = Z if level is None else float(norm.ppf(0.5 + level ** 0.25 / 2))
    qual = sorted(m for m, v in models.items() if v["n_q"] > 0 and v["n_r"] > 0)
    rows, empty = [], 0
    for m1, m2 in combinations(qual, 2):
        v1, v2 = models[m1], models[m2]
        a1, a2 = v1[acc_key], v2[acc_key]
        ws = 2 * Z * np.sqrt(a1 * (1 - a1) / v1["N"] + a2 * (1 - a2) / v2["N"])
        b1 = wilson(v1["x_q"], v1["n_q"], z) + wilson(v1["x_r"], v1["n_r"], z)
        b2 = wilson(v2["x_q"], v2["n_q"], z) + wilson(v2["x_r"], v2["n_r"], z)
        i1, i2 = mapper(a1, b1), mapper(a2, b2)
        if i1 is None or i2 is None:
            empty += 1
            continue
        lo, hi = i1[0] - i2[1], i1[1] - i2[0]
        rows.append({"pair": [m1, m2], "w_sampling": ws, "w_audit": hi - lo,
                     "ratio": (hi - lo) / ws, "resolved": bool(lo > 0 or hi < 0)})
    r = np.array([x["ratio"] for x in rows])
    return {"n_qualifying_models": len(qual), "n_pairs": len(qual) * (len(qual) - 1) // 2,
            "n_valid": len(rows), "n_empty": empty,
            "median_ratio": float(np.median(r)), "min_ratio": float(r.min()),
            "max_ratio": float(r.max()), "pct_dominant": float(100 * np.mean(r > 1)),
            "median_w_audit": float(np.median([x["w_audit"] for x in rows])),
            "median_w_sampling": float(np.median([x["w_sampling"] for x in rows])),
            "n_resolved": int(sum(x["resolved"] for x in rows))}


def heterogeneity(models, n_perm=20000, seed=SEED):
    """Permutation test (fixed margins) that q is equal across models."""
    rng = np.random.default_rng(seed)
    qual = [m for m, v in models.items() if v["n_q"] > 0]
    n = np.array([models[m]["n_q"] for m in qual]); x = np.array([models[m]["x_q"] for m in qual])
    lab = np.repeat(np.arange(len(qual)), n)
    ev = np.zeros(n.sum(), int); ev[:x.sum()] = 1

    def stat(xs):
        p = xs.sum() / n.sum()
        e = n * p
        return float(((xs - e) ** 2 / np.where(e > 0, e, 1)).sum())
    obs = stat(x)
    ge = 0
    for _ in range(n_perm):
        xs = np.bincount(lab, weights=rng.permutation(ev), minlength=len(qual))
        ge += stat(xs) >= obs - 1e-12
    return {"models": len(qual), "false_credits": int(x.sum()), "n_credited_audited": int(n.sum()),
            "false_miss_total": int(sum(v["x_r"] for v in models.values())),
            "n_wrong_audited": int(sum(v["n_r"] for v in models.values())),
            "chi2_obs": obs, "perm_p": (ge + 1) / (n_perm + 1)}


def crossover(a=0.2, N=1324):
    """Smallest per-stratum audit n (zero observed events, both strata) at
    which the PV audit width for one model falls below its sampling width."""
    ws = 2 * Z * np.sqrt(a * (1 - a) / N)
    for n in range(1, 20000):
        lo, hi = pv_interval(a, wilson(0, n) + wilson(0, n))
        if hi - lo < ws:
            return {"a": a, "N": N, "w_sampling": float(ws), "n_per_stratum": n}
    return None


def mechanism_model(models, n_boot=2000, seed=SEED):
    """H0: one (alpha, beta) for every model (scorer error depends only on
    truth, not on which model wrote the response). Then per model
    A_m = g(a_m, alpha, beta), q_m = alpha(1-A_m)/a_m, r_m = beta A_m/(1-a_m).
    ML fit on a grid, LR test vs saturated per-model (q_m, r_m), parametric
    bootstrap p-value. Also the 95% profile-likelihood set for alpha."""
    qual = [m for m, v in models.items() if v["n_q"] > 0 and v["n_r"] > 0]
    a = np.array([models[m]["a_sb"] for m in qual])
    nq = np.array([models[m]["n_q"] for m in qual]); nr = np.array([models[m]["n_r"] for m in qual])
    al_grid = np.linspace(0, 0.05, 501); be_grid = np.linspace(0, 0.3, 151)

    def binll(x, n, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return x * np.log(p) + (n - x) * np.log(1 - p)

    def fit(xq, xr):
        best = (-np.inf, None, None); prof = np.full(len(al_grid), -np.inf)
        for i, al in enumerate(al_grid):
            for be in be_grid:
                A = (a - al) / (1 - al - be)
                if np.any(A < 0) or np.any(A > 1):
                    continue
                q = np.where(a > 0, al * (1 - A) / np.where(a > 0, a, 1), 0.0)
                r = be * A / (1 - a)
                ll = binll(xq, nq, q).sum() + binll(xr, nr, r).sum()
                prof[i] = max(prof[i], ll)
                if ll > best[0]:
                    best = (ll, al, be)
        return best, prof

    def sat(xq, xr):
        return binll(xq, nq, xq / nq).sum() + binll(xr, nr, xr / nr).sum()

    xq = np.array([models[m]["x_q"] for m in qual]); xr = np.array([models[m]["x_r"] for m in qual])
    (ll0, al_hat, be_hat), prof = fit(xq, xr)
    lr = 2 * (sat(xq, xr) - ll0)
    ok = prof >= prof.max() - 3.84 / 2
    feas_max_alpha = float(a.min())  # alpha > min a_m makes that model's A negative
    rng = np.random.default_rng(seed)
    A_hat = (a - al_hat) / (1 - al_hat - be_hat)
    q0 = np.where(a > 0, al_hat * (1 - A_hat) / np.where(a > 0, a, 1), 0.0); r0 = be_hat * A_hat / (1 - a)
    ge = 0
    nb = n_boot
    for _ in range(nb):
        bq, br = rng.binomial(nq, np.clip(q0, 0, 1)), rng.binomial(nr, np.clip(r0, 0, 1))
        (llb, _, _), _ = fit(bq, br)
        ge += 2 * (sat(bq, br) - llb) >= lr - 1e-9
    return {"n_models": len(qual), "alpha_hat": float(al_hat), "beta_hat": float(be_hat),
            "alpha_95_profile": [float(al_grid[ok].min()), float(al_grid[ok].max())],
            "alpha_feasible_max": feas_max_alpha,
            "LR_vs_saturated": float(lr), "bootstrap_p": (ge + 1) / (nb + 1),
            "per_model_a_sb": dict(zip(qual, a.tolist()))}


# ---------------------------------------------------------------- 4 design
def design_458():
    pop = json.load(open("results/analysis/tc_expansion_population.json"))["population"]
    rows = [r for r in pop if not r["already_labeled"]]
    def cell(r):
        sb = r["score_boxed_stratum"] == "credited"; em = r["exact_match"] == 1
        return {(1, 1): "both_credited", (0, 0): "both_not_credited",
                (1, 0): "disagree_sb_credited_em_not", (0, 1): "disagree_em_credited_sb_not"}[(int(sb), int(em))]
    Nc = defaultdict(int)
    for r in rows:
        Nc[cell(r)] += 1
    N = sum(Nc.values())
    W = {c: Nc[c] / N for c in Nc}
    cur = {"both_credited": 200, "both_not_credited": 198,
           "disagree_sb_credited_em_not": 30, "disagree_em_credited_sb_not": 30}
    total = sum(cur.values())
    # planning priors for P(Y*=1 | cell), from the existing 350-item audit
    # (q_sb ~ 0.031, r_sb ~ 0 -> floor 0.01) and an agnostic 0.5 for disagreements
    priors = {"both_credited": 0.97, "both_not_credited": 0.01,
              "disagree_sb_credited_em_not": 0.5, "disagree_em_credited_sb_not": 0.5}
    # target coefficient vectors: estimand = sum_c k_c * W_c * p_c  (+ const)
    targets = {
        "A_true (pooled)": {c: 1.0 for c in W},
        "q_em = P(Y*=0|em=1)": "q_em", "q_sb = P(Y*=0|sb=1)": "q_sb",
        "r_em = P(Y*=1|em=0)": "r_em", "r_sb = P(Y*=1|sb=0)": "r_sb",
        "a_em - a_sb bias gap": {"disagree_em_credited_sb_not": 1.0, "disagree_sb_credited_em_not": -1.0},
    }
    em1 = ["both_credited", "disagree_em_credited_sb_not"]; sb1 = ["both_credited", "disagree_sb_credited_em_not"]
    em0 = ["both_not_credited", "disagree_sb_credited_em_not"]; sb0 = ["both_not_credited", "disagree_em_credited_sb_not"]

    def var(alloc, coef):
        return sum((coef.get(c, 0) * W[c]) ** 2 * priors[c] * (1 - priors[c]) / alloc[c]
                   * (1 - alloc[c] / Nc[c]) for c in W if coef.get(c, 0))

    def coef_for(t):
        if isinstance(t, dict):
            return t
        cells = {"q_em": em1, "q_sb": sb1, "r_em": em0, "r_sb": sb0}[t]
        tot = sum(W[c] for c in cells)
        return {c: W[c] / tot / W[c] for c in cells}  # ratio of domain -> weight W_c/W_dom

    def neyman(coef):
        s = {c: abs(coef.get(c, 0)) * W[c] * np.sqrt(priors[c] * (1 - priors[c])) for c in W}
        tot = sum(s.values())
        return {c: max(2, round(total * s[c] / tot)) if s[c] > 0 else 2 for c in W}

    prop = {c: max(2, round(total * W[c])) for c in W}
    out = {"population_cells_N": dict(Nc), "W": W, "planning_priors": priors, "current": cur,
           "proportional": prop, "targets": {}}
    for name, t in targets.items():
        coef = coef_for(t)
        ny = neyman(coef)
        out["targets"][name] = {
            "se_current": float(np.sqrt(var(cur, coef))),
            "se_proportional": float(np.sqrt(var(prop, coef))),
            "neyman_alloc": ny, "se_neyman": float(np.sqrt(var(ny, coef))),
        }
    return out


def main():
    res = {"algebra": algebra()}
    print("ALGEBRA", json.dumps(res["algebra"], indent=1))
    res["simulation"] = simulate()
    for k, v in res["simulation"].items():
        print(f"SIM {k:15s} pv_bias={v['pv_bias']:+.4f} old_bias={v['old_bias']} "
              f"pv_cov={v['pv_coverage']:.3f} old_cov={v['old_coverage_among_nonempty']} "
              f"old_empty={v['old_empty_rate']:.3f} w_pv={v['pv_median_width']:.3f} w_old={v['old_median_width']}")

    models, strat_check = load_real()
    res["real_stratum_check"] = strat_check
    res["real_models"] = models
    print("stratum check", strat_check)
    res["heterogeneity_q"] = heterogeneity(models)
    print("HETEROGENEITY", res["heterogeneity_q"])
    res["mechanism_model"] = mechanism_model(models, n_boot=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
    print("MECHANISM", {k: v for k, v in res["mechanism_model"].items() if k != "per_model_a_sb"})
    for m, v in sorted(models.items(), key=lambda kv: kv[1]["a_sb"]):
        if v["n_q"] and v["n_r"]:
            print(f"  {m[:36]:36s} a_sb={v['a_sb']:.3f} a_em={v['a_em']:.3f} q {v['x_q']}/{v['n_q']} "
                  f"(hajek {v['q_hajek']:.3f}) r {v['x_r']}/{v['n_r']} indet c/w {v['indet_cred']}/{v['indet_wrong']}")
    tables = {
        "A_old  (a=exact_match, box=score_boxed q/r, old map)": ("a_em", old_interval, None),
        "B_old  (a=score_boxed, old map)":                      ("a_sb", old_interval, None),
        "A_pv   (a=exact_match + score_boxed q/r: INCOHERENT)": ("a_em", pv_interval, None),
        "B_pv   (a=score_boxed, PV map, per-param 95%)":        ("a_sb", pv_interval, None),
        "B_pv_sidak (PV map, joint >=95% projection)":          ("a_sb", pv_interval, 0.95),
    }
    res["pairwise"] = {}
    for name, (ak, mp, lvl) in tables.items():
        t = pair_table(models, ak, mp, lvl)
        res["pairwise"][name] = t
        print(f"{name:55s} valid={t['n_valid']}/{t['n_pairs']} empty={t['n_empty']} "
              f"median={t['median_ratio']:.3f} min={t['min_ratio']:.3f} dom={t['pct_dominant']:.1f}% "
              f"w_audit={t['median_w_audit']:.3f} w_samp={t['median_w_sampling']:.4f} resolved={t['n_resolved']}")
    res["crossover"] = [crossover(a) for a in (0.05, 0.2, 0.35)]
    print("CROSSOVER", res["crossover"])
    res["design_458"] = design_458()
    d = res["design_458"]
    print("458 cells N", d["population_cells_N"])
    for k, v in d["targets"].items():
        print(f"  {k:28s} se cur={v['se_current']:.4f} prop={v['se_proportional']:.4f} "
              f"neyman={v['se_neyman']:.4f} alloc={v['neyman_alloc']}")
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
