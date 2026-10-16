"""Two-verifier (joint-cell) identification of MATH-Hard orderings -- final version.

Cells (score_boxed, exact_match): 11 both credit, 10 sb only, 01 em only, 00 both reject.
  A*_m = sum_c s_mc p_mc,   s_mc census shares (known), p_mc = P(Y*=1 | cell c, model m)
Delta* is linear in the p's; its range over a box is attained at corners (sharp).

Coverage (per ordering, >= 95%):
  audit part at 0.975: Clopper-Pearson intervals, Sidak over the k independently
    estimated parameters the ordering depends on (disjoint samples -> independent);
  sampling part at 0.975: normal interval for the accuracy difference;
  union bound (Bonferroni) combines them -- no independence assumption.
  Parameters bounded logically ([0,1], unaudited) carry no coverage cost.
Simultaneous variant: sampling at 0.975/136 per pair (audit part is already
simultaneous when it depends only on pooled parameters).

Analyses:
  shared      concordant cells pooled across models (homogeneity), disagreement [0,1]
  hetero_L    each model's concordant rates vary independently within
              [1 - L(1 - p11_lo), 1] and [0, L p00_hi]  (bounded heterogeneity), L grid
  pooled_dis  disagreement cells also pooled (assumption contradicted by command-r)
  census      projection: disagreement truths known (set to pooled point estimate)
  design sims 458 added labels under six designs, same objective
Indeterminate human labels: Y*=0 (primary); dropped (sensitivity).
Output: results/flipbudget/joint_verifier_results.json
"""
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import beta as Beta, norm

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from fb_e4_mathhard import load_data, matches_gold, is_indeterminate  # noqa: E402

OUT = "results/flipbudget/joint_verifier_results.json"
CELLS = ["11", "10", "01", "00"]
DIS = ("10", "01")
L_GRID = [1, 1.5, 2, 3, 5, 10, 20, 50]
N_REP = 200
BUDGET = 458
SEED = 42
Z_SAMP = float(norm.ppf(1 - 0.025 / 2))            # 0.975 two-sided, per ordering
Z_SAMP_SIM = float(norm.ppf(1 - 0.025 / 2 / 136))  # simultaneous over 136 orderings


def cp(x, n, level):
    """Clopper-Pearson interval at `level`."""
    if n == 0:
        return 0.0, 1.0
    a = 1 - level
    lo = 0.0 if x == 0 else Beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else Beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def lvl(k):
    """Per-parameter level so that k independent intervals hold jointly at 0.975."""
    return 0.975 ** (1 / max(k, 1))


def census():
    pop = json.load(open("results/analysis/tc_expansion_population.json"))["population"]
    cnt = defaultdict(lambda: dict.fromkeys(CELLS, 0))
    unl = defaultdict(lambda: dict.fromkeys(CELLS, 0))
    cellmap = {}
    for r in pop:
        c = f"{int(r['score_boxed_stratum'] == 'credited')}{int(r['exact_match'] == 1)}"
        cnt[r["model"]][c] += 1
        if not r["already_labeled"]:
            unl[r["model"]][c] += 1
        cellmap[(r["model"], r["item_id"])] = c
    shares = {m: {c: v[c] / sum(v.values()) for c in CELLS} | {"N": sum(v.values())} for m, v in cnt.items()}
    return shares, cellmap, {m: dict(v) for m, v in unl.items()}


def audit_counts(cellmap, indet):
    key, ans = load_data()
    pooled = {c: [0, 0] for c in CELLS}
    per_model = defaultdict(lambda: {c: [0, 0] for c in CELLS})
    for it in key:
        k = (it["model"], it["item_id"])
        if k not in cellmap:
            continue
        c = cellmap[k]
        a = ans.get(it["uid"])
        if is_indeterminate(a):
            if indet == "drop":
                continue
            y = 0
        else:
            y = int(matches_gold(a, it["gold"]))
        pooled[c][0] += y
        pooled[c][1] += 1
        per_model[it["model"]][c][0] += y
        per_model[it["model"]][c][1] += 1
    return pooled, {m: v for m, v in per_model.items()}


def rng_range(terms):
    lo = sum(k * (l if k > 0 else h) for k, l, h in terms)
    hi = sum(k * (h if k > 0 else l) for k, l, h in terms)
    return lo, hi


def acc_se(s1, s2):
    a1, a2 = s1["11"] + s1["10"], s2["11"] + s2["10"]
    return np.sqrt(a1 * (1 - a1) / s1["N"] + a2 * (1 - a2) / s2["N"])


def evaluate(shares, models, conc, dis, z=Z_SAMP):
    """conc: ('shared', box11, box00) or ('hetero', box11, box00) with per-model independence.
    dis: function (model, cell) -> (lo, hi)."""
    rows = []
    for m1, m2 in combinations(models, 2):
        s1, s2 = shares[m1], shares[m2]
        terms = []
        mode, b11, b00 = conc
        for c, b in (("11", b11), ("00", b00)):
            if mode == "shared":
                terms.append((s1[c] - s2[c], *b))
            elif mode == "permodel":  # b11/b00 are dicts model -> box
                terms += [(s1[c], *b[m1]), (-s2[c], *b[m2])]
            else:
                terms += [(s1[c], *b), (-s2[c], *b)]
        for c in DIS:
            terms += [(s1[c], *dis(m1, c)), (-s2[c], *dis(m2, c))]
        lo, hi = rng_range(terms)
        se = acc_se(s1, s2)
        rows.append({"pair": [m1, m2], "lo": lo, "hi": hi, "se": se,
                     "audit": bool(lo > 0 or hi < 0), "combined": bool(lo - z * se > 0 or hi + z * se < 0)})
    return {"audit_resolved": int(sum(r["audit"] for r in rows)),
            "combined_resolved": int(sum(r["combined"] for r in rows)),
            "median_width": float(np.median([r["hi"] - r["lo"] for r in rows])), "pairs": rows}


def sampling_ceiling(shares, models, z=Z_SAMP):
    out = []
    for m1, m2 in combinations(models, 2):
        s1, s2 = shares[m1], shares[m2]
        gap = (s1["11"] + s1["10"]) - (s2["11"] + s2["10"])
        out.append(abs(gap) > z * acc_se(s1, s2))
    return int(sum(out))


def single_verifier(models, z=Z_SAMP):
    R = {r["model"]: r for r in json.load(open("results/flipbudget/rebuild_results.json"))["per_model"]}
    L4 = lvl(4)
    n = 0
    for m1, m2 in combinations(models, 2):
        rng = []
        for m in (m1, m2):
            v = R[m]
            ql, qh = cp(v["x_q"], v["n_q"], L4)
            rl, rh = cp(v["x_r"], v["n_r"], L4)
            a = v["a_sb"]
            rng.append((a * (1 - qh) + (1 - a) * rl, a * (1 - ql) + (1 - a) * rh))
        lo, hi = rng[0][0] - rng[1][1], rng[0][1] - rng[1][0]
        se = np.sqrt(sum(R[m]["a_sb"] * (1 - R[m]["a_sb"]) / R[m]["N"] for m in (m1, m2)))
        n += (lo - z * se > 0) or (hi + z * se < 0)
    return int(n)


# ---------------------------------------------------------------- design simulation
def design_alloc(design, models, unl, shares, budget):
    """Per (model, cell) label counts, capped at unlabeled rows, shortfall redistributed."""
    cap = {(m, c): unl[m][c] for m in models for c in CELLS}
    if design == "uniform":            # SRS of all unlabeled rows: expected proportional
        w = {k: v for k, v in cap.items()}
    elif design == "one_verifier":     # equal per (model, score_boxed stratum), cells follow
        w = {}
        for m in models:
            cred = unl[m]["11"] + unl[m]["10"]
            rej = unl[m]["00"] + unl[m]["01"]
            for c in CELLS:
                tot = cred if c in ("11", "10") else rej
                w[(m, c)] = (unl[m][c] / tot) if tot else 0.0
    elif design == "prop_disagreement_plus_check":  # the preregistered design: 358 + 100
        return {**_prop({k: v for k, v in cap.items() if k[1] in DIS}, budget - 100, cap),
                **_even({k: v for k, v in cap.items() if k[1] in ("11", "00")}, 100, cap)}
    elif design == "full_disagreement":
        w = {k: v for k, v in cap.items() if k[1] in DIS}
    elif design == "width_min_disagreement":   # n ~ share^(2/3): minimizes sum s/sqrt(n)
        w = {(m, c): shares[m][c] ** (2 / 3) for m in models for c in DIS if cap[(m, c)]}
    elif design == "balanced_check":  # half disagreement (proportional), half concordant (even by model)
        half = budget // 2
        return {**_prop({k: v for k, v in cap.items() if k[1] in DIS}, budget - half, cap),
                **_even({k: v for k, v in cap.items() if k[1] in ("11", "00")}, half, cap)}
    elif design == "concordant_only":
        w = {k: v for k, v in cap.items() if k[1] in ("11", "00")}
    else:
        raise ValueError(design)
    return _prop(w, budget, cap)


def _prop(w, budget, cap):
    n = {k: 0 for k in w}
    remaining = budget
    active = {k for k in w if w[k] > 0 and cap[k] > 0}
    while remaining > 0 and active:
        tot = sum(w[k] for k in active)
        add = {k: min(cap[k] - n[k], int(np.floor(remaining * w[k] / tot))) for k in active}
        if sum(add.values()) == 0:  # hand out singles by weight
            for k in sorted(active, key=lambda k: -w[k])[:remaining]:
                add[k] = min(1, cap[k] - n[k])
        for k, v in add.items():
            n[k] += v
        remaining = budget - sum(n.values())
        active = {k for k in active if n[k] < cap[k]}
    return n


def _even(strata, budget, cap):
    n = {k: 0 for k in strata}
    order = sorted(strata)
    i = 0
    while sum(n.values()) < budget and any(n[k] < cap[k] for k in order):
        k = order[i % len(order)]
        if n[k] < cap[k]:
            n[k] += 1
        i += 1
    return n


def simulate_design(design, models, unl, shares, pooled, pm, scen, rng):
    alloc = design_alloc(design, models, unl, shares, BUDGET)
    res = []
    for _ in range(N_REP):
        p11 = rng.beta(pooled["11"][0] + .5, pooled["11"][1] - pooled["11"][0] + .5)
        p00 = rng.beta(pooled["00"][0] + .5, pooled["00"][1] - pooled["00"][0] + .5)
        truth = {}
        for m in models:
            truth[(m, "10")], truth[(m, "01")] = scen(rng)
        x11 = pooled["11"][0] + sum(rng.binomial(alloc.get((m, "11"), 0), p11) for m in models)
        n11 = pooled["11"][1] + sum(alloc.get((m, "11"), 0) for m in models)
        x00 = pooled["00"][0] + sum(rng.binomial(alloc.get((m, "00"), 0), p00) for m in models)
        n00 = pooled["00"][1] + sum(alloc.get((m, "00"), 0) for m in models)
        k = 6  # 2 pooled + up to 4 per-pair disagreement parameters
        b11, b00 = cp(x11, n11, lvl(k)), cp(x00, n00, lvl(k))
        dcount = {}
        for m in models:
            for c in DIS:
                n_new = alloc.get((m, c), 0)
                x = pm.get(m, {}).get(c, [0, 0])[0] + rng.binomial(n_new, truth[(m, c)])
                n = pm.get(m, {}).get(c, [0, 0])[1] + n_new
                dcount[(m, c)] = cp(x, n, lvl(k))
        r = evaluate(shares, models, ("shared", b11, b00), lambda m, c: dcount[(m, c)])
        res.append(r["combined_resolved"])
    res = np.array(res)
    # falsification power: P(at least one concordant error observed for a model whose
    # concordant error rate is RHO in both concordant cells), existing + new labels
    RHO = 0.10
    power = [1 - (1 - RHO) ** (sum(pm.get(m, {}).get(c, [0, 0])[1] for c in ("11", "00"))
                               + alloc.get((m, "11"), 0) + alloc.get((m, "00"), 0)) for m in models]
    return {"design": design, "labels": int(sum(alloc.values())),
            "falsification_power_rho10_median": float(np.median(power)),
            "falsification_power_rho10_min": float(min(power)),
            "labels_on_disagreement": int(sum(v for k, v in alloc.items() if k[1] in DIS)),
            "median": float(np.median(res)), "p10": float(np.quantile(res, .1)),
            "p90": float(np.quantile(res, .9)), "mean": float(res.mean())}


def main():
    shares, cellmap, unl = census()
    models = sorted(r["model"] for r in json.load(open("results/flipbudget/rebuild_results.json"))["per_model"])
    out = {"models": models, "census_shares": {m: shares[m] for m in models},
           "coverage": "per ordering >= 95%: audit 0.975 (Clopper-Pearson, Sidak over estimated params) "
                       "+ sampling 0.975, union bound",
           "n_rep": N_REP, "budget": BUDGET}
    out["sampling_ceiling"] = sampling_ceiling(shares, models)
    out["sampling_ceiling_simultaneous"] = sampling_ceiling(shares, models, Z_SAMP_SIM)
    out["single_verifier_combined"] = single_verifier(models)
    print("ceiling", out["sampling_ceiling"], "(simultaneous", out["sampling_ceiling_simultaneous"], ")",
          "single-verifier", out["single_verifier_combined"])
    rng = np.random.default_rng(SEED)
    for indet in ("as_incorrect", "drop"):
        pooled, pm = audit_counts(cellmap, indet)
        r = {"audit_cells": {c: {"x_correct": pooled[c][0], "n": pooled[c][1]} for c in CELLS},
             "per_model_cells": pm}
        b11, b00 = cp(*pooled["11"], lvl(2)), cp(*pooled["00"], lvl(2))
        r["concordant_boxes_k2"] = {"11": b11, "00": b00}
        bound = lambda m, c: (0.0, 1.0)  # noqa: E731
        r["shared"] = evaluate(shares, models, ("shared", b11, b00), bound)
        r["shared_simultaneous"] = evaluate(shares, models, ("shared", b11, b00), bound, Z_SAMP_SIM)
        r["hetero"] = []
        for L in L_GRID:
            h11 = (max(0.0, 1 - L * (1 - b11[0])), 1.0)
            h00 = (0.0, min(1.0, L * b00[1]))
            e = evaluate(shares, models, ("hetero", h11, h00), bound)
            r["hetero"].append({"L": L, "box11": h11, "box00": h00, "audit": e["audit_resolved"],
                                "combined": e["combined_resolved"]})
        pm11 = {m: cp(*pm.get(m, {}).get("11", [0, 0]), lvl(4)) for m in models}
        pm00 = {m: cp(*pm.get(m, {}).get("00", [0, 0]), lvl(4)) for m in models}
        r["per_model_concordant"] = {k: v for k, v in evaluate(shares, models, ("permodel", pm11, pm00), bound).items()
                                     if k != "pairs"}
        r["per_model_concordant_counts"] = {m.split("/")[-1]: {c: pm.get(m, {}).get(c, [0, 0]) for c in ("11", "00")}
                                            for m in models}
        print("   per-model concordant (no pooling): combined", r["per_model_concordant"]["combined_resolved"])
        # labels needed without the homogeneity assumption: add n zero-error concordant labels
        # per model per concordant cell (disagreement cells still [0,1])
        r["nonparametric_projection"] = []
        for n_add in (25, 50, 100, 200, 400, 800):
            b11n = {m: cp(pm.get(m, {}).get("11", [0, 0])[0] + n_add, pm.get(m, {}).get("11", [0, 0])[1] + n_add, lvl(4))
                    for m in models}
            b00n = {m: cp(pm.get(m, {}).get("00", [0, 0])[0], pm.get(m, {}).get("00", [0, 0])[1] + n_add, lvl(4))
                    for m in models}
            e = evaluate(shares, models, ("permodel", b11n, b00n), bound)
            r["nonparametric_projection"].append({"added_per_model_per_cell": n_add,
                                                  "labels_total": 2 * n_add * len(models),
                                                  "combined": e["combined_resolved"]})
        print("   nonparametric projection:", [(x["labels_total"], x["combined"]) for x in r["nonparametric_projection"]])
        b4 = {c: cp(*pooled[c], lvl(4)) for c in CELLS}
        r["pooled_dis"] = evaluate(shares, models, ("shared", b4["11"], b4["00"]),
                                   lambda m, c: b4[c])
        p10 = pooled["10"][0] / pooled["10"][1]
        p01 = pooled["01"][0] / pooled["01"][1] if pooled["01"][1] else 0.5
        cen = evaluate(shares, models, ("shared", b11, b00), lambda m, c: (p10, p10) if c == "10" else (p01, p01))
        missing = [p for p in cen["pairs"] if not p["combined"]]
        samp_ok = {tuple(p["pair"]) for p in cen["pairs"]}
        r["census_projection"] = {k: v for k, v in cen.items() if k != "pairs"}
        r["census_projection"]["not_resolved"] = [
            {"pair": [x.split("/")[-1] for x in p["pair"]], "lo": p["lo"], "hi": p["hi"],
             "sampling_halfwidth": Z_SAMP * p["se"],
             "sampling_alone_resolves": bool(abs((shares[p["pair"][0]]["11"] + shares[p["pair"][0]]["10"])
                                                 - (shares[p["pair"][1]]["11"] + shares[p["pair"][1]]["10"]))
                                             > Z_SAMP * p["se"])} for p in missing]
        del samp_ok
        print(f"[{indet}] cells {r['audit_cells']}")
        print(f"   shared: audit {r['shared']['audit_resolved']}, combined {r['shared']['combined_resolved']}, "
              f"simultaneous {r['shared_simultaneous']['combined_resolved']}")
        print("   hetero L:", [(h["L"], h["audit"], h["combined"]) for h in r["hetero"]])
        print(f"   pooled_dis combined {r['pooled_dis']['combined_resolved']}; census combined "
              f"{r['census_projection']['combined_resolved']}")
        if indet == "as_incorrect":
            scen = {"p10~Beta(audit),p01~U(0,1)": lambda g: (g.beta(pooled["10"][0] + .5, pooled["10"][1] - pooled["10"][0] + .5), g.uniform()),
                    "p10=0.5,p01=0.5": lambda g: (0.5, 0.5),
                    "p10=0.9,p01=0.1": lambda g: (0.9, 0.1)}
            r["designs"] = {s: [simulate_design(d, models, unl, shares, pooled, pm, f, rng)
                                for d in ["uniform", "one_verifier", "concordant_only", "balanced_check", "prop_disagreement_plus_check",
                                          "width_min_disagreement", "full_disagreement"]]
                            for s, f in scen.items()}
            for s, v in r["designs"].items():
                print("   ", s, [(d["design"], d["labels_on_disagreement"], d["median"], d["p10"], d["p90"]) for d in v])
        out[indet] = r
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
