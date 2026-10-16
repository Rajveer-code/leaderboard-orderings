"""G3.1 -- certification-falsification frontier for the new K=3 audit (PROJECTED, not observed).

Design family: B = 458 labels on unlabeled rows, stratified by (model, K=3 cell):
  falsification: n0 labels per concordant cell (000, 111) for each of the F models appearing
    most in fragile orderings (smallest breakdown eps*, certify_results.json K3);
  certification: the remaining B - 2*F*n0 labels on discordant (model, cell) strata,
    proportional to stratum size, capped at capacity.
Falsification rule (THEORY 7, fixes plan G3.5): reject H(eps0) in cell c if, for some audited
model m, LB(e_mc) - UB(e_c pooled) > eps0 (KL bounds, Bonferroni over 2F tests + pooled, 0.05).
Valid because min_m e_mc <= pooled e_c <= UB. Power is exact (binomial, which understates the
without-replacement power), with the pooled UB at its projected value.
Certification projection: discordant truths from two scenarios (p ~ U(0,1) per stratum; p = 0.5),
concordant truths at the existing pooled estimates; 100 replicates; certified orderings at
eps = 0 and eps0 = 0.01, K=3 primary configuration, per-model discordant KL intervals (valid:
the new design is stratified by model x cell).
Run: python scripts/fb_g3_frontier.py   Out: results/flipbudget/g3_frontier.json
"""
import gzip
import json
import sys
from collections import Counter
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import certify as C  # noqa: E402
from fb_certify import ALPHA_A, ALPHA_S, EPS0, N_ITEMS, Inputs  # noqa: E402

OUT = ROOT / "results" / "flipbudget" / "g3_frontier.json"
BUDGET = 458
ALPHA_F = 0.05
N_REP = 100
SEED = 20260922
F_GRID = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
N0_GRID = [4, 8, 12, 16, 20, 24]
RHO_GRID = [round(0.01 * i, 2) for i in range(1, 61)]


def fragile_models(cres):
    """Models ordered by how often they appear in the most fragile certified orderings."""
    b = cres["K3"]["breakdown_eps"]
    cut = float(np.median(list(b.values())))
    cnt = Counter(m for pair, e in b.items() if e <= cut for m in pair.split(" > "))
    return [m for m, _ in cnt.most_common()]


def fals_level(F):
    """Two-sided level whose one-sided tail is alpha_F / (2F + 1): 2F per-model tests + pooled."""
    return 1 - 2 * ALPHA_F / (2 * max(F, 1) + 1)


def pooled_error_ub(inp, F, c):
    """Upper bound on the pooled concordant error of cell c, at the falsification level."""
    x, n = inp.pooled[c]
    err = x if c == "0" * inp.K else n - x
    return C.kl_interval(err, n, fals_level(F))[1]


def reject_threshold(n0, F, ub_pool):
    """Smallest error count x (of n0) whose KL lower bound exceeds eps0 + pooled UB."""
    lvl = fals_level(F)
    for x in range(n0 + 1):
        if C.kl_interval(x, n0, lvl)[0] - ub_pool > EPS0:
            return x
    return None


def power(n0, F, ub_pool, rho):
    x = reject_threshold(n0, F, ub_pool)
    return 0.0 if x is None else float(binom.sf(x - 1, n0, rho))


def detectable_rho(n0, F, ub_pool, target=0.8):
    return next((r for r in RHO_GRID if power(n0, F, ub_pool, r) >= target), None)


def allocate(cap, total):
    """Largest-remainder proportional allocation, capped at capacity."""
    n = {k: 0 for k in cap}
    left = total
    while left > 0:
        act = {k: cap[k] - n[k] for k in cap if cap[k] > n[k]}
        if not act:
            break
        tot = sum(act.values())
        raw = {k: left * v / tot for k, v in act.items()}
        add = {k: min(act[k], int(raw[k])) for k in act}
        if sum(add.values()) == 0:
            for k in sorted(act, key=lambda k: raw[k], reverse=True)[:left]:
                add[k] = 1
        for k, v in add.items():
            n[k] += v
        left = total - sum(n.values())
    return n


def project(inp, alloc_disc, conc_new, scen, rng, eps):
    """Certified combined orderings for one simulated label outcome."""
    models = inp.models
    x_pool = {c: inp.pooled[c][0] + rng.binomial(conc_new[c], inp.pooled[c][0] / inp.pooled[c][1])
              for c in inp.conc_cells}
    n_pool = {c: inp.pooled[c][1] + conc_new[c] for c in inp.conc_cells}
    counts = {}
    for (m, c), n in alloc_disc.items():
        p = rng.uniform() if scen == "uniform" else 0.5
        counts[(m, c)] = (int(rng.binomial(n, p)), n)
    z = float(norm.ppf(1 - ALPHA_S / 2))
    cert = 0
    conc_cache, eb_cache = {}, {}
    for j, k in combinations(models, 2):
        audited = [(m, c) for m in (j, k) for c in inp.disc if counts.get((m, c), (0, 0))[1] > 0]
        kk = 2 + len(audited)
        lvl = C.sidak(kk, 1 - ALPHA_A)
        if kk not in conc_cache:
            conc_cache[kk] = {c: C.kl_interval(x_pool[c], n_pool[c], lvl) for c in inp.conc_cells}
        conc = conc_cache[kk]
        dj = {c: C.kl_interval(*counts[(j, c)], lvl) if (j, c) in audited else (0.0, 1.0) for c in inp.disc}
        dk = {c: C.kl_interval(*counts[(k, c)], lvl) if (k, c) in audited else (0.0, 1.0) for c in inp.disc}
        lo, hi = C.pair_range(inp.shares[j], inp.shares[k], conc, dj, dk, eps)
        for m, dm in ((j, dj), (k, dk)):
            if (m, kk) not in eb_cache:
                eb_cache[(m, kk)] = inp_ebar(inp, m, conc, dm, eps)
        ej, ek = eb_cache[(j, kk)], eb_cache[(k, kk)]
        h = C.halfwidth_paired(inp.d[(j, k)], ej, ek, N_ITEMS, z)
        cert += (lo - h > 0) or (hi + h < 0)
    return cert


def inp_ebar(inp, m, conc, disc_m, eps):
    pb = C.model_p_bounds(conc, disc_m, eps)
    return {f: C.error_bound(inp.shares[m], {c: int(c[n]) for c in inp.cells}, pb) for n, f in enumerate(inp.fields)}


CTX = {}


def _init():
    census = json.load(gzip.open(ROOT / "results" / "flipbudget" / "census_k3.json.gz", "rt", encoding="utf-8"))
    audit = json.load(open(ROOT / "results" / "flipbudget" / "audit_k3_map.json", encoding="utf-8"))
    key = {k["uid"]: k for k in json.load(open(ROOT / "results" / "analysis" / "mathhard_labelling_key.json",
                                                encoding="utf-8"))}
    for it in audit["items"]:
        it["subject"], it["quartile"] = key[it["uid"]]["subject"], key[it["uid"]]["quartile"]
    cres = json.load(open(ROOT / "results" / "flipbudget" / "certify_results.json", encoding="utf-8"))
    inp = Inputs(census, audit, 3, "sympy", "sympy", "as_incorrect", census["models"], "kl")
    order = fragile_models(cres)
    order += [m for m in inp.models if m not in order]
    cap = Counter()  # unlabeled rows per (model, cell)
    for r in census["rows"]:
        if not r["already_labeled"]:
            cap[(r["model"], f"{r['em']}{r['sb_sympy']}{r['mv']}")] += 1
    CTX.update(inp=inp, order=order, cap=cap)


def point(args):
    """args = (F, n0) with n0 an int, or (F, {cell: n0}) for the chosen operating point."""
    F, n0 = args
    per_cell = n0 if isinstance(n0, dict) else None
    inp, order, cap = CTX["inp"], CTX["order"], CTX["cap"]
    rng = np.random.default_rng([SEED, F, sum(n0.values()) if per_cell else n0])
    fals = order[:F]
    n0_of = (lambda c: per_cell[c]) if per_cell else (lambda c: n0)
    conc_new = {c: sum(min(n0_of(c), cap[(m, c)]) for m in fals) for c in inp.conc_cells}
    disc_cap = {(m, c): cap[(m, c)] for m in inp.models for c in inp.disc if cap[(m, c)] > 0}
    alloc_disc = allocate(disc_cap, BUDGET - sum(conc_new.values()))
    pt = {"F": F, "n0": (per_cell if per_cell else n0) if F else 0, "falsification_models": fals,
          "is_operating_point": bool(per_cell),
          "labels_concordant": sum(conc_new.values()), "labels_discordant": sum(alloc_disc.values()),
          "falsification": {}}
    for c in inp.conc_cells:
        ub = pooled_error_ub(inp, F, c)
        nc = n0_of(c) if F else 0
        pt["falsification"][c] = {"n0": nc, "pooled_error_ub": ub,
                                  "reject_if_errors_at_least": reject_threshold(nc, F, ub) if F else None,
                                  "power_rho10": power(nc, F, ub, 0.1) if F else 0.0,
                                  "power_rho20": power(nc, F, ub, 0.2) if F else 0.0,
                                  "detectable_rho_80": detectable_rho(nc, F, ub) if F else None}
    for scen in ("uniform", "half"):
        for eps in (0.0, EPS0):
            v = [project(inp, alloc_disc, conc_new, scen, rng, eps) for _ in range(N_REP)]
            pt[f"certified_{scen}_eps{eps}"] = {"median": float(np.median(v)), "p10": float(np.quantile(v, .1)),
                                                "p90": float(np.quantile(v, .9))}
    return pt


def main():
    op = json.load(open(ROOT / "results" / "flipbudget" / "g3_operating_point.json", encoding="utf-8"))
    if "--operating-point-only" in sys.argv:  # evaluate the chosen point and merge it into the frontier
        _init()
        pt = point((op["F"], op["n0_by_cell"]))
        out = json.load(open(OUT, encoding="utf-8"))
        out["points"] = [p for p in out["points"] if not p.get("is_operating_point")] + [pt]
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        print(json.dumps(pt["falsification"], indent=1), "certified:",
              {k: v["median"] for k, v in pt.items() if k.startswith("certified")}, "merged into", OUT)
        return
    grid = [(F, n0) for F in F_GRID for n0 in N0_GRID
            if 2 * F * n0 <= BUDGET and not (F == 0 and n0 != N0_GRID[0])] + [(op["F"], op["n0_by_cell"])]
    with Pool(18, initializer=_init) as pool:
        points = pool.map(point, grid, chunksize=1)
    _init()
    out = {"budget": BUDGET, "eps0": EPS0, "alpha_falsification": ALPHA_F, "n_rep": N_REP,
           "fragility_order": CTX["order"], "label": "PROJECTED -- simulated label outcomes, not observed",
           "points": points}
    for pt in points:
        f = pt["falsification"]
        print(pt["F"], pt["n0"], pt["labels_concordant"],
              {c: (v["reject_if_errors_at_least"], round(v["power_rho20"], 3), v["detectable_rho_80"]) for c, v in f.items()},
              pt["certified_uniform_eps0.01"]["median"], pt["certified_half_eps0.01"]["median"])
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
