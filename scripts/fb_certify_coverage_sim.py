"""G2.3 -- coverage simulation for the certification procedure (validates; proves nothing).

Frame: the real K=3 census (27 models x 1,324 items, verdict cells) and the real old-audit
allocation (SRSWOR within (boxed stratum, subject), pooled across models). Per replicate:
  1. truth, two worlds.
     finite (validity check): each concordant (stratum h, model m) sub-population has
       floor(eps_true * N_g) * u_g human-judged errors, u_g in {0, 1} at random, so the realized
       error fractions span [0, eps_true] and finite H(eps_true) holds exactly;
     propensity (diagnostic): concordant errors are Bernoulli(0.01) / Bernoulli(0.005) with
       homogeneous propensities, so finite H(0) is violated by sampling noise alone;
     discordant cells: Y* ~ Bernoulli(q_mc), q_mc ~ U(0, 1);
  2. benchmark: 1,324 items resampled with replacement from the frame, paired across models;
  3. audit: SRSWOR per stratum with the real allocation; pooled KL intervals (Sidak 2, 0.975);
  4. certification at eps_analysis with the primary configuration of fb_certify.py.
Errors: an audit-only claim is wrong if sign(claim) != sign(Delta_F) (finite benchmark);
a combined claim is wrong if sign(claim) != sign(delta_S) (superpopulation expectation).
Nominal: per ordering <= 0.05 (combined) / 0.025 (audit-only); simultaneous <= 0.05 / 0.025.

Run: python scripts/fb_certify_coverage_sim.py   (background; ~15-30 min)
Out: results/flipbudget/certify_coverage_sim.json
"""
import gzip
import json
import math
import sys
from collections import Counter
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flipbudget import certify as C  # noqa: E402

OUT = ROOT / "results" / "flipbudget" / "certify_coverage_sim.json"
N_REP = 2000
SEED = 20260922
ALPHA_A = ALPHA_S = 0.025
P_TOP, P_BOT = 0.99, 0.005
SCENARIOS = [  # (K, eps_true, eps_analysis); eps_true < 0 = propensity world (diagnostic)
    (3, 0.0, 0.0), (3, 0.01, 0.01), (2, 0.0, 0.0), (2, 0.01, 0.01),   # validity: finite H holds
    (3, 0.02, 0.0),                                                    # misspecified eps
    (3, -1, 0.0), (3, -1, 0.01), (2, -1, 0.0), (2, -1, 0.01),          # homogeneous propensities
]
FR = {}


def build_frame():
    census = json.load(gzip.open(ROOT / "results" / "flipbudget" / "census_k3.json.gz", "rt", encoding="utf-8"))
    models = census["models"]
    items = sorted({r["item_id"] for r in census["rows"]})
    mi, ii = {m: n for n, m in enumerate(models)}, {i: n for n, i in enumerate(items)}
    M, N = len(models), len(items)
    V = {f: np.zeros((M, N), np.int8) for f in ("em", "sb_sympy", "mv")}
    strat = np.zeros((M, N), np.int16)
    labels = {}
    for r in census["rows"]:
        a, b = mi[r["model"]], ii[r["item_id"]]
        for f in V:
            V[f][a, b] = r[f]
        s = "unparsed" if not r["sb_found"] else ("credited" if r["sb_string"] else "wrong_parsed")
        strat[a, b] = labels.setdefault((s, r["subject"]), len(labels))
    key = json.load(open(ROOT / "results" / "analysis" / "mathhard_labelling_key.json", encoding="utf-8"))
    alloc = Counter(labels[(k["stratum"], k["subject"])] for k in key)
    return {"models": models, "V": V, "strat": strat, "n_strata": len(labels), "alloc": dict(alloc)}


def _init():
    FR.update(build_frame())


def cells_of(K, idx_items):
    V = FR["V"]
    fields = ("sb_sympy", "em") if K == 2 else ("em", "sb_sympy", "mv")
    code = np.zeros_like(V["em"][:, idx_items], dtype=np.int16)
    for f in fields:
        code = code * 2 + V[f][:, idx_items]
    return code, fields


def replicate(args):
    K, eps_true, eps_an, rep = args
    rng = np.random.default_rng([SEED, K, int(round((eps_true + 2) * 1e4)), int(round(eps_an * 1e4)), rep])
    models, strat, H = FR["models"], FR["strat"], FR["n_strata"]
    M, N = strat.shape
    ncell = 2 ** K
    top, bot = ncell - 1, 0
    world = "propensity" if eps_true < 0 else "finite"
    eps_t = max(eps_true, 0.0)
    q = rng.uniform(0, 1, (M, ncell))           # discordant rates per (model, cell)
    u = rng.integers(0, 2, (H, M, ncell))       # which concordant sub-populations sit at the band's far end
    code_all, fields = cells_of(K, np.arange(N))
    items = rng.integers(0, N, N)               # benchmark: items resampled, paired across models
    code, _ = cells_of(K, items)
    st = strat[:, items]
    rows_m = np.broadcast_to(np.arange(M)[:, None], (M, N))

    def truth(codes, strata, realize):
        """Expected (realize=False) or realized Y* per row.
        finite: each concordant (h, m) sub-population gets floor(eps*N_g)*u errors, so its realized
                error fraction lies in [0, eps]: finite H(eps) holds exactly;
        propensity: concordant errors ~ Bernoulli(1 - P_TOP) / Bernoulli(P_BOT) with homogeneous
                propensities: finite H(eps) is violated by sampling noise alone."""
        exp = q[rows_m, codes].astype(float)
        y = (rng.uniform(size=codes.shape) < exp).astype(np.int8) if realize else None
        for c, good in ((top, 1), (bot, 0)):
            mask = codes == c
            if world == "propensity":
                pr = P_TOP if good else P_BOT
                exp[mask] = pr
                if realize:
                    y[mask] = rng.uniform(size=int(mask.sum())) < pr
                continue
            exp[mask] = good
            if realize:
                y[mask] = good
            key = strata.astype(np.int64) * M + rows_m
            for g in np.unique(key[mask]):
                idx = np.flatnonzero((mask & (key == g)).ravel())
                n_err = int(math.floor(eps_t * idx.size)) * int(u[g // M, g % M, c])
                if n_err:
                    exp.ravel()[idx] = good + (1 - 2 * good) * n_err / idx.size
                    if realize:
                        y.ravel()[rng.choice(idx, n_err, replace=False)] = 1 - good
        return exp, y

    exp_frame, _ = truth(code_all, strat, realize=False)
    mu = exp_frame.mean(axis=1)                 # superpopulation target delta (items uniform over frame)
    _, y = truth(code, st, realize=True)
    A = y.mean(axis=1)                          # finite-benchmark target Delta
    # 3. audit: SRSWOR per stratum over all (model, item) rows of the benchmark
    x = {top: 0, bot: 0}
    n = {top: 0, bot: 0}
    flat_st, flat_code, flat_y = st.ravel(), code.ravel(), y.ravel()
    for h, nh in FR["alloc"].items():
        rows = np.flatnonzero(flat_st == h)
        take = rng.choice(rows, min(nh, rows.size), replace=False)
        for c in (top, bot):
            sel = take[flat_code[take] == c]
            x[c] += int(flat_y[sel].sum())
            n[c] += int(sel.size)
    lvl = C.sidak(2, 1 - ALPHA_A)
    fmt = lambda c: format(c, f"0{K}b")  # noqa: E731
    conc = {fmt(c): C.kl_interval(x[c], n[c], lvl) for c in (top, bot)}
    disc = {fmt(c): (0.0, 1.0) for c in range(ncell) if c not in (top, bot)}
    shares = {}
    for a, m in enumerate(models):
        cnt = np.bincount(code[a], minlength=ncell) / N
        shares[m] = {fmt(c): float(cnt[c]) for c in range(ncell)} | {"N": N}
    # 4. certification
    vbench = {f: FR["V"][f][:, items] for f in fields}
    pb = C.model_p_bounds(conc, disc, eps_an)
    ebar = {m: {f: C.error_bound(shares[m], {fmt(c): (c >> (K - 1 - n_)) & 1 for c in range(ncell)}, pb)
                for n_, f in enumerate(fields)} for m in models}
    npairs = M * (M - 1) // 2
    z1, zs = float(norm.ppf(1 - ALPHA_S / 2)), float(norm.ppf(1 - ALPHA_S / 2 / npairs))
    wrong = {"audit": [], "comb": [], "comb_sim": []}
    certified = {"audit": 0, "comb": 0, "comb_sim": 0}
    for a, b in combinations(range(M), 2):
        j, k = models[a], models[b]
        lo, hi = C.pair_range(shares[j], shares[k], conc, disc, disc, eps_an)
        d = {f: float(np.mean(vbench[f][a] != vbench[f][b])) for f in fields}
        fbar = min(1.0, min(d[f] + ebar[j][f] + ebar[k][f] for f in fields))
        for name, h, truth in (("audit", 0.0, A[a] - A[b]), ("comb", z1 * math.sqrt(fbar / N), mu[a] - mu[b]),
                               ("comb_sim", zs * math.sqrt(fbar / N), mu[a] - mu[b])):
            if lo - h > 0 or hi + h < 0:
                certified[name] += 1
                claim = 1 if lo - h > 0 else -1
                if claim * truth <= 0:
                    wrong[name].append(a * M + b)
    return certified, wrong


def main():
    out = {"n_rep": N_REP, "seed": SEED, "p_top": P_TOP, "p_bot": P_BOT, "scenarios": []}
    with Pool(18, initializer=_init) as pool:
        for K, et, ea in SCENARIOS:
            res = pool.map(replicate, [(K, et, ea, r) for r in range(N_REP)], chunksize=10)
            per_pair = {name: Counter(p for _, w in res for p in w[name]) for name in ("audit", "comb", "comb_sim")}
            s = {"K": K, "eps_true": et, "eps_analysis": ea}
            for name, nominal in (("audit", ALPHA_A), ("comb", ALPHA_A + ALPHA_S), ("comb_sim", ALPHA_A + ALPHA_S)):
                fam = sum(1 for _, w in res if w[name]) / N_REP
                worst = max(per_pair[name].values(), default=0) / N_REP
                s[name] = {"mean_certified": float(np.mean([c[name] for c, _ in res])),
                           "familywise_error": fam, "max_per_ordering_error": worst, "nominal": nominal,
                           "mc_se_at_nominal": math.sqrt(nominal * (1 - nominal) / N_REP)}
            out["scenarios"].append(s)
            print(json.dumps(s), flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
