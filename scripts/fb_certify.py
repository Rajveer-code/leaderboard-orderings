"""G2.4 -- certification of MATH-Hard orderings on the existing audit, K=2 and K=3.

Primary configuration (G1 decisions): 27 models, faithful sympy semantics for the boxed
verifier, human reference = harness routine (sympy), indeterminate labels as incorrect,
KL intervals (Lemma 1), paired conservative sampling half-width (theory section 6).
Discordant cells are [0, 1]: the old audit is stratified across models, so a per-model
discordant estimate would target a stratum mixture, not p_mc (THEORY section 0).

Sensitivities: string semantics (historical), Math-Verify human reference, indeterminate
dropped, Clopper-Pearson intervals, legacy unpaired and Hoeffding sampling, 17-model subset,
no pooling (per-model concordant intervals).

Run: python scripts/fb_certify.py        Out: results/flipbudget/certify_results.json
"""
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flipbudget import certify as C  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
AUDIT = ROOT / "results" / "flipbudget" / "audit_k3_map.json"
OUT = Path(os.environ.get("FB_CERTIFY_OUT", ROOT / "results" / "flipbudget" / "certify_results.json"))
N_ITEMS = 1324
ALPHA_A, ALPHA_S = 0.025, 0.025
EPS0 = 0.01  # preregistered scientific tolerance (THEORY 1.1c), a judgement, not a measurement
EPS_GRID = [round(0.0025 * i, 4) for i in range(41)]  # 0 .. 0.10


V3_LABELS = Path(os.environ.get("FB_V3_OUT", ROOT / "results" / "flipbudget" / "audit_v3_labels.json"))
CENSUS_LABELS = Path(os.environ.get("FB_CENSUS_OUT", ROOT / "results" / "flipbudget" / "discordant_census_labels.json"))
XF_VERDICTS = ROOT / "results" / "flipbudget" / "xfinder_verdicts.json.gz"


def load_v3(ref, variant=None):
    """v3 label counts in Inputs' shape. Cells are the K=3 (em, sb_sympy, mv) strata of the draw.
    variant="d8" reads the analysis with extraction-only batch exclusions restored (decision D8)."""
    v3 = json.load(open(V3_LABELS, encoding="utf-8"))
    d = (v3["d8_extraction_restored"] if variant == "d8" else v3)["counts"][ref]
    per = {tuple(k.split("|")): v for k, v in d["per_model_cell"].items()}
    return {"pooled": d["pooled"],
            "disc": {k: v for k, v in per.items() if k[1] not in ("000", "111")},
            "disc_and_conc_per_model": per}


def spec(K, sem):
    """Verifier fields in cell-string order."""
    sb = f"sb_{sem}"
    return {2: (sb, "em"), 3: ("em", sb, "mv"), 4: ("em", sb, "mv", "xf")}[K]


class Inputs:
    def __init__(self, census, audit, K, sem, ref, indet, models, interval, new_labels=None, disc_fixed=None):
        """new_labels (G4.1, from scripts/fb_audit_v3_analysis.py): {"pooled": {cell: [x, n]},
        "disc": {(model, cell): [x, n]}} from the v3 audit. Concordant labels join the pooled
        cells; discordant labels give per-model intervals, which are design-valid because the
        v3 strata are (model x cell). Omitted -> the pre-v3 analysis, unchanged."""
        self.K, self.models, self.fields = K, models, spec(K, sem)
        rows = [r for r in census["rows"] if r["model"] in set(models)]
        items = sorted({r["item_id"] for r in rows})
        col = {i: n for n, i in enumerate(items)}
        self.V = {f: np.zeros((len(models), len(items)), dtype=np.int8) for f in self.fields}
        idx = {m: n for n, m in enumerate(models)}
        counts = {m: defaultdict(int) for m in models}
        for r in rows:
            for f in self.fields:
                self.V[f][idx[r["model"]], col[r["item_id"]]] = r[f]
            counts[r["model"]]["".join(str(r[f]) for f in self.fields)] += 1
        self.shares = {m: {c: v / N_ITEMS for c, v in counts[m].items()} | {"N": N_ITEMS} for m in models}
        self.cells = sorted({c for m in models for c in counts[m]})
        self.conc_cells = ["0" * K, "1" * K]
        self.idx = idx
        # audit counts (pooled per concordant cell, and per model for the no-pooling analysis)
        cell_key = f"k{K}_{sem}"
        pooled = {c: [0, 0] for c in self.conc_cells}
        per_model = {m: {c: [0, 0] for c in self.conc_cells} for m in models}
        subpop = {c: defaultdict(int) for c in self.conc_cells}
        for it in audit["items"]:
            if it["model"] not in idx or it[cell_key] not in pooled:
                continue
            if it["indeterminate"] and indet == "drop":
                continue
            y = 0 if it["indeterminate"] else it["ystar"][ref]
            c = it[cell_key]
            pooled[c][0] += y
            pooled[c][1] += 1
            per_model[it["model"]][c][0] += y
            per_model[it["model"]][c][1] += 1
            subpop[c][(it["key_stratum"], it.get("subject"), it.get("quartile"), it["model"])] += 1
        nl = new_labels or {}
        self.pooled_existing = {c: list(v) for c, v in pooled.items()}
        for c, (x, n) in (nl.get("pooled") or {}).items():
            if c in pooled:
                pooled[c][0] += x
                pooled[c][1] += n
        for m in models:  # v3 concordant labels are per (model, cell) too
            for c in self.conc_cells:
                x, n = (nl.get("disc_and_conc_per_model") or {}).get((m, c), (0, 0))
                per_model[m][c][0] += x
                per_model[m][c][1] += n
        self.pooled, self.per_model, self.subpop = pooled, per_model, subpop
        self.ivl = C.kl_interval if interval == "kl" else C.cp_interval
        ivl = self.ivl
        self.level = C.sidak(2, 1 - ALPHA_A)
        self.conc = {c: ivl(*pooled[c], self.level) for c in self.conc_cells}
        self.disc = {c: (0.0, 1.0) for c in self.cells if c not in self.conc_cells}
        # census bounds (Amendment 6): known or logically bounded, never estimated, so they take
        # precedence over sampled counts for the same domain and add no Sidak parameter
        self.disc_fixed = {k: v for k, v in (disc_fixed or {}).items() if k[1] in self.disc}
        self.disc_counts = {k: v for k, v in (nl.get("disc") or {}).items()
                            if k[1] in self.disc and k not in self.disc_fixed}
        # pairwise verifier disagreement d^v_jk (item-matched)
        self.d = {}
        for j, k in combinations(models, 2):
            a, b = idx[j], idx[k]
            self.d[(j, k)] = {f: float(np.mean(self.V[f][a] != self.V[f][b])) for f in self.fields}

    def acc(self, m, f):
        return float(self.V[f][self.idx[m]].mean())

    def n_audited_disc(self, m):
        """Number of discordant parameters of model m estimated from v3 labels."""
        return sum(1 for (mm, c) in self.disc_counts if mm == m)

    def disc_bounds(self, m, level):
        """Per-model discordant bounds: census bounds where they exist, else KL/CP from sampled
        v3 labels, else [0, 1]."""
        def one(c):
            if (m, c) in self.disc_fixed:
                return self.disc_fixed[(m, c)]
            if (m, c) in self.disc_counts:
                return self.ivl(*self.disc_counts[(m, c)], level)
            return (0.0, 1.0)
        return {c: one(c) for c in self.disc}

    def ebar(self, m, conc, eps, disc=None):
        pb = C.model_p_bounds(conc, self.disc if disc is None else disc, eps)
        return {f: C.error_bound(self.shares[m], {c: int(c[n]) for c in self.cells}, pb)
                for n, f in enumerate(self.fields)}

    def n_params(self, pair=None, pooling=True):
        """Audit parameters a statement uses: one ordering's (pair given) or the board's (None)."""
        nd = {m: self.n_audited_disc(m) for m in self.models}
        return C.audit_param_count(len(self.conc_cells), nd, pair, pooling)

    def pair_bounds(self, j, k, simultaneous=False, pooling=True):
        """(conc, disc_j, disc_k) at the Sidak level of the statement being made.

        A per-ordering statement allocates over the parameters that ordering uses; a statement
        about every ordering at once allocates over every parameter on the board. Without
        pooling, each model's concordant intervals enter as model-specific bounds, built at the
        same level as everything else in the statement.
        """
        lvl = C.sidak(self.n_params(None if simultaneous else (j, k), pooling), 1 - ALPHA_A)
        dj, dk = self.disc_bounds(j, lvl), self.disc_bounds(k, lvl)
        if pooling:
            return {c: self.ivl(*self.pooled[c], lvl) for c in self.conc_cells}, dj, dk
        own = {m: {c: self.ivl(*self.per_model[m][c], lvl) for c in self.conc_cells} for m in (j, k)}
        return {}, dj | own[j], dk | own[k]

    def halfwidth(self, mode, z, bounds=None):
        """Sampling half-width. `bounds(j, k)` must be the same function the range uses, so
        the variance bound rests on the same event as the claim it widens."""
        bounds = bounds or self.pair_bounds

        def h(j, k, eps):
            if mode == "none":
                return 0.0
            if mode == "hoeffding":
                return C.halfwidth_hoeffding(N_ITEMS, 2 * (1 - norm.cdf(z)))
            if mode == "legacy":  # unpaired plug-in from the boxed verifier's accuracy (anchor only)
                a = [self.acc(m, self.fields[0] if self.K == 2 else self.fields[1]) for m in (j, k)]
                return z * math.sqrt(sum(x * (1 - x) for x in a) / N_ITEMS)
            cj, dj, dk = bounds(j, k)
            return C.halfwidth_paired(self.d[(j, k)], self.ebar(j, cj, eps, dj), self.ebar(k, cj, eps, dk),
                                      N_ITEMS, z)
        return h


def run(inp, eps, sampling="paired", simultaneous=False, pooling=True):
    npairs = len(inp.models) * (len(inp.models) - 1) // 2
    z = float(norm.ppf(1 - ALPHA_S / 2 / (npairs if simultaneous else 1)))

    def bounds(j, k):
        return inp.pair_bounds(j, k, simultaneous=simultaneous, pooling=pooling)
    return C.evaluate(inp.models, inp.shares, bounds, inp.halfwidth(sampling, z, bounds), eps)


def ceiling(inp, simultaneous=False):
    """Orderings each verifier alone separates beyond paired sampling noise (no verifier error)."""
    npairs = len(inp.models) * (len(inp.models) - 1) // 2
    z = float(norm.ppf(1 - ALPHA_S / 2 / (npairs if simultaneous else 1)))
    return {f: sum(abs(inp.acc(j, f) - inp.acc(k, f)) > z * math.sqrt(inp.d[(j, k)][f] / N_ITEMS)
                   for j, k in combinations(inp.models, 2)) for f in inp.fields}


def sampling_pairs(inp, f, simultaneous=False):
    """(winner, loser) pairs verifier f separates beyond paired sampling noise, verifier taken as truth."""
    npairs = len(inp.models) * (len(inp.models) - 1) // 2
    z = float(norm.ppf(1 - ALPHA_S / 2 / (npairs if simultaneous else 1)))
    out = set()
    for j, k in combinations(inp.models, 2):
        gap = inp.acc(j, f) - inp.acc(k, f)
        if abs(gap) > z * math.sqrt(inp.d[(j, k)][f] / N_ITEMS):
            out.add((j, k) if gap > 0 else (k, j))
    return out


def breakdowns(inp, sampling="paired"):
    z = float(norm.ppf(1 - ALPHA_S / 2))
    h = inp.halfwidth(sampling, z)
    out = {}
    for j, k in combinations(inp.models, 2):
        conc, dj, dk = inp.pair_bounds(j, k)

        def margin(e, sign, conc=conc, dj=dj, dk=dk, j=j, k=k):
            lo, hi = C.pair_range(inp.shares[j], inp.shares[k], conc, dj, dk, e)
            return (lo if sign > 0 else -hi) - h(j, k, e)
        e_pos, e_neg = C.breakdown_eps(lambda e: margin(e, 1)), C.breakdown_eps(lambda e: margin(e, -1))
        if e_pos > 0 or e_neg > 0:
            out[f"{j} > {k}" if e_pos > 0 else f"{k} > {j}"] = max(e_pos, e_neg)
    return out


def certified_set(r):
    """(winner, loser) pairs of a run."""
    return {(p["winner"], [x for x in p["pair"] if x != p["winner"]][0]) for p in r["pairs"] if p["winner"]}


def fixed_board_summary(inp):
    """The fixed-board estimand (primary, MASTER_PLAN D1): the difference on these items, so no
    sampling term; every statement simultaneous over the roster, exact in finite samples."""
    runs = {name: run(inp, e, sampling="none", simultaneous=True, pooling=pool)
            for name, e, pool in (("eps0", 0.0, True), ("eps_preregistered", EPS0, True), ("no_pooling", 0.0, False))}
    return {**{f"certified_{k}": v["audit_resolved"] for k, v in runs.items()},
            **{f"rank_intervals_{k}": C.rank_intervals(inp.models, certified_set(v)) for k, v in runs.items()},
            "breakdown_eps": breakdowns(inp, sampling="none"),
            "eps_curve": [{"eps": e, "certified": run(inp, e, sampling="none", simultaneous=True)["audit_resolved"]}
                          for e in EPS_GRID]}


def eps_reference_a(inp):
    """THEORY 1.1(a): max_c u_c / min_g w_gc over sampled audit sub-populations."""
    refs = {}
    for c in inp.conc_cells:
        n = inp.pooled[c][1]
        if n == 0:
            continue
        err_hi = inp.conc[c][1] if c == "0" * inp.K else 1 - inp.conc[c][0]
        w_min = min(inp.subpop[c].values()) / n
        refs[c] = {"u_c": err_hi, "min_weight": w_min, "bound": err_hi / w_min,
                   "n_sampled_subpops": len(inp.subpop[c])}
    return {"per_cell": refs, "eps_ref_a": max(v["bound"] for v in refs.values())}


def summary(inp):
    return {"n_models": len(inp.models), "n_pairs": len(inp.d), "cells": inp.cells,
            "audit_pooled": {c: {"x_correct": x, "n": n} for c, (x, n) in inp.pooled.items()},
            "conc_intervals": inp.conc, "interval_level": inp.level}


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    audit = json.load(open(AUDIT, encoding="utf-8"))
    key = {k["uid"]: k for k in json.load(open(ROOT / "results" / "analysis" / "mathhard_labelling_key.json",
                                                encoding="utf-8"))}
    for it in audit["items"]:  # design strata of the old audit
        it["subject"], it["quartile"] = key[it["uid"]]["subject"], key[it["uid"]]["quartile"]
    m27, m17 = census["models"], audit["models17"]
    out = {"config": {"alpha_audit": ALPHA_A, "alpha_sampling": ALPHA_S, "eps0_preregistered": EPS0,
                      "eps_grid": EPS_GRID, "n_items": N_ITEMS}}
    for K in (2, 3):
        prim = Inputs(census, audit, K, "sympy", "sympy", "as_incorrect", m27, "kl")
        res = {"inputs": summary(prim), "ceiling": ceiling(prim), "ceiling_simultaneous": ceiling(prim, True),
               "eps_reference_a": eps_reference_a(prim)}
        res["eps_curve"] = []
        for e in EPS_GRID:
            a, s = run(prim, e), run(prim, e, simultaneous=True)
            # audit-only = the fixed-board estimand (these items, no sampling term); the
            # simultaneous version allocates the audit level over every parameter on the board
            res["eps_curve"].append({"eps": e, "audit_only": a["audit_resolved"],
                                     "audit_only_simultaneous": s["audit_resolved"],
                                     "combined": a["combined_resolved"],
                                     "combined_simultaneous": s["combined_resolved"]})
        at0 = run(prim, 0.0)
        sim0 = run(prim, 0.0, simultaneous=True)
        res["eps0_pairs"] = at0["pairs"]
        cert = lambda r: {(p["winner"], [x for x in p["pair"] if x != p["winner"]][0])  # noqa: E731
                          for p in r["pairs"] if p["winner"]}
        res["rank_intervals_simultaneous_eps0"] = C.rank_intervals(m27, cert(sim0))
        res["rank_intervals_per_ordering_eps0_NOT_FAMILYWISE"] = C.rank_intervals(m27, cert(at0))
        # F1 comparison layers, all simultaneous over the 351 orderings
        res["rank_intervals_simultaneous_eps_preregistered"] = C.rank_intervals(
            m27, cert(run(prim, EPS0, simultaneous=True)))
        res["rank_intervals_simultaneous_no_pooling"] = C.rank_intervals(
            m27, cert(run(prim, 0.0, simultaneous=True, pooling=False)))
        res["rank_intervals_sampling_only"] = {f: C.rank_intervals(m27, sampling_pairs(prim, f, True))
                                               for f in prim.fields}
        res["breakdown_eps"] = breakdowns(prim)
        res["no_pooling"] = {k: v for k, v in run(prim, 0.0, pooling=False).items() if k != "pairs"}
        res["fixed_board"] = fixed_board_summary(prim)
        # the ranking a leaderboard prints: the published verifier taken as truth, no
        # uncertainty; tied accuracies share a rank interval
        acc = {m: prim.acc(m, "em") for m in m27}
        res["published_ranks"] = {m: (1 + sum(acc[k] > acc[m] for k in m27), sum(acc[k] >= acc[m] for k in m27))
                                  for m in m27}
        if K == 3 and V3_LABELS.exists():  # observed v3 audit (its strata are K=3 cells)
            with_v3 = Inputs(census, audit, K, "sympy", "sympy", "as_incorrect", m27, "kl",
                             new_labels=load_v3("sympy"))
            res["with_v3"] = {
                "inputs": summary(with_v3),
                "n_new_labels": sum(n for _, n in load_v3("sympy")["disc_and_conc_per_model"].values()),
                "eps_curve": [{"eps": e, "audit_only": run(with_v3, e)["audit_resolved"],
                               "audit_only_simultaneous": run(with_v3, e, sampling="none",
                                                              simultaneous=True)["audit_resolved"],
                               "combined": run(with_v3, e)["combined_resolved"],
                               "combined_simultaneous": run(with_v3, e, simultaneous=True)["combined_resolved"]}
                              for e in EPS_GRID],
                "rank_intervals_simultaneous_eps0": C.rank_intervals(m27, cert(run(with_v3, 0.0, simultaneous=True))),
                "breakdown_eps": breakdowns(with_v3),
                "no_pooling": {k: v for k, v in run(with_v3, 0.0, pooling=False).items() if k != "pairs"},
                "fixed_board": fixed_board_summary(with_v3)}
            if "d8_extraction_restored" in json.load(open(V3_LABELS, encoding="utf-8")):
                with_v3_d8 = Inputs(census, audit, K, "sympy", "sympy", "as_incorrect", m27, "kl",
                                    new_labels=load_v3("sympy", "d8"))
                res["with_v3"]["d8_extraction_restored"] = {
                    "fixed_board": fixed_board_summary(with_v3_d8),
                    "combined_eps0_prereg": [run(with_v3_d8, e)["combined_resolved"] for e in (0.0, EPS0)]}
            print("   with v3 labels:", [(x["eps"], x["combined"]) for x in res["with_v3"]["eps_curve"][::4]])
        if K == 3 and CENSUS_LABELS.exists():  # Amendment 6: the primary analysis once labels exist
            cl = json.load(open(CENSUS_LABELS, encoding="utf-8"))
            res["census"] = {}
            for variant, domains in cl["bounds_input"].items():
                _, ref, indet_mode = variant.split("|")  # e.g. gate|sympy|as_incorrect
                fixed_b = {tuple(k.split("|")): C.census_bounds(v["x"], v["n"], v["size"])
                           for k, v in domains.items()}
                inp_c = Inputs(census, audit, K, "sympy", ref, "as_incorrect" if indet_mode == "as_incorrect" else "drop",
                               m27, "kl", new_labels=load_v3(ref) if V3_LABELS.exists() else None,
                               disc_fixed=fixed_b)
                res["census"][variant] = {
                    "completeness": cl["completeness"][variant],
                    "fixed_board": fixed_board_summary(inp_c),
                    "generalising_eps_curve": [{"eps": e, "combined": run(inp_c, e)["combined_resolved"],
                                                "combined_simultaneous": run(inp_c, e, simultaneous=True)
                                                ["combined_resolved"]} for e in EPS_GRID]}
            print("   census:", {v: r["fixed_board"]["certified_eps_preregistered"] for v, r in res["census"].items()})
        sens = {}
        for name, args in {
            "string_semantics": (K, "string", "string", "as_incorrect", m27, "kl"),
            "mv_reference": (K, "sympy", "mv", "as_incorrect", m27, "kl"),
            "indeterminate_dropped": (K, "sympy", "sympy", "drop", m27, "kl"),
            "clopper_pearson": (K, "sympy", "sympy", "as_incorrect", m27, "cp"),
            "models17": (K, "sympy", "sympy", "as_incorrect", m17, "kl"),
        }.items():
            inp = Inputs(census, audit, *args)
            r0 = run(inp, 0.0)
            sens[name] = {"audit_pooled": summary(inp)["audit_pooled"], "audit_only": r0["audit_resolved"],
                          "combined": r0["combined_resolved"], "ceiling": ceiling(inp),
                          "combined_eps0": [run(inp, e)["combined_resolved"] for e in (0.0, EPS0)],
                          # fixed-board, simultaneous, at eps = 0 and at the registered tolerance
                          "fixed_board": [run(inp, e, sampling="none", simultaneous=True)["audit_resolved"]
                                          for e in (0.0, EPS0)]}
        for mode in ("legacy", "hoeffding"):
            r0 = run(prim, 0.0, sampling=mode)
            sens[f"sampling_{mode}"] = {"combined": r0["combined_resolved"]}
        res["sensitivity"] = sens
        out[f"K{K}"] = res
        c0 = res["eps_curve"][0]
        print(f"K={K}: audit-only {c0['audit_only']}, combined {c0['combined']}, simultaneous "
              f"{c0['combined_simultaneous']}, ceiling {res['ceiling']}, eps_ref_a {res['eps_reference_a']['eps_ref_a']:.4f}")
        print("   eps curve:", [(x["eps"], x["combined"]) for x in res["eps_curve"][::4]])
        print("   no pooling:", res["no_pooling"], "| n breakdowns:", len(res["breakdown_eps"]))
        print("   sensitivity:", {k: (v.get("audit_only"), v["combined"]) for k, v in sens.items()})
    if XF_VERDICTS.exists():  # T17 / decision D6: a fourth, learned verifier (xFinder)
        xfv = json.load(gzip.open(XF_VERDICTS, "rt", encoding="utf-8"))
        out["K4"] = {"primary_arm": xfv["primary_arm"]}
        for arm in xfv["arms"]:
            v = xfv["verdicts"][arm]
            for r in census["rows"]:
                r["xf"] = v[f"{r['model']}|{r['item_id']}"]["verdict"]
            for it in audit["items"]:
                it["k4_sympy"] = it["k3_sympy"] + str(v[f"{it['model']}|{it['item_id']}"]["verdict"])
            inp4 = Inputs(census, audit, 4, "sympy", "sympy", "as_incorrect", m27, "kl")
            cells4 = Counter("".join(str(r[f]) for f in inp4.fields) for r in census["rows"])
            out["K4"][arm] = {
                "inputs": summary(inp4), "cells": dict(sorted(cells4.items())),
                "n_discordant": sum(n for c, n in cells4.items() if c not in ("0000", "1111")),
                "fixed_board": fixed_board_summary(inp4),
                "generalising_eps_curve": [{"eps": e, "combined": run(inp4, e)["combined_resolved"],
                                            "combined_simultaneous": run(inp4, e, simultaneous=True)
                                            ["combined_resolved"]} for e in (0.0, EPS0)]}
            print(f"K=4 ({arm}): discordant {out['K4'][arm]['n_discordant']}, fixed-board at eps0 "
                  f"{out['K4'][arm]['fixed_board']['certified_eps_preregistered']}")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
