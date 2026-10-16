"""T13 -- would labelling every discordant response change the fixed-board result enough to pay?

Decision rule, fixed in MASTER_PLAN.md section 4c before this script was run: label the
remaining discordant responses only if, under both truth scenarios of the registered frontier,
the projected median gain in the fixed-board simultaneous certified count at eps = 0.01 is at
least 10% of the current count.

Two designs are projected on the same simulated truth, draw by draw:
  registered  the 458-row audit as drawn (results/analysis/audit_v3_provenance.json): its
              discordant labels give per-model intervals, each a Sidak parameter, so the
              board-wide allocation grows with them; its concordant labels join the pools.
  census      the registered audit plus a label on every discordant response: every
              discordant rate is then known exactly, adds no parameter, and enters as a point.
Truth for a discordant (model, cell) rate: uniform on [0, 1], or 0.5 (the frontier's two
scenarios). Concordant labels are simulated at the pooled rate of the existing audit, as in the
frontier. Everything is fixed-board: no sampling term; certification is lo > 0 or hi < 0.

Run: python scripts/fb_g3_census_projection.py   Out: results/flipbudget/g3_census_projection.json
"""
import gzip
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import certify as C  # noqa: E402
from fb_certify import ALPHA_A, AUDIT, CENSUS, EPS0, Inputs  # noqa: E402

V3 = ROOT / "results" / "analysis" / "audit_v3_provenance.json"
OUT = ROOT / "results" / "flipbudget" / "g3_census_projection.json"
N_DRAW = 200
SEED = 20260925
THRESHOLD_SHARE = 0.10  # MASTER_PLAN 4c


def certified(inp, conc, disc_by_model, eps):
    n = 0
    for j, k in combinations(inp.models, 2):
        lo, hi = C.pair_range(inp.shares[j], inp.shares[k], conc, disc_by_model[j], disc_by_model[k], eps)
        n += (lo > 0) or (hi < 0)
    return n


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    audit = json.load(open(AUDIT, encoding="utf-8"))
    key = {k["uid"]: k for k in json.load(open(ROOT / "results" / "analysis" / "mathhard_labelling_key.json",
                                                encoding="utf-8"))}
    for it in audit["items"]:
        it["subject"], it["quartile"] = key[it["uid"]]["subject"], key[it["uid"]]["quartile"]
    inp = Inputs(census, audit, 3, "sympy", "sympy", "as_incorrect", census["models"], "kl")
    v3 = json.load(open(V3, encoding="utf-8"))["rows"]
    alloc = Counter((r["model"], r["cell"]) for r in v3)
    disc_alloc = {mc: n for mc, n in alloc.items() if mc[1] in inp.disc}
    conc_new = Counter()
    for (m, c), n in alloc.items():
        if c in inp.conc_cells:
            conc_new[c] += n
    domains = [(m, c) for m in inp.models for c in inp.disc if inp.shares[m].get(c, 0) > 0]
    level_all = 1 - ALPHA_A

    current = certified(inp, {c: C.kl_interval(*inp.pooled[c], C.sidak(2, level_all)) for c in inp.conc_cells},
                        {m: dict(inp.disc) for m in inp.models}, EPS0)
    rng = np.random.default_rng(SEED)
    out = {"current_existing_audit_only": current, "threshold_orderings": math.ceil(THRESHOLD_SHARE * current),
           "n_discordant_domains": len(domains), "n_registered_discordant_labels": sum(disc_alloc.values()),
           "scenarios": {}}
    for scen in ("uniform", "half"):
        reg, cen, gain = [], [], []
        for _ in range(N_DRAW):
            truth = {d: (rng.uniform() if scen == "uniform" else 0.5) for d in domains}
            x_pool = {c: inp.pooled[c][0] + rng.binomial(conc_new[c], inp.pooled[c][0] / inp.pooled[c][1])
                      for c in inp.conc_cells}
            n_pool = {c: inp.pooled[c][1] + conc_new[c] for c in inp.conc_cells}
            # registered design: every audited (model, discordant cell) is one more Sidak parameter
            lvl = C.sidak(len(inp.conc_cells) + len(disc_alloc), level_all)
            conc = {c: C.kl_interval(x_pool[c], n_pool[c], lvl) for c in inp.conc_cells}
            disc = {m: dict(inp.disc) for m in inp.models}
            for (m, c), n in disc_alloc.items():
                disc[m][c] = C.kl_interval(int(rng.binomial(n, truth[(m, c)])), n, lvl)
            r = certified(inp, conc, disc, EPS0)
            # census: discordant rates known exactly, so only the pooled intervals are estimated
            lvl2 = C.sidak(len(inp.conc_cells), level_all)
            conc2 = {c: C.kl_interval(x_pool[c], n_pool[c], lvl2) for c in inp.conc_cells}
            disc2 = {m: {c: (truth[(m, c)], truth[(m, c)]) if (m, c) in truth else (0.0, 1.0) for c in inp.disc}
                     for m in inp.models}
            s = certified(inp, conc2, disc2, EPS0)
            reg.append(r)
            cen.append(s)
            gain.append(s - r)
        q = lambda v: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),  # noqa: E731
                       "p90": float(np.percentile(v, 90))}
        out["scenarios"][scen] = {"registered": q(reg), "census": q(cen), "gain": q(gain)}
    gains = [out["scenarios"][s]["gain"]["median"] for s in out["scenarios"]]
    out["decision"] = ("label the remaining discordant responses" if min(gains) >= out["threshold_orderings"]
                       else "do not label; the projected gain is below the registered threshold")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "scenarios"}, indent=1))
    for s, v in out["scenarios"].items():
        print(s, v)


if __name__ == "__main__":
    main()
