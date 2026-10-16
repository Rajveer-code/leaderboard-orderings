"""G1 -- impact of the Windows is_equiv bug on the K=2 anchor (17 models, existing audit).

Same code as the anchor (fb_joint_verifier.evaluate), two input semantics:
  string: sb_string cells + string human reference   (historical: 84 / 101 / 118)
  sympy:  sb_sympy cells + sympy human reference      (faithful harness equivalence)
Run: python scripts/fb_g1_semantics_impact.py   Out: results/flipbudget/g1_semantics_impact.json
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fb_joint_verifier as jv  # noqa: E402

OUT = ROOT / "results" / "flipbudget" / "g1_semantics_impact.json"


def main():
    amap = json.load(open(ROOT / "results" / "flipbudget" / "audit_k3_map.json", encoding="utf-8"))
    rows = json.load(gzip.open(ROOT / "results" / "flipbudget" / "census_k3.json.gz", "rt", encoding="utf-8"))["rows"]
    models = amap["models17"]
    out = {}
    for sem in ("string", "sympy"):
        counts = {m: dict.fromkeys(jv.CELLS, 0) for m in models}
        for r in rows:
            if r["model"] in counts:
                counts[r["model"]][f"{r[f'sb_{sem}']}{r['em']}"] += 1
        shares = {m: {c: v[c] / sum(v.values()) for c in jv.CELLS} | {"N": sum(v.values())} for m, v in counts.items()}
        out[sem] = {"census_cells_17": {c: sum(v[c] for v in counts.values()) for c in jv.CELLS},
                    "sampling_ceiling": jv.sampling_ceiling(shares, models)}
        for indet in ("as_incorrect", "drop"):
            cells = amap["audit_cells"][f"17|k2_{sem}|{sem}|{indet}"]
            b11 = jv.cp(cells["11"]["x_correct"], cells["11"]["n"], jv.lvl(2))
            b00 = jv.cp(cells["00"]["x_correct"], cells["00"]["n"], jv.lvl(2))
            r = jv.evaluate(shares, models, ("shared", b11, b00), lambda m, c: (0.0, 1.0))
            out[sem][indet] = {"audit_cells": cells, "audit_resolved": r["audit_resolved"],
                               "combined_resolved": r["combined_resolved"]}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    for sem, v in out.items():
        print(sem, "cells", v["census_cells_17"], "ceiling", v["sampling_ceiling"],
              {k: (v[k]["audit_resolved"], v[k]["combined_resolved"]) for k in ("as_incorrect", "drop")})


if __name__ == "__main__":
    main()
