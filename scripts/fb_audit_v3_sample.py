"""G3.2 -- draw the 458-row K=3 MATH-Hard audit sheet (TC_EXPANSION_PREREGISTRATION.md Amendment 4).

Design (fixed in results/flipbudget/g3_operating_point.json, committed before this runs):
  strata = (model, K=3 cell) over unlabeled rows only, cell = (em, sb_sympy, mv);
  falsification: n0_by_cell[c] rows in each concordant cell c (000, 111) for each listed model;
  certification: the rest on discordant strata, largest-remainder proportional to stratum
  size, capped at capacity. SRSWOR within each stratum (seed 20260922); row order shuffled.
Second-labeler overlap: ~20% of rows, stratified by (falsification vs certification), own seed.
Outputs:
  labeling/audit_v3_l1.csv, labeling/audit_v3_l2_overlap.csv  (uid, problem, response, your_answer)
  results/analysis/audit_v3_provenance.json  (strata, inclusion probabilities; NOT for labelers)
Run: python scripts/fb_audit_v3_sample.py
"""
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fb_census_k3 import load_rows  # noqa: E402
from fb_g3_frontier import allocate  # noqa: E402

OP = ROOT / "results" / "flipbudget" / "g3_operating_point.json"
SHEET = ROOT / "labeling" / "audit_v3_l1.csv"
OVERLAP = ROOT / "labeling" / "audit_v3_l2_overlap.csv"
PROV = ROOT / "results" / "analysis" / "audit_v3_provenance.json"
SEED = 20260922
OVERLAP_SHARE = 0.20


def uid(model, item):
    return hashlib.sha256(f"v3|{SEED}|{model}|{item}".encode()).hexdigest()[:16]


def main():
    op = json.load(open(OP, encoding="utf-8"))
    census = json.load(gzip.open(ROOT / "results" / "flipbudget" / "census_k3.json.gz", "rt", encoding="utf-8"))
    frame = defaultdict(list)  # (model, cell) -> unlabeled item ids
    for r in census["rows"]:
        if r["scored"] and not r["already_labeled"]:
            frame[(r["model"], f"{r['em']}{r['sb_sympy']}{r['mv']}")].append(r["item_id"])
    for v in frame.values():
        v.sort()
    n = {}
    for m in op["falsification_models"]:
        for c in ("000", "111"):
            n[(m, c)] = min(op["n0_by_cell"][c], len(frame[(m, c)]))
    disc_cap = {k: len(v) for k, v in frame.items() if k[1] not in ("000", "111")}
    n.update({k: v for k, v in allocate(disc_cap, op["budget"] - sum(n.values())).items() if v})
    rng = random.Random(SEED)
    drawn = []
    for (m, c) in sorted(n):
        for item in rng.sample(frame[(m, c)], n[(m, c)]):
            drawn.append({"model": m, "item_id": item, "cell": c,
                          "role": "falsification" if c in ("000", "111") else "certification",
                          "stratum_size": len(frame[(m, c)]), "n_stratum": n[(m, c)],
                          "inclusion_probability": n[(m, c)] / len(frame[(m, c)])})
    rng.shuffle(drawn)
    for d in drawn:
        d["uid"] = uid(d["model"], d["item_id"])
    # overlap: stratified by role, own seed
    orng = random.Random(SEED + 1)
    overlap = set()
    for role in ("falsification", "certification"):
        idx = [i for i, d in enumerate(drawn) if d["role"] == role]
        overlap.update(orng.sample(idx, round(OVERLAP_SHARE * len(idx))))
    # texts
    need = {(d["model"], d["item_id"]) for d in drawn}
    text = {}
    for r in load_rows(sorted({m for m, _ in need}))[0]:
        if (r["model"], r["item_id"]) in need:
            text[(r["model"], r["item_id"])] = (r["problem"], r["response"])
    for path, rows in ((SHEET, drawn), (OVERLAP, [drawn[i] for i in sorted(overlap)])):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uid", "problem", "response", "your_answer"])
            for d in rows:
                p, resp = text[(d["model"], d["item_id"])]
                w.writerow([d["uid"], p, resp, ""])
    for i, d in enumerate(drawn):
        d["in_overlap"] = i in overlap
    json.dump({"operating_point": op, "seed": SEED, "n_rows": len(drawn), "n_overlap": len(overlap),
               "allocation": {f"{m}|{c}": v for (m, c), v in sorted(n.items())},
               "by_role": dict(Counter(d["role"] for d in drawn)), "rows": drawn},
              open(PROV, "w", encoding="utf-8"), indent=1)
    print(f"{len(drawn)} rows ({dict(Counter(d['role'] for d in drawn))}), overlap {len(overlap)}")


if __name__ == "__main__":
    main()
