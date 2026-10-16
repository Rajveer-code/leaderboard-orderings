"""Draw the timed re-labeling check sheet for the 458-row tc_expansion2 audit.

Purpose (VALIDATION_RELABEL.md): the 2026-09-14/17 submissions of this sheet were excluded on a
provenance judgement, not on a measurement. The author states the labels are their own. This
draws a random, cell-stratified subsample for a fresh pass in the instrumented app
(labeling/app/label.html: paste blocked, per-item timing, export hash), so that agreement with
the earlier labels can be measured instead of argued.

The check sheet deliberately does NOT contain the earlier answers.

Run: python scripts/fb_relabel_sample.py    Out: labeling/relabel_check_l1.csv,
     results/analysis/relabel_check_provenance.json
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "labeling" / "tc_expansion2_l1.csv"
PROV_IN = ROOT / "results" / "analysis" / "tc_expansion2_provenance.json"
OUT = ROOT / "labeling" / "relabel_check_l1.csv"
PROV_OUT = ROOT / "results" / "analysis" / "relabel_check_provenance.json"
N_CHECK = 60
SEED = 20260924


def main():
    with open(SHEET, newline="", encoding="utf-8") as f:
        rows = {r["uid"]: r for r in csv.DictReader(f)}
    prov = json.load(open(PROV_IN, encoding="utf-8"))
    recs = prov["rows"] if isinstance(prov, dict) and "rows" in prov else prov
    cell = {r["uid"]: r.get("cell") or r.get("stratum") for r in recs if r.get("uid") in rows}
    by_cell = defaultdict(list)
    for uid in sorted(rows):
        by_cell[cell.get(uid, "unknown")].append(uid)
    rng = random.Random(SEED)
    # proportional allocation over cells, at least 5 per non-empty cell
    total = sum(len(v) for v in by_cell.values())
    take = {c: max(5, round(N_CHECK * len(v) / total)) for c, v in by_cell.items()}
    while sum(take.values()) > N_CHECK:
        take[max(take, key=lambda c: take[c])] -= 1
    while sum(take.values()) < N_CHECK:
        take[max(take, key=lambda c: len(by_cell[c]) - take[c])] += 1
    picked = [u for c, v in sorted(by_cell.items()) for u in rng.sample(v, min(take[c], len(v)))]
    rng.shuffle(picked)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "problem", "response", "your_answer"])
        for uid in picked:
            w.writerow([uid, rows[uid]["problem"], rows[uid]["response"], ""])
    json.dump({"seed": SEED, "n": len(picked), "source_sheet": SHEET.name,
               "by_cell": dict(Counter(cell.get(u, "unknown") for u in picked)),
               "uids": picked, "note": "earlier answers deliberately excluded from the check sheet"},
              open(PROV_OUT, "w", encoding="utf-8"), indent=1)
    print(f"{len(picked)} rows -> {OUT}")
    print("cells:", dict(Counter(cell.get(u, "unknown") for u in picked)))


if __name__ == "__main__":
    main()
