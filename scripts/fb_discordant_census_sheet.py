"""Draw the discordant-census labelling sheet (MASTER_PLAN T18, TC_EXPANSION_PREREGISTRATION
Amendment 6). Written and run before any census label exists.

Every response on which the three verifiers disagree gets a reader-derived label, except those
already labelled elsewhere: the registered 458-row audit (its 180 discordant rows are labelled
in that job) and the existing 400-item audit. With the census complete, every discordant
(model, cell) rate is known exactly rather than estimated.

The sheet carries only uid, problem and response, as every labelling sheet does, in an order
shuffled with a fixed seed so that neither model nor cell can be read from position. The
mapping from uid to (model, item, cell) is kept in a separate provenance file the labeller
does not open.

Run: python scripts/fb_discordant_census_sheet.py
Out: labeling/discordant_census_l1.csv, results/analysis/discordant_census_provenance.json
"""
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fb_census_k3 import load_rows  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
V3 = ROOT / "results" / "analysis" / "audit_v3_provenance.json"
EXISTING = ROOT / "results" / "flipbudget" / "audit_k3_map.json"
SHEET = ROOT / "labeling" / "discordant_census_l1.csv"
PROV = ROOT / "results" / "analysis" / "discordant_census_provenance.json"
OTHER_SHEETS = ["audit_v3_l1.csv", "audit_v3_l2_overlap.csv", "relabel_check_l1.csv", "tc_expansion2_l1.csv"]
SEED = 20260925
FIELDS = ("em", "sb_sympy", "mv")  # cell order: leaderboard 2024, boxed, Math-Verify


def main():
    if SHEET.exists():
        raise SystemExit(f"{SHEET} already exists; the sheet is drawn once and never re-drawn")
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    cell = {(r["model"], r["item_id"]): "".join(str(r[f]) for f in FIELDS) for r in census["rows"]}
    discordant = {mi: c for mi, c in cell.items() if c not in ("000", "111")}
    in_v3 = {(r["model"], r["item_id"]) for r in json.load(open(V3, encoding="utf-8"))["rows"]}
    in_existing = {(it["model"], it["item_id"]) for it in json.load(open(EXISTING, encoding="utf-8"))["items"]}
    todo = sorted(mi for mi in discordant if mi not in in_v3 and mi not in in_existing)

    rows, _ = load_rows(census["models"])
    text = {(r["model"], r["item_id"]): r for r in rows}
    missing = [mi for mi in todo if mi not in text or not isinstance(text[mi]["response"], str)]
    if missing:
        raise SystemExit(f"{len(missing)} responses not found in the generation cache; nothing written")

    uid = {mi: hashlib.sha256(f"discordant-census|{mi[0]}|{mi[1]}".encode()).hexdigest()[:16] for mi in todo}
    taken = set()
    for name in OTHER_SHEETS:
        p = ROOT / "labeling" / name
        if p.exists():
            taken |= {r["uid"] for r in csv.DictReader(open(p, encoding="utf-8"))}
    assert len(set(uid.values())) == len(uid) and not (set(uid.values()) & taken), "uid collision"

    order = list(todo)
    random.Random(SEED).shuffle(order)
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "problem", "response", "your_answer"])
        for mi in order:
            w.writerow([uid[mi], text[mi]["problem"], text[mi]["response"], ""])

    prov = {
        "amendment": "TC_EXPANSION_PREREGISTRATION.md, Amendment 6",
        "seed": SEED,
        "n_discordant_total": len(discordant),
        "n_labelled_in_registered_audit": len([mi for mi in discordant if mi in in_v3]),
        "n_labelled_in_existing_audit": len([mi for mi in discordant if mi in in_existing]),
        "n_census_rows": len(order),
        "by_cell": dict(sorted(Counter(discordant[mi] for mi in order).items())),
        "sheet_sha256": hashlib.sha256(SHEET.read_bytes()).hexdigest(),
        # git stores text with LF line endings, so a clone reproduces this one, not the above
        "sheet_sha256_lf": hashlib.sha256(SHEET.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "rows": [{"uid": uid[mi], "model": mi[0], "item_id": mi[1], "cell": discordant[mi]} for mi in order],
    }
    json.dump(prov, open(PROV, "w", encoding="utf-8"), indent=1)
    print(f"wrote {SHEET} ({len(order)} rows) and {PROV}")
    print({k: v for k, v in prov.items() if k != "rows"})


if __name__ == "__main__":
    main()
