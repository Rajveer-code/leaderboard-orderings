"""Draw the preregistered ~458-row stratified sample for the T-C expansion,
from the population built by fb_tc_expansion_population.py. Design (see
TC_EXPANSION_PREREGISTRATION.md for the full justification, written BEFORE
this script was run):

  - EXCLUDE all (model, item_id) already in the existing 400-item T-C
    population (avoid overlap with existing labels, per instruction).
  - Cell 1 "both_credited"      (score_boxed=credited, exact_match=credited): 200
  - Cell 2 "both_not_credited"  (score_boxed!=credited, exact_match!=credited): 198
  - Cell 3 "disagree_sb_credited_em_not" : 30
  - Cell 4 "disagree_em_credited_sb_not" : 30
  Total: 458.

Fixed seed (42, this project's standing convention) for reproducibility.
Then re-fetches FULL response + problem text (not just the 300-char tail
kept in the lightweight population file) for exactly the sampled items,
and writes the final blinded CSV in the EXACT existing T-C column format
(uid, problem, response, your_answer) -- fb_tc_labeling_package.py's own
format, unchanged, so the existing verified protocol and README instructions
apply without modification.

Run: python scripts/fb_tc_expansion_sample.py
"""
import csv
import json
import random
import sys
import urllib.request
import uuid

sys.path.insert(0, "scripts")
from huggingface_hub import get_token
import truststore
truststore.inject_into_ssl()

SEED = 42
CELL_TARGETS = {
    "both_credited": 200,
    "both_not_credited": 198,
    "disagree_sb_credited_em_not": 30,
    "disagree_em_credited_sb_not": 30,
}


def cell_of(row):
    sb_credited = row["score_boxed_stratum"] == "credited"
    em_credited = row["exact_match_stratum"] == "credited"
    if sb_credited and em_credited:
        return "both_credited"
    if not sb_credited and not em_credited:
        return "both_not_credited"
    if sb_credited and not em_credited:
        return "disagree_sb_credited_em_not"
    return "disagree_em_credited_sb_not"


def main():
    with open("results/analysis/tc_expansion_population.json", encoding="utf-8") as f:
        pop = json.load(f)["population"]

    eligible = [r for r in pop if not r["already_labeled"] and r["exact_match"] is not None]
    print(f"Eligible (not already labeled, exact_match present): {len(eligible)} of {len(pop)}")

    by_cell = {}
    for r in eligible:
        by_cell.setdefault(cell_of(r), []).append(r)
    for cell, target in CELL_TARGETS.items():
        print(f"  {cell}: population {len(by_cell.get(cell, []))}, target {target}")

    rng = random.Random(SEED)
    sample = []
    shortfalls = {}
    for cell, target in CELL_TARGETS.items():
        pool = by_cell.get(cell, [])
        if len(pool) < target:
            shortfalls[cell] = (len(pool), target)
            chosen = pool
        else:
            chosen = rng.sample(pool, target)
        for c in chosen:
            c["cell"] = cell
        sample.extend(chosen)

    print(f"\nTotal sampled: {len(sample)}")
    if shortfalls:
        print(f"SHORTFALLS (population smaller than target): {shortfalls}")

    # ---- Re-fetch full response + problem text for exactly the sampled items ----
    token = get_token()
    needed = {}  # (model, subject) -> set of item_ids
    for r in sample:
        needed.setdefault((r["model"], r["subject"]), set()).add(r["item_id"])

    def latest_file(model, subject):
        encoded = model.replace("/", "__")
        api_url = f"https://huggingface.co/api/datasets/open-llm-leaderboard/{encoded}-details"
        req = urllib.request.Request(api_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta = json.load(resp)
        names = [s["rfilename"] for s in meta["siblings"]]
        cands = sorted(n for n in names if f"samples_leaderboard_math_{subject}_hard" in n)
        return cands[-1] if cands else None

    def fetch(model, fname):
        encoded = model.replace("/", "__")
        file_url = f"https://huggingface.co/datasets/open-llm-leaderboard/{encoded}-details/resolve/main/{fname}"
        req = urllib.request.Request(file_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    full_by_key = {}
    for i, ((model, subject), item_ids) in enumerate(needed.items(), 1):
        fname = latest_file(model, subject)
        recs = fetch(model, fname)
        for r in recs:
            doc = r.get("doc") or {}
            doc_hash = r.get("doc_hash")
            if doc_hash in item_ids:
                resps = r.get("resps") or []
                t = resps[0][0] if isinstance(resps[0], list) else resps[0]
                full_by_key[(model, doc_hash)] = {
                    "problem": doc.get("problem") or doc.get("question") or "",
                    "response": t if isinstance(t, str) else "",
                }
        print(f"  [{i}/{len(needed)}] {model}/{subject}: refetched", flush=True)

    # ---- Build blinded CSV (existing format, no model/stratum/cell columns) ----
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(sample)  # blind the ORDER too -- do not group by cell/model
    rows_csv = []
    provenance = []
    for r in sample:
        key = (r["model"], r["item_id"])
        full = full_by_key.get(key)
        if not full or not full["response"]:
            continue
        uid = uuid.uuid4().hex[:16]
        rows_csv.append({"uid": uid, "problem": full["problem"], "response": full["response"], "your_answer": ""})
        provenance.append({"uid": uid, "model": r["model"], "item_id": r["item_id"], "subject": r["subject"],
                            "cell": r["cell"], "score_boxed_stratum": r["score_boxed_stratum"],
                            "exact_match_stratum": r["exact_match_stratum"], "gold": r["gold"]})

    print(f"\nFinal CSV rows (after full-text refetch, some may drop if response empty): {len(rows_csv)}")

    with open("labeling/tc_expansion2_l1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "problem", "response", "your_answer"])
        w.writeheader()
        w.writerows(rows_csv)
    print("Written to labeling/tc_expansion2_l1.csv (BLINDED -- send this one)")

    # Provenance kept SEPARATELY, never distributed to the labeler.
    with open("results/analysis/tc_expansion2_provenance.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "cell_targets": CELL_TARGETS, "shortfalls": shortfalls,
                    "n_final": len(rows_csv), "rows": provenance}, f, indent=2)
    print("Written to results/analysis/tc_expansion2_provenance.json (NOT for distribution)")


if __name__ == "__main__":
    main()
