"""Build the T-C human-labeling package: a BLINDED CSV ready to label, plus a
separate key file for scoring afterward. Uses the real candidate items from
fb_tc_ee_real_worklist.py's output (results/flipbudget/tc_ee_real_worklist.json).

BLINDING, reused exactly from knowledgeshift's scripts/63_make_mathhard_sheets.py --
not reimplemented, because this project already had one real regression here (an
earlier version put raw "stratum|model|item_id" strings into sheet HTML; never
rendered on screen, but sat in the raw file for anyone who viewed source, defeating
blinding just as surely as if shown). Same construction here:
    raw_id = f"{stratum}|{model}|{item_id}"
    uid = sha256(raw_id)[:16]
The CSV has ONLY uid, problem, response, your_answer -- no model name, no item_id, no
stratum, matching HUMAN_LABELING_GUIDE.md's sheet contract exactly. The key (uid ->
model/item_id/stratum/gold/subject) is written to a SEPARATE file. Per the guide: the
labeller must not see the key before labeling; only consult it afterward to score.

Run: python scripts/fb_tc_labeling_package.py
"""
import csv
import hashlib
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, "scripts")
from fb_pull_roster import load_l1, subject_of, decode_model_name, KNOWLEDGESHIFT_ROOT, HARNESS_ROOT

TOP_N_PAIRS = 40  # matches the number already reported as achievable (30 distinct items)
RESPONSE_TRUNCATE = 6000  # matches scripts/63's own convention exactly


def load_candidates():
    with open("results/flipbudget/tc_ee_real_worklist.json", encoding="utf-8") as f:
        d = json.load(f)
    return d["tc_top_candidates"][:TOP_N_PAIRS]


def needed_records(candidates):
    """Every distinct (model, item_id, stratum) the candidate pairs reference."""
    needed = set()
    for c in candidates:
        needed.add((c["model_a"], c["item_id"], c["stratum_a"]))
        needed.add((c["model_b"], c["item_id"], c["stratum_b"]))
    return needed


def pull_records(l1, math, needed):
    """One pass over cache_l1, keeping only the (model,item_id) pairs actually needed
    -- reuses the exact same score_boxed/latest_math_files/rows_of path as
    fb_pull_roster.py, so the gold/problem/response fields are pulled from the same
    place that already cross-validated at 0/400 mismatches against the live audit."""
    needed_models = {m for m, _, _ in needed}
    needed_items_by_model = {}
    for m, item_id, stratum in needed:
        needed_items_by_model.setdefault(m, set()).add(item_id)

    found = {}
    cache_l1 = KNOWLEDGESHIFT_ROOT / "results" / "cache_l1"
    dirs = sorted(p for p in cache_l1.iterdir() if p.is_dir())
    for d in dirs:
        model = decode_model_name(d.name)
        if model not in needed_models:
            continue
        want_items = needed_items_by_model[model]
        files = l1.latest_math_files([p.name for p in d.glob("*.json")])
        for fname in files:
            subject = subject_of(fname)
            if subject is None:
                continue
            text_content = (d / fname).read_text(encoding="utf-8", errors="replace")
            for row in l1.rows_of(text_content):
                doc = row.get("doc") or {}
                doc_hash = row.get("doc_hash")
                if doc_hash not in want_items:
                    continue
                gold = doc.get("answer")
                resps = row.get("resps") or []
                if not (gold and resps and doc_hash):
                    continue
                text = resps[0][0] if isinstance(resps[0], list) else resps[0]
                if not isinstance(text, str):
                    continue
                found[(model, doc_hash)] = {
                    "problem": doc.get("problem", ""), "response": text,
                    "gold": gold, "subject": subject,
                }
    return found


if __name__ == "__main__":
    print("=" * 70)
    print("T-C human-labeling package: blinded CSV + separate scoring key")
    print("=" * 70)

    candidates = load_candidates()
    needed = needed_records(candidates)
    distinct_items = {item_id for _, item_id, _ in needed}
    print(f"Top {len(candidates)} candidate pairs -> {len(needed)} distinct (model,item,stratum) "
          f"records needed, {len(distinct_items)} distinct items")

    if not HARNESS_ROOT.is_dir():
        raise SystemExit(f"harness not found at {HARNESS_ROOT}")
    l1 = load_l1()
    math = l1.load_math_utils(HARNESS_ROOT)
    records = pull_records(l1, math, needed)
    print(f"Pulled {len(records)} of {len(needed)} needed records from cache_l1")
    missing = {(m, i, s) for (m, i, s) in needed if (m, i) not in records}
    if missing:
        print(f"  MISSING (not found in cache_l1, excluded from package): {len(missing)}")
        for m in sorted(missing)[:5]:
            print(f"    {m}")

    rows_csv = []
    rows_key = []
    seen_uids = set()
    for model, item_id, stratum in sorted(needed):
        rec = records.get((model, item_id))
        if rec is None:
            continue
        raw_id = f"{stratum}|{model}|{item_id}"
        uid = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        if uid in seen_uids:
            continue  # defensive: a hash collision would silently merge two rows
        seen_uids.add(uid)
        rows_csv.append({"uid": uid, "problem": rec["problem"],
                          "response": rec["response"][:RESPONSE_TRUNCATE], "your_answer": ""})
        rows_key.append({"uid": uid, "model": model, "item_id": item_id, "stratum": stratum,
                          "subject": rec["subject"], "gold": rec["gold"]})

    print(f"\nFinal package: {len(rows_csv)} blinded rows ({len(rows_csv)} of "
          f"{2*len(candidates)} theoretical max, some pairs share a model/item across "
          f"multiple candidate pairs so the true row count is less than 2x pairs)")

    out_dir = Path("labeling")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tc_expansion_l1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "problem", "response", "your_answer"])
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nBLINDED CSV (open this to label): {csv_path} ({len(rows_csv)} rows)")

    key_path = Path("results/flipbudget/tc_expansion_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({"n_rows": len(rows_key), "key": rows_key,
                   "pair_candidates": candidates[:len(candidates)]}, f, indent=2)
    print(f"SCORING KEY (do NOT open while labeling): {key_path}")

    # Sanity: every uid in the CSV has exactly one key entry and vice versa
    csv_uids = {r["uid"] for r in rows_csv}
    key_uids = {r["uid"] for r in rows_key}
    ok = csv_uids == key_uids and len(csv_uids) == len(rows_csv)
    print(f"\n[{'OK' if ok else 'FAIL'}] CSV and key uid sets match exactly, no duplicates: {ok}")
