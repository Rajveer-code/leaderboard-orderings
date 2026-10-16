"""Build the eligible item population for the T-C large-audit expansion
(the ~458-row audit approved this phase). Extends fb_pull_roster.py's own
population (score_boxed stratum only) with the exact_match field too, for
the 17 qualifying models across all 7 MATH-Hard subjects, so the new
sample can be stratified on BOTH scorers plus their disagreement --
per the joint design (this session's recommendation, confirmed by the
user's follow-up instruction, which asked for a formal power calculation
"explaining exactly what this additional audit is intended to determine").

Excludes any (model, item_id) already present in the existing 400-item T-C
population (results/analysis/mathhard_labelling_key.json) so the new
sample does not overlap already-labeled items where avoidable, per the
user's explicit instruction.

Run: python scripts/fb_tc_expansion_population.py
"""
import importlib.util
import json
import sys
import types
import urllib.request
from pathlib import Path

sys.path.insert(0, "scripts")
from huggingface_hub import get_token
import truststore
truststore.inject_into_ssl()
from ea_dominance_study import load, build_model_table

if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    sys.modules["datasets"] = _stub
_spec = importlib.util.spec_from_file_location("mathutils", "vendor/leaderboard_math/utils.py")
mathutils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mathutils)

SUBJECTS = ["algebra", "counting_and_prob", "geometry", "intermediate_algebra",
            "num_theory", "prealgebra", "precalculus"]


def score_boxed(math, response, gold):
    boxed = math.last_boxed_only_string(response)
    if not boxed or boxed == math.INVALID_ANSWER:
        return 0, False
    try:
        answer = math.normalize_final_answer(math.remove_boxed(boxed))
    except (AssertionError, IndexError, ValueError):
        return 0, False
    normalized_gold = math.normalize_final_answer(gold)
    return int(answer.strip() == normalized_gold.strip() or math.is_equiv(answer, normalized_gold)), True


def latest_file(token, model, subject):
    encoded = model.replace("/", "__")
    api_url = f"https://huggingface.co/api/datasets/open-llm-leaderboard/{encoded}-details"
    req = urllib.request.Request(api_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        meta = json.load(resp)
    names = [s["rfilename"] for s in meta["siblings"]]
    cands = sorted(n for n in names if f"samples_leaderboard_math_{subject}_hard" in n)
    return cands[-1] if cands else None


def fetch(token, model, fname):
    encoded = model.replace("/", "__")
    file_url = f"https://huggingface.co/datasets/open-llm-leaderboard/{encoded}-details/resolve/main/{fname}"
    req = urllib.request.Request(file_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    token = get_token()
    if not token:
        raise SystemExit("No local HF token found.")

    accs, margins = load()
    rows, excl = build_model_table(accs, margins)
    models = sorted(rows.keys())
    print(f"Building population for {len(models)} qualifying models x {len(SUBJECTS)} subjects")

    with open("results/analysis/mathhard_labelling_key.json") as f:
        existing_key = json.load(f)
    already_labeled = {(r["model"], r["item_id"]) for r in existing_key}
    print(f"Existing labeled items to exclude where possible: {len(already_labeled)}")

    population = []
    n_fetch_fail = 0
    for i, model in enumerate(models, 1):
        for subject in SUBJECTS:
            try:
                fname = latest_file(token, model, subject)
                if not fname:
                    continue
                recs = fetch(token, model, fname)
            except Exception as e:
                n_fetch_fail += 1
                print(f"  [{model}/{subject}] FETCH FAILED: {e}")
                continue
            for r in recs:
                doc = r.get("doc") or {}
                gold = doc.get("answer")
                resps = r.get("resps") or []
                doc_hash = r.get("doc_hash")
                if not (gold and resps and doc_hash):
                    continue
                t = resps[0][0] if isinstance(resps[0], list) else resps[0]
                if not isinstance(t, str):
                    continue
                sb_score, sb_found = score_boxed(mathutils, t, gold)
                sb_stratum = "unparsed" if not sb_found else ("credited" if sb_score == 1 else "wrong_parsed")
                em = r.get("exact_match")
                em_stratum = "credited" if em == 1 else "wrong"
                population.append({
                    "model": model, "item_id": doc_hash, "subject": subject,
                    "score_boxed_stratum": sb_stratum, "exact_match": int(em) if em is not None else None,
                    "exact_match_stratum": em_stratum,
                    "disagree": bool(sb_score != em) if em is not None else None,
                    "already_labeled": (model, doc_hash) in already_labeled,
                    "response_tail": t[-300:],
                    "gold": gold,
                })
        print(f"  [{i}/{len(models)}] {model}: {len(population)} rows so far", flush=True)

    n_already = sum(1 for p in population if p["already_labeled"])
    n_disagree = sum(1 for p in population if p["disagree"])
    print(f"\nTotal population rows: {len(population)}")
    print(f"Already-labeled (excludable): {n_already}")
    print(f"Disagreement rows (score_boxed != exact_match): {n_disagree} "
          f"({100*n_disagree/len(population):.2f}%)")
    print(f"Fetch failures: {n_fetch_fail}")

    out_path = Path("results/analysis/tc_expansion_population.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_rows": len(population), "n_already_labeled": n_already,
                    "n_disagree": n_disagree, "n_fetch_failures": n_fetch_fail,
                    "population": population}, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
