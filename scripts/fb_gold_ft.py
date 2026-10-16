"""The "ft" substring rule in the harness normaliser, and whether it touches any reader verdict.

normalize_final_answer removes every expression in REMOVED_EXPRESSIONS by plain substring
replacement, and the list contains "ft" (feet). Inside LaTeX that deletes letters from commands:
\\infty becomes \\iny and \\left becomes \\le. The harness applies the normaliser to the gold answer
when it builds the dataset, and the reader verdict (verifiers.human_correct, harness routine)
normalises both the transcription and the gold again. This script

  1. finds every item whose gold answer the rule changes, by normalising the solution's boxed
     answer with and without "ft" in the list (leaderboard discussion #1171 reports 27 such items);
  2. recomputes the reader verdict of every audited label on those items with a normaliser
     without "ft", from the solution's boxed answer, and compares it with the verdict the paper uses.

Run: python scripts/fb_gold_ft.py      Out: results/analysis/gold_ft.json
"""
import csv
import gzip
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import verifiers as V  # noqa: E402
from fb_census_k3 import load_rows  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
KEY = ROOT / "results" / "analysis" / "mathhard_labelling_key.json"
ANSWERS = ROOT / "results" / "analysis" / "mathhard_l1.csv"
OUT = ROOT / "results" / "analysis" / "gold_ft.json"
FIELDS = ("em", "sb_sympy", "mv")


@contextmanager
def without_ft():
    m = V.utils("head")
    saved = list(m.REMOVED_EXPRESSIONS)
    m.REMOVED_EXPRESSIONS[:] = [e for e in saved if e != "ft"]
    try:
        yield m
    finally:
        m.REMOVED_EXPRESSIONS[:] = saved


def ystar_without_ft(answer: str, boxed: str) -> int:
    with without_ft() as m:
        a, g = m.normalize_final_answer(answer), m.normalize_final_answer(boxed)
        if a.strip() == g.strip():
            return 1
        return int(V._timed_is_equiv("head", a, g)[0])


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    cells = {(r["model"], r["item_id"]): "".join(str(int(r[f])) for f in FIELDS) for r in census["rows"]}
    m = V.utils("head")
    items = {}
    for t in load_rows(census["models"][:1])[0]:
        box = m.last_boxed_only_string(t["solution"])
        boxed = m.remove_boxed(box)
        with_rule = m.normalize_final_answer(boxed)
        with without_ft() as mf:
            no_rule = mf.normalize_final_answer(boxed)
        items[t["item_id"]] = {"boxed": boxed, "stored": t["gold"], "with_rule": with_rule, "without_rule": no_rule}
    affected = {k: v for k, v in items.items() if v["with_rule"] != v["without_rule"]}
    assert all(v["stored"] == v["with_rule"] for v in affected.values()), "stored answer is not the normalised boxed answer"

    key = json.load(open(KEY, encoding="utf-8"))
    with open(ANSWERS, newline="", encoding="utf-8") as f:
        answers = {r["uid"]: r["your_answer"] for r in csv.DictReader(f)}
    labels = []
    for k in key:
        if k["item_id"] not in affected:
            continue
        a = answers.get(k["uid"])
        indet = V.is_indeterminate(a)
        used = 0 if indet else V.human_correct(a, k["gold"], "sympy")
        fixed = 0 if indet else ystar_without_ft(a, affected[k["item_id"]]["boxed"])
        labels.append({"uid": k["uid"], "model": k["model"], "item_id": k["item_id"],
                       "cell": cells.get((k["model"], k["item_id"])), "transcription": a,
                       "ystar_used": used, "ystar_without_ft": fixed})
    concordant = [x for x in labels if x["cell"] in ("000", "111")]
    out = {"n_items": len(items), "n_affected": len(affected),
           "affected": {k: {kk: v[kk] for kk in ("boxed", "stored")} for k, v in affected.items()},
           "n_audited_labels_on_affected": len(labels), "n_in_concordant_cells": len(concordant),
           "n_verdicts_changed": sum(x["ystar_used"] != x["ystar_without_ft"] for x in labels),
           "n_concordant_verdicts_changed": sum(x["ystar_used"] != x["ystar_without_ft"] for x in concordant),
           "labels": labels}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print({k: v for k, v in out.items() if k not in ("affected", "labels")})
    for x in labels:
        print(x["cell"], repr(x["transcription"]), x["ystar_used"], x["ystar_without_ft"])


if __name__ == "__main__":
    main()
