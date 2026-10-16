"""Do the two gold answers the verifiers use agree with each other, item by item?

The 2024 and boxed verifiers compare against the dataset's answer field; Math-Verify parses the
answer from the reference solution's last \\boxed{}. For each of the 1,324 MATH-Hard items this
script compares the two, with the benchmark's own routine (normalise both, then string or
symbolic equality, as in verifiers.human_correct(..., "sympy")). Agreement says the two fields
are consistent; it does not say either is mathematically right, and no human judges anything here.
Disagreements are listed in full for a reader to inspect.

Run: python scripts/fb_gold_consistency.py      Out: results/analysis/gold_consistency.json
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import verifiers as V  # noqa: E402
from fb_census_k3 import load_rows  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
OUT = ROOT / "results" / "analysis" / "gold_consistency.json"


def compare(answer: str, solution: str) -> dict:
    m = V.utils("head")
    box = m.last_boxed_only_string(solution or "")
    boxed = m.remove_boxed(box) if box else None
    if boxed is None:
        return {"solution_boxed": None, "agree": False, "reason": "no boxed answer in the solution"}
    return {"solution_boxed": boxed, "agree": bool(V.human_correct(boxed, answer, "sympy")), "reason": None}


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    items = {}
    for t in load_rows(census["models"])[0]:
        items.setdefault(t["item_id"], {"subject": t["subject"], "answer": t["gold"], "solution": t["solution"]})
    per_item = {}
    for k, it in sorted(items.items()):
        per_item[k] = {"subject": it["subject"], "answer": it["answer"], **compare(it["answer"], it["solution"])}
    dis = {k: v for k, v in per_item.items() if not v["agree"]}
    out = {"n_items": len(per_item), "n_agree": len(per_item) - len(dis), "n_disagree": len(dis),
           "n_no_boxed_solution": sum(v["reason"] is not None for v in dis.values()),
           "routine": "verifiers.human_correct(solution_boxed, answer, 'sympy')", "disagreements": dis}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print({k: v for k, v in out.items() if k != "disagreements"})
    for k, v in dis.items():
        print(k[:12], v["subject"], repr(v["answer"]), "|", repr(v["solution_boxed"]), v["reason"] or "")


if __name__ == "__main__":
    main()
