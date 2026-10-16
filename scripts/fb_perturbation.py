"""T23 -- the same answer, a different score: meaning-preserving surface changes.

A seeded random sample of responses that all three verifiers credit (cell 111) is rewritten in
ways that leave the stated answer untouched, each of a kind that occurs in the real responses,
and re-scored with the census's own scoring function. A reader would credit every rewritten
response exactly as before, because the answer is still stated; any change of verdict is due to
surface form alone.

  signoff      "I hope it is correct." -> "Great job!" (the paper's opening example)
  no_template  the "Final Answer: ..." sentence removed; the boxed answer stays
  no_box       every \\boxed{X} replaced by X; the "Final Answer" sentence stays
  control      trailing whitespace appended; nothing should change

A rewrite is applied only where it leaves the answer stated at least once, so each has its own
denominator.

Run: python scripts/fb_perturbation.py   Out: results/flipbudget/perturbation.json
"""
import gzip
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fb_census_k3 import load_rows, score  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
OUT = ROOT / "results" / "flipbudget" / "perturbation.json"
N, SEED = 300, 20260926
TEMPLATE = re.compile(r"Final Answer: The final answer is(.*?)\. I hope it is correct\.", re.S)
FIELDS = ("em_rec", "sb_sympy", "mv")


def strip_boxes(s):
    """Replace every \\boxed{...} by its content, braces balanced."""
    out, i = [], 0
    while True:
        j = s.find("\\boxed{", i)
        if j < 0:
            return "".join(out) + s[i:]
        out.append(s[i:j])
        k, depth = j + len("\\boxed{"), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        out.append(s[j + len("\\boxed{"):k - 1])
        i = k


def rewrites(resp):
    out = {"control": resp + "\n   "}
    if TEMPLATE.search(resp):
        out["signoff"] = resp.replace("I hope it is correct.", "Great job!")
        if "\\boxed{" in resp:
            out["no_template"] = TEMPLATE.sub("", resp)
            out["no_box"] = strip_boxes(resp)
    return out


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    cell111 = sorted((r["model"], r["item_id"]) for r in census["rows"]
                     if (r["em"], r["sb_sympy"], r["mv"]) in ((1, 1, 1), ("1", "1", "1")))
    sample = random.Random(SEED).sample(cell111, N)
    text = {(r["model"], r["item_id"]): r for r in load_rows(census["models"])[0]}
    rows, meta = [], []
    for k in sample:
        base = text[k]
        for name, resp in rewrites(base["response"]).items():
            rows.append({"response": resp, "gold": base["gold"], "solution": base["solution"], "scored": True})
            meta.append((name, k))
    score(rows)
    res = {}
    for (name, _), r in zip(meta, rows):
        d = res.setdefault(name, {"n": 0, **{f: 0 for f in FIELDS}})
        d["n"] += 1
        for f in FIELDS:
            d[f] += int(r[f])  # every sampled response was credited by all three before rewriting
    out = {"n_sampled": N, "seed": SEED, "population": "cell 111 (all three verifiers credit)",
           "n_population": len(cell111), "credited_after_rewrite": res}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    for name, d in res.items():
        print(f"{name:12s} n={d['n']:3d} | still credited: " + ", ".join(f"{f} {d[f]}" for f in FIELDS))


if __name__ == "__main__":
    main()
