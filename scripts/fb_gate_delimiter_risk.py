"""Before Jobs B and C: how a typing habit the v3 rulebook does not address moves the gate and Y*.

The app shows responses as raw text, so an answer given only in the Minerva sentence appears as
`$\\frac{5}{12}$`. The v3 rulebook says to type what is inside \\boxed{} without the wrapper and
not to convert forms; it says nothing about the $...$ delimiters. The gate's extraction check
compares typed answers with the regex extraction verbatim, against a baseline from the September
audit, whose rules transcribed "$1/2$" as "1/2".

1. Gate: for each registered batch, the probability that a genuine batch fails the extraction
   check when the labeller keeps the delimiters of a phrase-only answer with probability q.
   Boxed and delimiter-free phrase-only rows match at the gate's own baseline rates.
2. Y*: for every phrase-only answer written in $...$, whether the reference routine (harness
   sympy, primary; Math-Verify, sensitivity) gives the same verdict with and without them.

No label is involved. Run: python scripts/fb_gate_delimiter_risk.py
Out: results/analysis/gate_delimiter_risk.json
"""
import csv
import gzip
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fb_audit_v3_gate import BASELINE, P_AUTOMATION, extraction_info, poisson_binomial_sf  # noqa: E402

SEED, REPS, QS, BATCH = 20260926, 400, (0.0, 0.25, 0.5, 1.0), 50
SHEETS = {"audit_v3_l1.csv": "audit_v3_provenance.json",
          "discordant_census_l1.csv": "discordant_census_provenance.json"}
OUT = ROOT / "results" / "analysis" / "gate_delimiter_risk.json"
DELIMS = re.compile(r"\$\$([^$]*)\$\$|\$([^$]*)\$|\\\((.*)\\\)|\\\[(.*)\\\]", re.S)


def strip_delimiters(s: str) -> str:
    """The answer with its math delimiters removed; text outside them is kept."""
    s = s.strip()
    m = DELIMS.fullmatch(s)
    if m:
        return next(g for g in m.groups() if g is not None).strip()
    return re.sub(r"\$([^$]*)\$", r"\1", s)


def gate_failures(info: list, q: float, rng: random.Random) -> list:
    """Expected share of failing replicates per registered batch of 50."""
    pb = BASELINE["boxed"][0] / BASELINE["boxed"][1]
    pp = BASELINE["phrase_only"][0] / BASELINE["phrase_only"][1]
    out = []
    for b in range(0, len(info), BATCH):
        batch = [i for i in info[b:b + BATCH] if i[1]]
        ps = [BASELINE[k][0] / BASELINE[k][1] for _, k in batch]
        fails = 0
        for _ in range(REPS):
            k = 0
            for ex, kind in batch:
                if kind == "boxed":
                    k += rng.random() < pb
                elif "$" in sorted(ex)[0]:
                    k += rng.random() < q
                else:
                    k += rng.random() < pp
            fails += poisson_binomial_sf(k, ps) < P_AUTOMATION
        out.append(fails / REPS)
    return out


def main():
    from fb_census_k3 import load_rows
    from flipbudget.verifiers import human_correct
    census = json.load(gzip.open(ROOT / "results" / "flipbudget" / "census_k3.json.gz", "rt", encoding="utf-8"))
    ref = load_rows(census["models"][:1])[0]
    gold = {r["item_id"]: r["gold"] for r in ref}
    solution = {r["item_id"]: r["solution"] for r in ref}
    rng = random.Random(SEED)
    res = {"seed": SEED, "reps": REPS, "baseline": BASELINE, "p_automation": P_AUTOMATION, "sheets": {}}
    for name, prov_name in SHEETS.items():
        with open(ROOT / "labeling" / name, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        prov = {r["uid"]: r for r in json.load(open(ROOT / "results" / "analysis" / prov_name,
                                                    encoding="utf-8"))["rows"]}
        info = [extraction_info(r["response"]) for r in rows]
        phrase = [(r, sorted(ex)[0]) for r, (ex, kind) in zip(rows, info) if kind == "phrase_only"]
        dollar = [(r, e) for r, e in phrase if "$" in e]
        differ = {"sympy": [], "mv": []}
        for r, e in dollar:
            item = prov[r["uid"]]["item_id"]
            for routine, g in (("sympy", gold[item]), ("mv", solution[item])):
                a, b = human_correct(e, g, routine), human_correct(strip_delimiters(e), g, routine)
                if a != b:
                    differ[routine].append({"uid": r["uid"], "extraction": e, "stripped": strip_delimiters(e),
                                            "with": int(a), "without": int(b)})
        fails = {str(q): gate_failures(info, q, rng) for q in QS}
        res["sheets"][name] = {
            "rows": len(rows), "batches": len(fails["0.0"]),
            "boxed": sum(k == "boxed" for _, k in info), "phrase_only": len(phrase),
            "phrase_only_with_dollar": len(dollar), "no_extraction": sum(k is None for _, k in info),
            "expected_failing_batches": {q: round(sum(v), 2) for q, v in fails.items()},
            "per_batch_failure_probability": fails,
            "ystar_differs_with_vs_without_delimiters": {k: len(v) for k, v in differ.items()},
            "ystar_differences": differ}
        s = res["sheets"][name]
        print(f"{name}: phrase-only {s['phrase_only']} ({s['phrase_only_with_dollar']} in $...$); expected failing "
              f"batches by q {s['expected_failing_batches']} of {s['batches']}; Y* differs "
              f"{s['ystar_differs_with_vs_without_delimiters']}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
