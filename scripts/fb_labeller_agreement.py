"""How reliable are the 400-item audit's labels, and does the headline survive a different labeller?

The first labeller (the author) labelled all 400 items. A second labeller labelled a pre-specified
overlap of 151 of them and a third a nested overlap of 75, fixed before any label was seen. This
script reports, with no new labelling:
  - agreement on the quantity that enters the certification, the reader verdict Y* (the labeller's
    transcription judged by the harness routine), overall and by verifier cell, with Cohen's kappa;
  - the string-level agreement recorded when the overlaps were scored (mathhard_human_margins.json);
  - the certified count at eps = 0 and at the registered tolerance when the first labeller's labels on
    an overlap are replaced by the other labeller's, and when any concordant-cell disagreement is
    resolved against the first labeller.

Run: python scripts/fb_labeller_agreement.py
Out: results/flipbudget/labeller_agreement.json
"""
import copy
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget.verifiers import human_correct, is_indeterminate  # noqa: E402
import fb_certify as F  # noqa: E402

RES = ROOT / "results" / "flipbudget"
ANA = ROOT / "results" / "analysis"
OUT = RES / "labeller_agreement.json"
LABELLERS = {"two": "mathhard_l2.csv", "three": "mathhard_l3.csv"}


def load():
    census = json.load(gzip.open(RES / "census_k3.json.gz", "rt", encoding="utf-8"))
    audit = json.load(open(RES / "audit_k3_map.json", encoding="utf-8"))
    key = {k["uid"]: k for k in json.load(open(ANA / "mathhard_labelling_key.json", encoding="utf-8"))}
    for it in audit["items"]:  # design strata of the audit, as scripts/fb_certify.py attaches them
        it["subject"], it["quartile"] = key[it["uid"]]["subject"], key[it["uid"]]["quartile"]
    return census, audit, key


def other_labels(name, by_uid, key):
    out = {}
    for r in csv.DictReader(open(ANA / name, encoding="utf-8")):
        uid, ans = r["uid"], r["your_answer"]
        if uid not in by_uid:
            continue
        indet = is_indeterminate(ans)
        out[uid] = {"indet": indet, "y": 0 if indet else int(human_correct(ans, key[uid]["gold"], "sympy"))}
    return out


def y_first(it):
    return 0 if it["indeterminate"] else int(it["ystar"]["sympy"])


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    p1, p2 = sum(a) / n, sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(lab, by_uid):
    cells = Counter()
    for uid, v in lab.items():
        c = by_uid[uid]["k3_sympy"]
        cells[(c if c in ("000", "111") else "discordant", y_first(by_uid[uid]) == v["y"])] += 1
    first = [y_first(by_uid[u]) for u in lab]
    second = [lab[u]["y"] for u in lab]
    return {"n": len(lab), "agree": sum(x == y for x, y in zip(first, second)),
            "kappa": cohen_kappa(first, second),
            "by_cell": {c: [cells[(c, True)], cells[(c, True)] + cells[(c, False)]]
                        for c in ("000", "111", "discordant")}}


def certified(census, audit):
    inp = F.Inputs(census, audit, 3, "sympy", "sympy", "as_incorrect", census["models"], "kl")
    fb = F.fixed_board_summary(inp)
    return {"eps0": fb["certified_eps0"], "prereg": fb["certified_eps_preregistered"],
            "pooled": {c: list(v) for c, v in inp.pooled.items()}}


def main():
    census, audit, key = load()
    by_uid = {it["uid"]: it for it in audit["items"]}
    labs = {k: other_labels(f, by_uid, key) for k, f in LABELLERS.items()}
    out = {"y_star": {k: agreement(v, by_uid) for k, v in labs.items()},
           "string_level": json.load(open(ANA / "mathhard_human_margins.json", encoding="utf-8"))["inter_rater"],
           "certified": {"baseline": certified(census, audit)}}
    for k, lab in labs.items():
        alt = copy.deepcopy(audit)
        for it in alt["items"]:
            if it["uid"] in lab:
                it["indeterminate"], it["ystar"]["sympy"] = lab[it["uid"]]["indet"], lab[it["uid"]]["y"]
        out["certified"][f"replace_with_{k}"] = certified(census, alt)
    alt = copy.deepcopy(audit)
    for lab in labs.values():
        for it in alt["items"]:
            v = lab.get(it["uid"])
            if v and it["k3_sympy"] in ("000", "111") and v["y"] != y_first(it):
                it["indeterminate"], it["ystar"]["sympy"] = False, v["y"]
    out["certified"]["any_concordant_disagreement"] = certified(census, alt)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    for k, v in out["y_star"].items():
        print(f"labeller {k}: Y* agreement {v['agree']}/{v['n']}, kappa {v['kappa']:.3f}, by cell {v['by_cell']}")
    for k, v in out["certified"].items():
        print(f"{k:32s} eps=0: {v['eps0']}  registered: {v['prereg']}  pooled {v['pooled']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
