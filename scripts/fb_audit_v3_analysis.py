"""G3.5 + G4.1 -- turn exported v3 label batches into audit counts and run the falsification test.

Reads every batch in labeling/exports/ (a batch that fails scripts/fb_audit_v3_gate.py is
excluded and named in the output), joins uids to results/analysis/audit_v3_provenance.json,
and scores each label with the harness routine (primary) plus the string and Math-Verify
references as sensitivities.

Falsification test, exactly as preregistered (TC_EXPANSION_PREREGISTRATION.md Amendment 4):
reject H(eps0) for a screened model m in concordant cell c if LB(e_mc) - UB(pooled e_c) > eps0,
with KL bounds at one-sided alpha_F/(2F+1) and the pooled cell taking the existing audit plus
all new labels. Valid because min_m e_mc <= pooled e_c.

Run: python scripts/fb_audit_v3_analysis.py [--allow-failed-batches]
Out: results/flipbudget/audit_v3_labels.json  (counts, per-model errors, test, kappa)
"""
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import certify as C  # noqa: E402
from flipbudget.verifiers import human_correct, is_indeterminate  # noqa: E402
from fb_audit_v3_gate import gate, kappa_overlap  # noqa: E402
from fb_certify import ALPHA_A, EPS0  # noqa: E402
from fb_g3_frontier import ALPHA_F, fals_level  # noqa: E402

EXPORTS = Path(os.environ.get("FB_EXPORTS", ROOT / "labeling" / "exports"))
SHEET = ROOT / "labeling" / "audit_v3_l1.csv"
OVERLAP = ROOT / "labeling" / "audit_v3_l2_overlap.csv"
PROV = ROOT / "results" / "analysis" / "audit_v3_provenance.json"
OP = ROOT / "results" / "flipbudget" / "g3_operating_point.json"
OUT = Path(os.environ.get("FB_V3_OUT", ROOT / "results" / "flipbudget" / "audit_v3_labels.json"))
REFS = ("sympy", "string", "mv")


def read_sheet(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["uid"]: r for r in csv.DictReader(f)}


def extraction_only(failures):
    """True when every gate failure is the extraction-agreement check (decision D8)."""
    return bool(failures) and all(f.startswith("extraction:") for f in failures)


def load_batches(sheet, allow_failed=False, subdir="", restore_extraction_only=False):
    """Returns (records, excluded): app exports that passed the gate, and the batches that did not.
    The primary labeler's batches are labeling/exports/*.csv; the second labeler's are in l2/.
    restore_extraction_only re-admits batches whose only failure is the extraction check, for the
    inclusion sensitivity of decision D8; the registered gate is otherwise unchanged."""
    records, excluded = {}, []
    for path in sorted((EXPORTS / subdir).glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        ok, rep = gate(rows, sheet)
        restored = restore_extraction_only and extraction_only(rep["failures"])
        if not ok and not allow_failed and not restored:
            excluded.append({"file": path.name, "failures": rep["failures"]})
            continue
        for r in rows:
            if r["uid"] in records and records[r["uid"]]["your_answer"] != r["your_answer"]:
                raise SystemExit(f"conflicting labels for {r['uid']} in {path.name}")
            records[r["uid"]] = r | {"file": path.name}
    return records, excluded


def score(records, prov, gold, solution):
    """Adds Y* under each reference routine. Indeterminate answers are kept as a flag."""
    out = {}
    for uid, r in records.items():
        p = prov.get(uid)
        if p is None:
            continue
        a = r["your_answer"]
        indet = is_indeterminate(a)
        out[uid] = {"model": p["model"], "item_id": p["item_id"], "cell": p["cell"], "role": p["role"],
                    "in_overlap": p["in_overlap"], "indeterminate": indet, "file": r["file"],
                    "ystar": {ref: None if indet else human_correct(
                        a, solution[p["item_id"]] if ref == "mv" else gold[p["item_id"]], ref) for ref in REFS}}
    return out


def counts(labels, ref, indet="as_incorrect"):
    """(pooled concordant, per (model, cell)) counts of Y*=1, in the certify input shape."""
    pooled, per = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for v in labels.values():
        if v["indeterminate"] and indet == "drop":
            continue
        y = 0 if v["indeterminate"] else v["ystar"][ref]
        c, m = v["cell"], v["model"]
        per[(m, c)][0] += y
        per[(m, c)][1] += 1
        if c in ("000", "111"):
            pooled[c][0] += y
            pooled[c][1] += 1
    return {c: list(v) for c, v in pooled.items()}, {k: list(v) for k, v in per.items()}


def errors(x, n, cell):
    """Human-judged errors in a concordant cell: Y*=0 where all verifiers credit, Y*=1 where none do."""
    return n - x if cell == "111" else x


def falsification(per, pooled_new, pooled_existing, op):
    """The preregistered range test (Amendment 5: leave-one-out pool). Returns (rejected, rows).

    For model m in concordant cell c the pooled comparison excludes m's own labels, because
    including them lets a bad model inflate the bound it is tested against. Validity is
    unchanged: the range over sub-populations is at least e_mc - min_{g not in m} e_gc, and the
    leave-one-out pooled error is a weighted mean of those other sub-populations, so its upper
    bound bounds their minimum."""
    F = op["F"]
    lvl = fals_level(F)
    rows = []
    for c in ("000", "111"):
        xe, ne = pooled_existing[c]
        xn, nn = pooled_new.get(c, [0, 0])
        for m in op["falsification_models"]:
            x, n = per.get((m, c), [0, 0])
            if not n:
                continue
            e = errors(x, n, c)
            # pool = existing audit + every new label in this cell except this model's own
            loo_err = errors(xe, ne, c) + errors(xn - x, nn - n, c)
            loo_n = ne + nn - n
            ub = C.kl_interval(loo_err, loo_n, lvl)[1]
            lb = C.kl_interval(e, n, lvl)[0]
            rows.append({"model": m, "cell": c, "errors": e, "n": n, "lb_model": lb,
                         "pooled_errors_excl_model": loo_err, "pooled_n_excl_model": loo_n, "ub_pooled": ub,
                         "margin": lb - ub, "reject": bool(lb - ub > op["eps0"])})
    return any(r["reject"] for r in rows), rows


def main():
    allow = "--allow-failed-batches" in sys.argv
    sheet = read_sheet(SHEET)
    prov = {r["uid"]: r for r in json.load(open(PROV, encoding="utf-8"))["rows"]}
    op = json.load(open(OP, encoding="utf-8"))
    census = json.load(gzip.open(ROOT / "results" / "flipbudget" / "census_k3.json.gz", "rt", encoding="utf-8"))
    from fb_census_k3 import load_rows
    ref_rows = load_rows(census["models"][:1])[0]  # gold and solution are per item, not per model
    gold = {r["item_id"]: r["gold"] for r in ref_rows}
    solution = {r["item_id"]: r["solution"] for r in ref_rows}
    records, excluded = load_batches(sheet, allow)
    labels = score(records, prov, gold, solution)
    # decision D8: the same analysis with the batches excluded only by the extraction check restored
    d8_records, d8_excluded = load_batches(sheet, allow, restore_extraction_only=True)
    d8_labels = score(d8_records, prov, gold, solution)
    if not labels and not d8_labels:
        print("no usable labels yet in", EXPORTS)
        return
    overlap_uids = set(read_sheet(OVERLAP))
    by_file = defaultdict(list)
    for uid, v in labels.items():
        by_file[v["file"]].append(uid)
    out = {"n_labels": len(labels), "excluded_batches": excluded,
           "batches": {f: len(v) for f, v in sorted(by_file.items())},
           "n_indeterminate": sum(v["indeterminate"] for v in labels.values()),
           "alpha_falsification": ALPHA_F, "alpha_audit": ALPHA_A, "eps0": EPS0,
           "counts": {}, "falsification": {}}
    # certify inputs and the test, per reference routine (sympy primary)
    existing = json.load(open(ROOT / "results" / "flipbudget" / "certify_results.json", encoding="utf-8"))
    pooled_existing = {c: v for c, v in
                       [(c, [d["x_correct"], d["n"]]) for c, d in existing["K3"]["inputs"]["audit_pooled"].items()]}
    for ref in REFS:
        pooled, per = counts(labels, ref)
        rejected, rows = falsification(per, pooled, pooled_existing, op)
        out["counts"][ref] = {"pooled": pooled, "per_model_cell": {f"{m}|{c}": v for (m, c), v in sorted(per.items())}}
        out["falsification"][ref] = {"rejected": rejected, "rows": rows}
    restored = sorted({r["file"] for r in d8_records.values()} - {r["file"] for r in records.values()})
    out["d8_extraction_restored"] = {"restored_batches": restored, "n_labels": len(d8_labels),
                                     "excluded_batches": d8_excluded, "counts": {}, "falsification": {}}
    for ref in REFS:
        pooled, per = counts(d8_labels, ref)
        rejected, rows = falsification(per, pooled, pooled_existing, op)
        out["d8_extraction_restored"]["counts"][ref] = {
            "pooled": pooled, "per_model_cell": {f"{m}|{c}": v for (m, c), v in sorted(per.items())}}
        out["d8_extraction_restored"]["falsification"][ref] = {"rejected": rejected, "rows": rows}
    l2_records, l2_excluded = load_batches(sheet, allow, "l2")
    l2_labels = score(l2_records, prov, gold, solution)
    y1 = {u: v["ystar"]["sympy"] for u, v in labels.items() if v["ystar"]["sympy"] is not None}
    y2 = {u: v["ystar"]["sympy"] for u, v in l2_labels.items() if v["ystar"]["sympy"] is not None}
    common = set(y1) & set(y2)
    out["overlap"] = {"n_overlap_sheet": len(overlap_uids), "n_second_labeler": len(l2_labels),
                      "n_common_scored": len(common), "excluded_batches_l2": l2_excluded,
                      "kappa": kappa_overlap(y1, y2) if common else None,
                      "kappa_threshold": 0.6,
                      "agreement": sum(y1[u] == y2[u] for u in common) / len(common) if common else None}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    f = out["falsification"]["sympy"]
    print(f"{out['n_labels']} labels from {len(by_file)} batch file(s); excluded {len(excluded)}; "
          f"restored under D8 {len(restored)}")
    print("falsification (sympy):", "REJECTED H(eps0)" if f["rejected"] else "no rejection")
    for r in f["rows"]:
        print(f"  {r['model'].split('/')[-1]:32s} cell {r['cell']}  {r['errors']}/{r['n']}  "
              f"LB {r['lb_model']:.4f} - UB {r['ub_pooled']:.4f} = {r['margin']:+.4f}  reject={r['reject']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
