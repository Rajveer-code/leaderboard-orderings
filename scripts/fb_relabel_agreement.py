"""Measure the re-labeling check against the earlier 458-row submissions (VALIDATION_RELABEL.md).

Fresh labels: labeling/exports/relabel/*.csv (exported from labeling/app/label.html on the
check sheet). Compared with the round-3 and round-1 submissions on the same uids.

Criteria, fixed before labeling: the batch must pass the quality gate, and agreement on Y*
must reach kappa >= 0.8 with raw agreement >= 0.90. For this check the gate is every check of
fb_audit_v3_gate.py except the extraction-equality one, which VALIDATION_RELABEL.md reports as a
secondary, non-gating statistic (clarification of 2026-10-03, written before any re-label existed).

Run: python scripts/fb_relabel_agreement.py   Out: results/flipbudget/relabel_agreement.json
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget.verifiers import human_correct, is_indeterminate  # noqa: E402
from fb_audit_v3_gate import BASELINE, extraction_info, gate, kappa_overlap, poisson_binomial_sf  # noqa: E402

CHECK = ROOT / "labeling" / "relabel_check_l1.csv"
EXPORTS = ROOT / "labeling" / "exports" / "relabel"
PROV = ROOT / "results" / "analysis" / "tc_expansion2_provenance.json"
EXCLUDED = ROOT / "provenance" / "excluded_labeling_attempts"
OUT = ROOT / "results" / "flipbudget" / "relabel_agreement.json"
KAPPA_MIN, AGREE_MIN = 0.8, 0.90


def ystar(answer, gold):
    return 0 if is_indeterminate(answer) else human_correct(answer, gold, "sympy")


def extraction_stats(pairs):
    """pairs: (answer, response). Returns (equal, n_extractable, p under the genuine baseline)."""
    eq, ps = 0, []
    for ans, resp in pairs:
        ex, kind = extraction_info(resp)
        if not ex:
            continue
        eq += ans.strip() in ex
        ps.append(BASELINE[kind][0] / BASELINE[kind][1])
    return eq, len(ps), (poisson_binomial_sf(eq, ps) if ps else 1.0)


def job_a_gate_ok(rep: dict) -> bool:
    """Every check of the quality gate except extraction equality (VALIDATION_RELABEL.md, 2026-10-03)."""
    return not [f for f in rep["failures"] if not f.startswith("extraction:")]


def main():
    with open(CHECK, newline="", encoding="utf-8") as f:
        sheet = {r["uid"]: r for r in csv.DictReader(f)}
    gold = {r["uid"]: r["gold"] for r in json.load(open(PROV, encoding="utf-8"))["rows"]}
    fresh, batches = {}, []
    for path in sorted(EXPORTS.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        full_ok, rep = gate(rows, sheet)
        ok = job_a_gate_ok(rep)
        batches.append({"file": path.name, "passed": ok, "passed_all_checks": full_ok, "report": rep})
        for r in rows:
            fresh[r["uid"]] = r
    if not fresh:
        raise SystemExit(f"no re-labeled batches in {EXPORTS}")
    out = {"n_fresh": len(fresh), "n_check_sheet": len(sheet), "batches": batches,
           "gate_passed": all(b["passed"] for b in batches),
           "criteria": {"kappa_min": KAPPA_MIN, "agreement_min": AGREE_MIN}}
    dwell = [b["report"]["median_dwell_s"] for b in batches]
    out["median_dwell_s"] = statistics.median(dwell)
    out["paste_attempts"] = sum(b["report"]["paste_attempts"] for b in batches)
    eq, n_ex, p = extraction_stats([(r["your_answer"], sheet[r["uid"]]["response"])
                                    for r in fresh.values() if r["uid"] in sheet])
    out["fresh_extraction_equality"] = {"equal": eq, "extractable": n_ex, "rate": eq / n_ex if n_ex else None,
                                        "p_vs_genuine_baseline": p}
    for name, fn in (("round3", "tc_expansion2_round3_submission.csv"),
                     ("round1", "tc_expansion2_round1_submission.csv")):
        with open(EXCLUDED / fn, newline="", encoding="utf-8") as f:
            old = {r["uid"]: r.get("your_answer", "") for r in csv.DictReader(f)}
        common = sorted(set(fresh) & set(old) & set(gold))
        y_new = {u: ystar(fresh[u]["your_answer"], gold[u]) for u in common}
        y_old = {u: ystar(old[u], gold[u]) for u in common}
        agree = sum(y_new[u] == y_old[u] for u in common) / len(common)
        exact = sum(fresh[u]["your_answer"].strip() == old[u].strip() for u in common) / len(common)
        k = kappa_overlap(y_new, y_old)
        out[name] = {"n_common": len(common), "ystar_agreement": agree, "kappa": k,
                     "exact_string_agreement": exact,
                     "disagreeing_uids": [u for u in common if y_new[u] != y_old[u]][:20]}
    r3 = out["round3"]
    out["verdict"] = {
        "passed": bool(out["gate_passed"] and r3["kappa"] >= KAPPA_MIN and r3["ystar_agreement"] >= AGREE_MIN),
        "means": "PASS -> all 458 labels enter the analysis, reliability reported; "
                 "FAIL -> they stay excluded and the basis is stated in the limitations (VALIDATION_RELABEL.md)"}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print(f"gate passed: {out['gate_passed']} | median dwell {out['median_dwell_s']:.0f}s | "
          f"paste attempts {out['paste_attempts']}")
    print(f"fresh labels typed == extraction: {eq}/{n_ex} = {eq / n_ex:.3f} (p vs genuine baseline {p:.3g})")
    for name in ("round3", "round1"):
        v = out[name]
        print(f"vs {name}: n={v['n_common']} | Y* agreement {v['ystar_agreement']:.3f} | kappa {v['kappa']:.3f} "
              f"| exact-string {v['exact_string_agreement']:.3f}")
    print("VERDICT:", "PASS" if out["verdict"]["passed"] else "FAIL", "->", OUT)


if __name__ == "__main__":
    main()
