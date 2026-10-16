"""G3.4 -- quality gate for one exported batch of the v3 audit (labeling/app/label.html).

Hard checks (any failure excludes the batch):
  structure   every uid on the sheet, no duplicates, one batch id, <= 50 rows, start < end;
  paste       zero blocked paste/drop attempts;
  dwell       median seconds per item >= 15 (MATH);
  burst       no run of more than 20 consecutive items under 3 s each;
  artifacts   fewer than 3 answers with generation artifacts ("I hope it is correct",
              "Final Answer", \\boxed, newlines, > 200 chars);
  extraction  among rows with an extractable answer (last \\boxed{} or Minerva phrase), the number
              typed exactly equal to an extraction is not implausibly high: exact one-sided
              Poisson-binomial p >= 0.001, per-row rates from the genuine 400-item audit (BASELINE).
Reported, not gating: NONE usage on rows with no extractable answer (genuine audit: 29/29).
Overlap agreement (Cohen's kappa >= 0.6 on harness-routine correctness) is `kappa_overlap`.

Run: python scripts/fb_audit_v3_gate.py <batch_export.csv>
Out: provenance/audit_v3_gate/<batch>.json; a failed file is copied to
     provenance/excluded_labeling_attempts/.
"""
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flipbudget import verifiers as V  # noqa: E402

SHEET = ROOT / "labeling" / "audit_v3_l1.csv"
# overridable so that tests and smoke runs never write into the real provenance record
REPORTS = Path(os.environ.get("FB_GATE_REPORTS", ROOT / "provenance" / "audit_v3_gate"))
EXCLUDED = Path(os.environ.get("FB_GATE_EXCLUDED", ROOT / "provenance" / "excluded_labeling_attempts"))
# genuine 400-item audit (results/analysis/mathhard_l1.csv), share typed exactly equal to an extraction:
# rows with a \boxed{} answer 287/350 (credited 146/175 + wrong_parsed 141/175); Minerva-phrase only 2/21
BASELINE = {"boxed": (287, 350), "phrase_only": (2, 21)}
MIN_MEDIAN_DWELL, FAST_SECONDS, MAX_FAST_RUN, MAX_ARTIFACTS, P_AUTOMATION = 15.0, 3.0, 20, 3, 0.001
ARTIFACT_MARKERS = ("i hope it is correct", "final answer", "\\boxed")


def extractions(response: str) -> set:
    return extraction_info(response)[0]


def extraction_info(response: str) -> tuple:
    """(set of regex extractions, 'boxed' | 'phrase_only' | None)."""
    m = V.utils("head")
    out, kind = set(), None
    b = m.last_boxed_only_string(response)
    if b != m.INVALID_ANSWER:
        out.add(m.remove_boxed(b).strip())
        kind = "boxed"
    a = m.get_unnormalized_answer(response)
    if a != m.INVALID_ANSWER:
        out.add(a.strip())
        kind = kind or "phrase_only"
    return out, kind


def poisson_binomial_sf(k: int, ps: list) -> float:
    """P(sum of independent Bernoulli(ps) >= k), exact by convolution."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            nxt[i] += q * (1 - p)
            nxt[i + 1] += q * p
        dist = nxt
    return float(sum(dist[k:]))


def _secs(r):
    return (datetime.fromisoformat(r["end"].replace("Z", "+00:00"))
            - datetime.fromisoformat(r["start"].replace("Z", "+00:00"))).total_seconds()


def gate(records: list, sheet: dict) -> tuple:
    """records: app export rows; sheet: uid -> {"response": ...}. Returns (passed, report)."""
    fail, uids = [], [r["uid"] for r in records]
    dup = sorted({u for u in uids if uids.count(u) > 1})
    if dup:
        fail.append(f"structure: duplicate uids {dup[:5]}")
    unknown = [u for u in uids if u not in sheet]
    if unknown:
        fail.append(f"structure: {len(unknown)} uid(s) not on the sheet")
    if len({r["batch_id"] for r in records}) != 1 or len(records) > 50:
        fail.append("structure: expected one batch id and at most 50 rows")
    dwell = [_secs(r) for r in records]
    if any(d <= 0 for d in dwell):
        fail.append("structure: an end time is not after its start time")
    paste = sum(int(r["paste_attempts"] or 0) for r in records)
    if paste:
        fail.append(f"paste: {paste} blocked paste/drop attempt(s)")
    med = statistics.median(dwell) if dwell else 0.0
    if med < MIN_MEDIAN_DWELL:
        fail.append(f"dwell: median {med:.1f} s < {MIN_MEDIAN_DWELL} s")
    run = longest = 0
    for d in dwell:
        run = run + 1 if d < FAST_SECONDS else 0
        longest = max(longest, run)
    if longest > MAX_FAST_RUN:
        fail.append(f"burst: {longest} consecutive items under {FAST_SECONDS} s")
    artifacts = [r["uid"] for r in records
                 if any(k in r["your_answer"].lower() for k in ARTIFACT_MARKERS)
                 or "\n" in r["your_answer"] or len(r["your_answer"]) > 200]
    if len(artifacts) >= MAX_ARTIFACTS:
        fail.append(f"artifacts: {len(artifacts)} answers with generation artifacts")
    info = {r["uid"]: extraction_info(sheet[r["uid"]]["response"]) for r in records if r["uid"] in sheet}
    ex = {u: v[0] for u, v in info.items()}
    extractable = [r for r in records if ex.get(r["uid"])]
    equal = sum(r["your_answer"].strip() in ex[r["uid"]] for r in extractable)
    ps = [BASELINE[info[r["uid"]][1]][0] / BASELINE[info[r["uid"]][1]][1] for r in extractable]
    p_auto = poisson_binomial_sf(equal, ps) if extractable else 1.0
    if p_auto < P_AUTOMATION:
        fail.append(f"extraction: {equal}/{len(extractable)} typed exactly as a regex extraction "
                    f"(p = {p_auto:.2g} vs genuine baseline)")
    no_ex = [r for r in records if r["uid"] in ex and not ex[r["uid"]]]
    report = {"n": len(records), "batch_id": records[0]["batch_id"] if records else None,
              "median_dwell_s": med, "longest_fast_run": longest, "paste_attempts": paste,
              "artifact_uids": artifacts, "extractable": len(extractable), "equal_to_extraction": equal,
              "p_automation": p_auto,
              "none_on_non_extractable": [sum(r["your_answer"].strip().upper() == "NONE" for r in no_ex), len(no_ex)],
              "n_none": sum(r["your_answer"].strip().upper() == "NONE" for r in records),
              "n_contradictory": sum(r["your_answer"].strip().upper() == "CONTRADICTORY" for r in records),
              "failures": fail}
    return not fail, report


def kappa_overlap(y1: dict, y2: dict) -> float:
    """Cohen's kappa for two binary label maps over their common uids."""
    common = sorted(set(y1) & set(y2))
    n = len(common)
    po = sum(y1[u] == y2[u] for u in common) / n
    p1, p2 = sum(y1[u] for u in common) / n, sum(y2[u] for u in common) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


SHEETS = [SHEET] + [ROOT / "labeling" / n for n in ("audit_v3_l2_overlap.csv", "relabel_check_l1.csv",
                                                      "discordant_census_l1.csv")]


def sheet_for(records: list) -> tuple:
    """The registered sheet this batch was labelled from, found by its first uid. The gate's
    checks are the same for every sheet; only the reference rows differ."""
    first = records[0]["uid"] if records else None
    for p in SHEETS:
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                rows = {r["uid"]: r for r in csv.DictReader(f)}
            if first in rows:
                return p, rows
    return SHEET, {}  # unknown uids: the structure check fails the batch


def main():
    path = Path(sys.argv[1])
    raw = path.read_bytes()
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    sheet_path, sheet = sheet_for(records)
    ok, rep = gate(records, sheet)
    rep["sheet"] = sheet_path.name
    rep["file"], rep["sha256"], rep["passed"] = path.name, hashlib.sha256(raw).hexdigest(), ok
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{path.stem}.json"
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    if not ok:
        EXCLUDED.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, EXCLUDED / path.name)
    print("PASS" if ok else "FAIL", rep["failures"], "->", out)


if __name__ == "__main__":
    main()
