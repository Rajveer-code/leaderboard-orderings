"""G3.4 quality gate on synthetic app exports: genuine-looking passes, each failure mode fails."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fb_audit_v3_gate import gate  # noqa: E402

T0 = datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc)
SHEET = {f"u{i:03d}": {"response": f"so the answer is \\boxed{{{i}}}."} for i in range(50)}


def export(answers, dwell=30.0, paste=0, batch="h-b1-x"):
    recs, t = [], T0
    for i, a in enumerate(answers):
        d = dwell[i] if isinstance(dwell, list) else dwell
        recs.append({"uid": f"u{i:03d}", "your_answer": a, "start": t.isoformat(), "end": (t + timedelta(seconds=d)).isoformat(),
                     "batch_id": batch, "paste_attempts": str(paste)})
        t += timedelta(seconds=d)
    return recs


def genuine():
    # ~78% equal to the boxed extraction, the rest differing as real transcriptions do
    return [str(i) if i % 9 != 0 else f"x={i}" for i in range(50)]


def test_genuine_batch_passes():
    ok, rep = gate(export(genuine()), SHEET)
    assert ok, rep["failures"]


def test_paste_attempt_fails():
    ok, rep = gate(export(genuine(), paste=1), SHEET)
    assert not ok and any("paste" in f for f in rep["failures"])


def test_fast_dwell_fails():
    ok, rep = gate(export(genuine(), dwell=5.0), SHEET)
    assert not ok and any("dwell" in f for f in rep["failures"])


def test_burst_fails_even_with_ok_median():
    dw = [2.0] * 21 + [40.0] * 29
    ok, rep = gate(export(genuine(), dwell=dw), SHEET)
    assert not ok and any("burst" in f for f in rep["failures"])


def test_automated_extraction_pattern_fails():
    ok, rep = gate(export([str(i) for i in range(50)]), SHEET)
    assert not ok and any("extraction" in f for f in rep["failures"])


def test_structural_problems_fail():
    recs = export(genuine())
    recs[1]["uid"] = recs[0]["uid"]
    ok, rep = gate(recs, SHEET)
    assert not ok and any("duplicate" in f for f in rep["failures"])
    recs = export(genuine())
    recs[2]["uid"] = "not-in-sheet"
    assert not gate(recs, SHEET)[0]


def test_transcription_artifacts_fail():
    ans = genuine()
    for i in (1, 2, 3):
        ans[i] = f"{i}. I hope it is correct."
    ok, rep = gate(export(ans), SHEET)
    assert not ok and any("artifact" in f for f in rep["failures"])


def test_kappa_overlap():
    from fb_audit_v3_gate import kappa_overlap
    a = {f"u{i}": int(i % 3 == 0) for i in range(30)}
    assert kappa_overlap(a, a) == 1.0
    b = dict(a, u1=1 - a["u1"], u2=1 - a["u2"])  # 28/30 agree
    po, p1, p2 = 28 / 30, sum(a.values()) / 30, sum(b.values()) / 30
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    assert abs(kappa_overlap(a, b) - (po - pe) / (1 - pe)) < 1e-12
