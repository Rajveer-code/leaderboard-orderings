"""The Job A gate ignores extraction equality and nothing else (VALIDATION_RELABEL.md, 2026-10-03)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fb_relabel_agreement import job_a_gate_ok  # noqa: E402


def test_extraction_failure_alone_passes():
    assert job_a_gate_ok({"failures": ["extraction: 52/54 typed exactly as a regex extraction (p = 1e-07)"]})


def test_other_failures_still_fail():
    for f in ("paste: 1 blocked paste/drop attempt(s)", "dwell: median 9.0 s < 15.0 s",
              "burst: 21 consecutive items under 3.0 s", "artifacts: 3 answers with generation artifacts",
              "structure: expected one batch id and at most 50 rows"):
        assert not job_a_gate_ok({"failures": ["extraction: x", f]})


def test_clean_batch_passes():
    assert job_a_gate_ok({"failures": []})
