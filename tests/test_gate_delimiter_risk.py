import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fb_gate_delimiter_risk import strip_delimiters  # noqa: E402


def test_strip_delimiters_removes_only_the_delimiters():
    assert strip_delimiters("$\\frac{5}{12}$") == "\\frac{5}{12}"
    assert strip_delimiters("$$7$$") == "7"
    assert strip_delimiters("\\(x=2\\)") == "x=2"
    assert strip_delimiters("$1.25$ miles") == "1.25 miles"  # text outside the math is kept
    assert strip_delimiters("$12$ positive and $12$ negative") == "12 positive and 12 negative"
    assert strip_delimiters("$0,$ $1,$ and $-1$") == "0, 1, and -1"
    assert strip_delimiters("$5$ values for $m\\angle PQR$") == "5 values for m\\angle PQR"
    assert strip_delimiters("1016") == "1016"
