"""Count the repository's checks so Appendix B can state the number without typing it.

The manuscript rule is that every number in the paper comes from a committed result file.
Appendix B describes the numerical checks of the propositions, and the count of those checks
is itself a number, so it is measured here rather than written by hand. Parametrized cases
count individually, because each one is a separate comparison against an independent
computation.

Run: python scripts/fb_test_census.py
Out: results/flipbudget/test_census.json
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flipbudget" / "test_census.json"
# Files whose counts the manuscript cites by name.
NAMED = {
    "test_certification.py": "certification",
    "test_manuscript_claims.py": "manuscript_claims",
    "test_regression_anchor.py": "regression_anchor",
    "test_verifiers.py": "verifiers",
    "test_audit_gate.py": "audit_gate",
    "test_audit_v3_analysis.py": "audit_v3_analysis",
}


def main():
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise SystemExit("collection failed; fix the suite before regenerating the census")
    # Node ids look like "tests/test_certification.py::test_name[param]".
    ids = [ln.strip() for ln in r.stdout.splitlines() if re.match(r"^[\w/\\.-]+\.py::", ln.strip())]
    per_file = Counter(Path(i.split("::")[0]).name for i in ids)
    per_func = Counter(f"{Path(i.split('::')[0]).name}::{i.split('::')[1].split('[')[0]}" for i in ids)
    census = {
        "total_cases": len(ids),
        "total_functions": len(per_func),
        "per_file": dict(sorted(per_file.items())),
        "named": {NAMED[f]: per_file.get(f, 0) for f in NAMED},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(census, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}: {census['total_cases']} cases in {census['total_functions']} functions")
    for k, v in census["named"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
