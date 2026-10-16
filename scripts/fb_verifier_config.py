"""G1.3 -- freeze the three verifiers: source hashes, library versions, exact configuration.

Run: python scripts/fb_verifier_config.py   Out: results/flipbudget/verifier_config.json
"""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flipbudget" / "verifier_config.json"


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    from math_verify import ExprExtractionConfig, LatexExtractionConfig

    cfg = {
        "platform": platform.platform(), "python": platform.python_version(),
        "libraries": {p: version(p) for p in ("math-verify", "latex2sympy2_extended", "sympy",
                                              "antlr4-python3-runtime", "pebble")},
        "timeout_mechanism": "original SIGALRM guards replaced by timing; a call longer than its "
                             "original deadline gets the original timeout verdict (0); 60 s pebble kill",
        "verifiers": {
            "em": {"what": "published per-item exact_match in the 2024 samples files (not re-computed)"},
            "em_rec": {
                "source": "provenance/verifier/lmeval_leaderboard_math_utils_3c8db1bb7b.py",
                "source_sha256": sha("provenance/verifier/lmeval_leaderboard_math_utils_3c8db1bb7b.py"),
                "extraction": "get_unnormalized_answer: regex 'Final Answer: The final answer is(.*?). I hope it is correct.'",
                "normalization": "normalize_final_answer on the prediction only",
                "gold": "doc['answer'] (remove_boxed(last_boxed_only_string(solution))), not normalized",
                "equivalence": "is_equiv: sympy parse_latex both sides, simplify(diff) == 0; no exact-string shortcut",
                "deadline_seconds": 5},
            "sb": {
                "source": "vendor/leaderboard_math/utils.py (lm-eval HEAD before 95469434ff)",
                "source_sha256": sha("vendor/leaderboard_math/utils.py"),
                "extraction": "last_boxed_only_string + remove_boxed on the response (last \\boxed{} or \\fbox)",
                "normalization": "normalize_final_answer on prediction and gold",
                "gold": "doc['answer'], normalized",
                "equivalence": {"sb_string": "exact string after normalization only (historical Windows behaviour)",
                                "sb_sympy": "exact string OR is_equiv (sympy parse_latex, simplify(diff) == 0)"},
                "deadline_seconds": 1},
            "mv": {
                "source": "provenance/verifier/lmeval_leaderboard_math_utils_61ee867816.py (process_results)",
                "source_sha256": sha("provenance/verifier/lmeval_leaderboard_math_utils_61ee867816.py"),
                "prediction_parsing": f"parse(response) with defaults {[LatexExtractionConfig(), ExprExtractionConfig()]!r}, "
                                      "fallback_mode='first_match', extraction_mode='any_match'",
                "gold_parsing": f"parse(doc['solution'], extraction_config=[{LatexExtractionConfig()!r}])",
                "equivalence": "verify(gold, pred) defaults: float_rounding=6, numeric_precision=15, strict=True, "
                               "allow_set_relation_comp=False",
                "deadline_seconds": {"parse": 5, "verify": 5}},
        },
    }
    json.dump(cfg, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
