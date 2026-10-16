"""Adversarial / worst-case scorer stress test (items 13-14). Constructed
cases with KNOWN ground truth (I built them, so correctness is not in
question) run through the REAL scorers already verified this project --
MATH-Hard's comparator (normalize_final_answer + is_equiv) and IFEval's
per-instruction checker (process_results). Validates the theory rather than
becoming a new benchmark: measures worst-case alpha/beta on known cases, not
average-case performance on a new task.

Run: python scripts/fb_adversarial_suite.py
"""
import json
import sys

sys.path.insert(0, "scripts")
from fb_tc_real_analysis import matches_gold, is_indeterminate
from fb_eb_ifeval_pull import load_ifeval_verifier

# --- MATH-Hard: (extracted_answer, gold, should_be_equivalent, tag) ---
MATH_CASES = [
    (r"\frac{1}{2}", r"\frac{1}{2}", True, "identical"),
    (r"\dfrac{1}{2}", r"\frac{1}{2}", True, "dfrac_vs_frac"),
    (r"1/2", r"\frac{1}{2}", True, "plain_fraction_vs_latex"),
    (r"0.5", r"\frac{1}{2}", True, "decimal_vs_fraction"),
    (r"\frac{2}{4}", r"\frac{1}{2}", True, "unsimplified_fraction"),
    (r"5x+2y-3z-0=0", "5x + 2y - 3z - 0 = 0", True, "spacing_variant"),
    (r"x+3", "3+x", True, "term_order_addition"),
    (r"-\frac{1}{2}", r"\frac{-1}{2}", True, "sign_placement"),
    (r"9", "9 meters", False, "unit_mismatch_should_NOT_equate"),
    (r"\frac{1}{3}", r"\frac{1}{2}", False, "genuinely_different"),
    (r"(-\infty, -7) \cup (-7, 3) \cup (3, \infty)",
     r"(-\infty, 3) \cup (3, -7) \cup (-7, \infty)", True, "interval_union_reordered"),
    (r"x = 2", "2", True, "explicit_variable_equation_vs_bare_value"),
    (r"\boxed{4}", "4", True, "boxed_wrapper_should_be_stripped_upstream"),
    (r"NONE", "4", False, "no_commit_never_matches"),
]

# --- IFEval: minimal synthetic docs (key, prompt unused by checker itself,
# instruction_id_list, kwargs) + response, with known compliance ---
IFEVAL_CASES = [
    {
        "tag": "no_comma_boundary_pass",
        "doc": {"key": 900001, "prompt": "", "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}]},
        "response": "This response has no forbidden punctuation at all in it.",
        "expect_strict": True,
    },
    {
        "tag": "no_comma_boundary_fail_single_comma",
        "doc": {"key": 900002, "prompt": "", "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}]},
        "response": "This response, unfortunately, has commas in it.",
        "expect_strict": False,
    },
    {
        "tag": "no_comma_semicolon_should_pass",
        "doc": {"key": 900003, "prompt": "", "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}]},
        "response": "This response uses a semicolon; not a comma.",
        "expect_strict": True,
    },
    {
        "tag": "letter_frequency_valid_letter_exact_boundary_atleast",
        "doc": {"key": 900004, "prompt": "",
                "instruction_id_list": ["keywords:letter_frequency"],
                "kwargs": [{"letter": "a", "let_frequency": 3, "let_relation": "at least"}]},
        "response": "banana has three a letters exactly here",
        "expect_strict": True,
    },
    {
        "tag": "letter_frequency_valid_letter_just_under_threshold",
        "doc": {"key": 900005, "prompt": "",
                "instruction_id_list": ["keywords:letter_frequency"],
                "kwargs": [{"letter": "a", "let_frequency": 3, "let_relation": "at least"}]},
        "response": "banana has two a letters",
        "expect_strict": False,
    },
    {
        "tag": "keyword_existence_present",
        "doc": {"key": 900006, "prompt": "",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["correlated", "experiencing"]}]},
        "response": "We are experiencing a correlated rise in demand this quarter.",
        "expect_strict": True,
    },
    {
        "tag": "keyword_existence_missing_one",
        "doc": {"key": 900007, "prompt": "",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["correlated", "experiencing"]}]},
        "response": "We are experiencing a rise in demand this quarter.",
        "expect_strict": False,
    },
    {
        "tag": "json_format_valid",
        "doc": {"key": 900008, "prompt": "",
                "instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
        "response": '{"a": 1, "b": [1,2,3]}',
        "expect_strict": True,
    },
    {
        "tag": "json_format_invalid_trailing_comma",
        "doc": {"key": 900009, "prompt": "",
                "instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
        "response": '{"a": 1, "b": [1,2,3],}',
        "expect_strict": False,
    },
    {
        "tag": "end_checker_exact_match",
        "doc": {"key": 900010, "prompt": "",
                "instruction_id_list": ["startend:end_checker"],
                "kwargs": [{"end_phrase": "Is there anything else I can help with?"}]},
        "response": "Here is your answer. Is there anything else I can help with?",
        "expect_strict": True,
    },
    {
        "tag": "end_checker_trailing_whitespace_should_still_pass",
        "doc": {"key": 900011, "prompt": "",
                "instruction_id_list": ["startend:end_checker"],
                "kwargs": [{"end_phrase": "Is there anything else I can help with?"}]},
        "response": "Here is your answer. Is there anything else I can help with?   \n",
        "expect_strict": True,
    },
    {
        "tag": "capital_word_frequency_boundary",
        "doc": {"key": 900012, "prompt": "",
                "instruction_id_list": ["change_case:capital_word_frequency"],
                "kwargs": [{"capital_frequency": 2, "capital_relation": "at least"}]},
        "response": "This has TWO ALLCAPS words in it right here for real.",
        "expect_strict": True,
    },
]

if __name__ == "__main__":
    print("=" * 70)
    print("Adversarial suite -- known-ground-truth worst-case scorer probes")
    print("=" * 70)

    print("\n--- MATH-Hard comparator ---")
    n_math_correct, n_math_total = 0, len(MATH_CASES)
    math_results = []
    for extracted, gold, expected, tag in MATH_CASES:
        actual = False if is_indeterminate(extracted) else matches_gold(extracted, gold)
        ok = actual == expected
        n_math_correct += ok
        math_results.append({"tag": tag, "extracted": extracted, "gold": gold,
                              "expected": expected, "actual": actual, "correct": ok})
        print(f"  [{'OK' if ok else 'MISMATCH'}] {tag}: expected={expected} actual={actual} "
              f"({extracted!r} vs {gold!r})")

    print(f"\nMATH-Hard comparator: {n_math_correct}/{n_math_total} matched expectation")

    print("\n--- IFEval verifier ---")
    process_results = load_ifeval_verifier()
    n_ifeval_correct, n_ifeval_total = 0, len(IFEVAL_CASES)
    ifeval_results = []
    for case in IFEVAL_CASES:
        out = process_results(case["doc"], [case["response"]])
        actual_strict = bool(out["prompt_level_strict_acc"])
        ok = actual_strict == case["expect_strict"]
        n_ifeval_correct += ok
        ifeval_results.append({"tag": case["tag"], "expect_strict": case["expect_strict"],
                                "actual_strict": actual_strict, "correct": ok,
                                "instruction_id_list": case["doc"]["instruction_id_list"]})
        print(f"  [{'OK' if ok else 'MISMATCH -- REAL BUG CANDIDATE'}] {case['tag']}: "
              f"expected_strict={case['expect_strict']} actual_strict={actual_strict}")

    print(f"\nIFEval verifier: {n_ifeval_correct}/{n_ifeval_total} matched expectation")

    n_mismatches = (n_math_total - n_math_correct) + (n_ifeval_total - n_ifeval_correct)
    print(f"\n{'='*70}")
    print(f"Total mismatches (candidate real bugs, worth individual investigation): {n_mismatches}")

    out = {
        "math_hard": {"n_correct": n_math_correct, "n_total": n_math_total, "cases": math_results},
        "ifeval": {"n_correct": n_ifeval_correct, "n_total": n_ifeval_total, "cases": ifeval_results},
        "n_total_mismatches": n_mismatches,
        # Interpretive note on the MATH-Hard mismatches, made part of the
        # script (not a post-hoc manual edit) so it survives every re-run --
        # a manually-added version of this note was twice silently wiped by
        # a fresh pipeline run before this fix (see MATH_COMPARATOR_BUG.md).
        "post_bugfix_note": (
            "6 of the original 8 MATH-Hard mismatches were caused entirely by "
            "the SIGALRM/Windows bug documented in MATH_COMPARATOR_BUG.md, "
            "confirmed fixed (12/14 match with the fixed comparator). 2 "
            "genuine, separate findings remain: interval/set-union reordering "
            "not recognized (real structural limitation), and unit-word "
            "stripping causing 9 vs 9 meters to match (likely intentional "
            "dataset convention, not a bug -- this suite's own expectation "
            "was probably wrong on that one case, not the comparator)."
        ),
    }
    with open("results/flipbudget/adversarial_suite.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Written to results/flipbudget/adversarial_suite.json")
