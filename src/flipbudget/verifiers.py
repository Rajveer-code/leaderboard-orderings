"""The three MATH-Hard verifiers, run faithfully on Windows.

Each verifier is the original lm-eval / Math-Verify code, loaded from a pinned source file.
The only change is the timeout mechanism. The originals time out with SIGALRM, which Windows
lacks: lm-eval's `is_equiv` then swallows the AttributeError and returns False for every
call (MATH_COMPARATOR_BUG.md), and Math-Verify falls back to one process per call. Here the
signal guard is replaced by a no-op, each guarded call is timed, and a call that ran past its
original deadline gets the verdict the original gives on timeout (not equivalent). The
`timed_out` flag records every such case. True hangs are killed by the caller's per-task
watchdog (scripts/fb_census_k3.py).

Verifiers (provenance in results/flipbudget/verifier_provenance.json):
  em_recorded  process_results recorded in the leaderboard's 2024 results JSON
               (= upstream lm-eval 3c8db1bb7b): Minerva "Final Answer" regex,
               normalize_final_answer, is_equiv(answer, raw gold), 5 s deadline.
  sb           this project's boxed verifier: last \\boxed{} of the response,
               normalize both sides, exact string OR is_equiv (lm-eval HEAD helpers, 1 s).
               mode="string" skips is_equiv -- the historical Windows behaviour behind
               every score_boxed verdict computed before 2026-09-19.
  mv           Math-Verify 0.9.0 via lm-eval 61ee867816 process_results: parse(response),
               parse(solution, [LatexExtractionConfig()]), verify(gold, pred); 5 s deadlines.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD_UTILS = ROOT / "vendor" / "leaderboard_math" / "utils.py"
RECORDED_UTILS = ROOT / "provenance" / "verifier" / "lmeval_leaderboard_math_utils_3c8db1bb7b.py"
IS_EQUIV_DEADLINE = {"head": 1.0, "recorded": 5.0}  # seconds, from each file's timeout(seconds=)
MV_DEADLINE = 5.0  # math-verify 0.9.0 defaults: parsing_timeout=5, timeout_seconds=5


class _NoSignalTimeout:
    """Stands in for lm-eval's SIGALRM `timeout` context manager; deadlines are timed outside."""

    def __init__(self, seconds=1, error_message="Timeout"):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _load(path: Path, name: str):
    if "datasets" not in sys.modules:  # utils.py annotates process_docs with datasets.Dataset
        stub = types.ModuleType("datasets")
        stub.Dataset = object
        sys.modules["datasets"] = stub
    if "lm_eval.utils" not in sys.modules:  # the 3c8db1bb7b file imports lm_eval.utils.eval_logger
        lm_eval, lm_utils = types.ModuleType("lm_eval"), types.ModuleType("lm_eval.utils")
        lm_utils.eval_logger = logging.getLogger("lm_eval")
        lm_eval.utils = lm_utils
        sys.modules.setdefault("lm_eval", lm_eval)
        sys.modules["lm_eval.utils"] = lm_utils
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.timeout = _NoSignalTimeout
    return module


_MODULES: dict[str, types.ModuleType] = {}


def utils(version: str):
    if version not in _MODULES:
        _MODULES[version] = _load(HEAD_UTILS if version == "head" else RECORDED_UTILS, f"lb_math_{version}")
    return _MODULES[version]


def _timed_is_equiv(version: str, a: str, b: str) -> tuple[bool, bool]:
    t0 = time.perf_counter()
    verdict = bool(utils(version).is_equiv(a, b))
    timed_out = time.perf_counter() - t0 > IS_EQUIV_DEADLINE[version]
    return (False if timed_out else verdict), timed_out


def em_template_found(response: str) -> bool:
    """Whether the 2024 scorer's own extraction finds an answer, i.e. whether the response
    follows the output template its prompt demonstrates. Decided by the recorded scorer's
    get_unnormalized_answer, so compliance means exactly what that scorer requires."""
    # the recorded file defines its sentinel inside get_unnormalized_answer (line 188)
    return utils("recorded").get_unnormalized_answer(response) != "[invalidanswer]"


def em_recorded(response: str, gold: str) -> dict:
    m = utils("recorded")
    answer = m.normalize_final_answer(m.get_unnormalized_answer(response))
    verdict, timed_out = _timed_is_equiv("recorded", answer, gold)
    return {"verdict": int(verdict), "timed_out": timed_out}


def sb(response: str, gold: str, mode: str = "sympy") -> dict:
    m = utils("head")
    boxed = m.last_boxed_only_string(response)
    if not boxed or boxed == m.INVALID_ANSWER:
        return {"verdict": 0, "found": False, "timed_out": False}
    try:
        answer = m.normalize_final_answer(m.remove_boxed(boxed))
    except (AssertionError, IndexError, ValueError):
        return {"verdict": 0, "found": False, "timed_out": False}
    g = m.normalize_final_answer(gold)
    if answer.strip() == g.strip():
        return {"verdict": 1, "found": True, "timed_out": False}
    if mode == "string":
        return {"verdict": 0, "found": True, "timed_out": False}
    verdict, timed_out = _timed_is_equiv("head", answer, g)
    return {"verdict": int(verdict), "found": True, "timed_out": timed_out}


def is_indeterminate(answer) -> bool:
    """Human sentinel for 'cannot commit to an answer' (scripts/64_score_mathhard_human.py)."""
    return answer is None or answer.strip() == "" or answer.strip().upper() in ("NONE", "CONTRADICTORY")


def human_correct(answer: str, gold: str, mode: str) -> int:
    """Human reference Y*: does the labeller's transcription equal gold?
    string: normalize both, exact string (the historical Windows behaviour of matches_gold);
    sympy:  string OR is_equiv (lm-eval HEAD, 1 s);
    mv:     Math-Verify verify(parse(solution), parse($answer$)); for this mode `gold` must be the
            full reference solution, as in the mv verifier: the dataset's doc["answer"] is
            already normalize_final_answer'd ("ft" deleted: \\infty -> \\iny), which Math-Verify
            cannot parse.
    Indeterminate answers are handled by the caller."""
    if mode == "mv":
        return mv(f"${answer}$", gold)["verdict"]
    m = utils("head")
    a, g = m.normalize_final_answer(answer), m.normalize_final_answer(gold)
    if mode in ("string", "sympy") and a.strip() == g.strip():
        return 1
    if mode == "string":
        return 0
    return int(_timed_is_equiv("head", a, g)[0])


def mv(response: str, solution: str) -> dict:
    from math_verify import LatexExtractionConfig, parse, verify

    timed_out = False
    t0 = time.perf_counter()
    pred = parse(response, parsing_timeout=None)
    if time.perf_counter() - t0 > MV_DEADLINE:
        pred, timed_out = [], True
    t0 = time.perf_counter()
    gold = parse(solution, extraction_config=[LatexExtractionConfig()], parsing_timeout=None)
    if time.perf_counter() - t0 > MV_DEADLINE:
        gold, timed_out = [], True
    t0 = time.perf_counter()
    verdict = bool(verify(gold, pred, timeout_seconds=None))
    if time.perf_counter() - t0 > MV_DEADLINE:
        verdict, timed_out = False, True
    return {"verdict": int(verdict), "pred_parsed": bool(pred), "gold_parsed": bool(gold),
            "timed_out": timed_out}
