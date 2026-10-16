"""The per-model and per-subject tables reconcile with the totals the paper reports elsewhere."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = json.load(open(ROOT / "results" / "analysis" / "model_subject_table.json", encoding="utf-8"))
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{,\})*)\}",
                         (ROOT / "manuscript" / "tmlr" / "macros.tex").read_text(encoding="utf-8")))


def macro_int(name):
    return int(MACROS[name].replace("{,}", ""))


def test_totals_match_the_reported_board():
    for part in ("per_model", "per_subject"):
        groups = T[part].values()
        assert sum(g["n"] for g in groups) == macro_int("NumRows")
        assert round(sum(g["template"] * g["n"] for g in groups)) == macro_int("TemplateCompliant")
        assert sum(g["discordant"] for g in groups) == macro_int("RowsAnyDisagree")
        assert sum(g["discordant_extraction"] for g in groups) == macro_int("TaxExtraction")
        assert round(sum(g["acc"]["xf"] * g["n"] for g in groups)) == macro_int("XfCredited")


def test_ranks_are_competition_ranks():
    for v, r in T["ranks"].items():
        acc = {m: T["per_model"][m]["acc"][v] for m in r}
        for m in r:
            assert r[m] == 1 + sum(a > acc[m] for a in acc.values())
    for m, s in T["rank_spread"].items():
        rs = [T["ranks"][v][m] for v in T["verifiers"]]
        assert s == max(rs) - min(rs)
