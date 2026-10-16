"""Were the board's responses the same before and after the February 2025 re-scoring?

The board-wide reversal statistic (fb_board_verifier_shift.py) attributes every reordering to the
verifier, which holds only if the MATH-Hard responses were not regenerated between the two scores.
The roster was checked file by file (fb_verifier_provenance.py). This script checks a seeded random
sample of the models with both scores the same way: every MATH-Hard samples file in the model's
details repository must have been last committed on or before the date of the pre-re-scoring
snapshot (2025-01-31). A model whose details repository is gone is reported as unavailable, not as
a pass.

Run: python scripts/fb_board_response_dates.py      Out: results/flipbudget/board_response_dates.json
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fb_board_verifier_shift import CONTENTS, CONTENTS_FILE, CONTENTS_REV, LOCAL  # noqa: E402

OUT = ROOT / "results" / "flipbudget" / "board_response_dates.json"
SHIFT = ROOT / "results" / "flipbudget" / "board_verifier_shift.json"
SNAPSHOT_DATE = "2025-01-31"  # revision date of the pre-re-scoring results (fb_board_verifier_shift.RESULTS_REV)
N_SAMPLE, SEED = 50, 20261004


def board_models():
    """The models with both scores, rebuilt from the cached snapshot with the shift script's rules."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    root = LOCAL / "results_2025-01-31"
    latest = {}
    for p in root.rglob("*.json"):
        rel = p.relative_to(root).as_posix()
        model = "/".join(rel.split("/")[:2])
        if model not in latest or rel > latest[model]:
            latest[model] = rel
    old = set()
    for model, rel in latest.items():
        d = json.load(open(root / rel, encoding="utf-8"))
        v = (d.get("results", {}).get("leaderboard_math_hard") or {}).get("exact_match,none")
        if isinstance(v, (int, float)):
            old.add(model)
    df = pd.read_parquet(hf_hub_download(CONTENTS, CONTENTS_FILE, repo_type="dataset", revision=CONTENTS_REV))
    new = set(df.dropna(subset=["MATH Lvl 5 Raw"])["fullname"])
    return sorted(old & new)


def check(api, model):
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    repo = f"open-llm-leaderboard/{model.replace('/', '__')}-details"
    try:
        paths = [f.path for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, expand=False)
                 if "samples_leaderboard_math_" in f.path and "_hard_" in f.path]
        if not paths:
            return {"repo": repo, "status": "no MATH-Hard samples files"}
        dates = sorted(str(i.last_commit.date)[:10] for i in
                       api.get_paths_info(repo, paths, repo_type="dataset", expand=True))
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        return {"repo": repo, "status": f"unavailable: {type(e).__name__}"}
    return {"repo": repo, "status": "checked", "n_files": len(paths), "first": dates[0], "last": dates[-1],
            "all_on_or_before_snapshot": dates[-1] <= SNAPSHOT_DATE}


def main():
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    both = board_models()
    expected = json.load(open(SHIFT, encoding="utf-8"))["n_models_both"]
    assert len(both) == expected, f"rebuilt {len(both)} models, the shift analysis has {expected}"
    sample = random.Random(SEED).sample(both, N_SAMPLE)
    api = HfApi()
    per_model = {m: check(api, m) for m in sample}
    checked = [v for v in per_model.values() if v["status"] == "checked"]
    out = {"snapshot_date": SNAPSHOT_DATE, "seed": SEED, "n_board": len(both), "n_sampled": N_SAMPLE,
           "n_checked": len(checked),
           "n_unchanged": sum(v["all_on_or_before_snapshot"] for v in checked),
           "n_unavailable": N_SAMPLE - len(checked), "per_model": per_model}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print({k: v for k, v in out.items() if k != "per_model"})
    for m, v in per_model.items():
        if v["status"] != "checked" or not v["all_on_or_before_snapshot"]:
            print(m, v)


if __name__ == "__main__":
    main()
