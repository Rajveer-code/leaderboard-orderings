"""The historical timepoints behind the board counts, from the Hugging Face commit history.

The paper uses three board counts that belong to different moments: the models the maintainers
re-scored in February 2025 (their post), the models with a score in the leaderboard's results
dataset at its last revision before that re-scoring, and the models in the final public table.
This records the pinned revisions and their commit dates so every count can name its moment.

Run: python scripts/fb_board_timepoints.py
Out: results/flipbudget/board_timepoints.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flipbudget" / "board_timepoints.json"
PINNED = {  # the revisions scripts/fb_board_verifier_shift.py and fb_board_position.py read
    "results_before_rescoring": ("open-llm-leaderboard/results", "c6b22601e39f8288b3852cf6421291fed2250050"),
    "final_public_table": ("open-llm-leaderboard/contents", "9c09a7cae43334062a82cb164f2ef255013dafa2"),
}


def main():
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    api = HfApi()
    out = {"retrieved": datetime.now(timezone.utc).date().isoformat(), "revisions": {}}
    for key, (repo, rev) in PINNED.items():
        commits = api.list_repo_commits(repo, repo_type="dataset")
        match = [c for c in commits if c.commit_id == rev]
        if not match:
            raise SystemExit(f"{repo}@{rev} not in the commit history")
        out["revisions"][key] = {"repo": repo, "revision": rev, "date": match[0].created_at.date().isoformat(),
                                 "is_latest_commit": commits[0].commit_id == rev}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
