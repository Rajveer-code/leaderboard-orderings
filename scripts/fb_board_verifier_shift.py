"""T20 (reduced) -- is the verifier dependence visible at the scale of the whole board?

Per-item re-scoring of every model is too slow on this machine (decision D3's stop rule). The
leaderboard's own records answer the question more cheaply. Its public results dataset, pinned
to its last revision before the February 2025 re-scoring, holds every model's MATH-Hard score
under the 2024 verifier; its public contents table, pinned to its final revision, holds the
score displayed after the re-scoring, under Math-Verify. The model outputs did not change
between the two, so any reordering between the two scores is due to the verifier alone.

Reported: how many models have both scores, rank agreement between the two (Kendall's tau-b)
and the share of pairwise orderings that reverse, for the whole board and for the 27-model
roster, plus the roster's position in the distribution of score shifts.

Run: python scripts/fb_board_verifier_shift.py
Out: results/flipbudget/board_verifier_shift.json, manuscript/tmlr/figures/f7_board_verifier_shift.{pdf,png}
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flipbudget" / "board_verifier_shift.json"
FIG = ROOT / "manuscript" / "tmlr" / "figures" / "f7_board_verifier_shift"
ROSTER = ROOT / "results" / "flipbudget" / "e4_mathhard_per_model.json"
RESULTS, RESULTS_REV = "open-llm-leaderboard/results", "c6b22601e39f8288b3852cf6421291fed2250050"  # 2025-01-31
CONTENTS, CONTENTS_REV = "open-llm-leaderboard/contents", "9c09a7cae43334062a82cb164f2ef255013dafa2"  # final
CONTENTS_FILE = "data/train-00000-of-00001.parquet"
# plain files, no cache symlinks: Windows without developer mode cannot create them in parallel
LOCAL = Path(os.environ.get("FB_HF_LOCAL", Path.home() / ".cache" / "flipbudget_board_snapshot"))


def discordant_share(a, b):
    """Share of pairs ordered strictly in opposite directions by a and b, among pairs strictly
    ordered by both (exact, O(n^2) in memory-bounded blocks)."""
    a, b = np.asarray(a), np.asarray(b)
    disc = both = 0
    for i in range(0, len(a), 512):
        da = np.sign(a[i:i + 512, None] - a[None, :])
        db = np.sign(b[i:i + 512, None] - b[None, :])
        strict = (da != 0) & (db != 0)
        disc += int(((da * db) < 0).sum())
        both += int(strict.sum())
    return disc / both if both else float("nan")


def pair_counts(a, b):
    """Unordered pairs {i, j}: total, tied under a, tied under b, strictly ordered by both, and
    reversed ((a_i - a_j)(b_i - b_j) < 0). discordant_share is reversed / strictly_ordered_by_both."""
    a, b = np.asarray(a), np.asarray(b)
    tied_a = tied_b = strict = rev = 0
    for i in range(0, len(a), 512):
        da = np.sign(a[i:i + 512, None] - a[None, :])
        db = np.sign(b[i:i + 512, None] - b[None, :])
        upper = np.arange(i, i + len(da))[:, None] < np.arange(len(a))[None, :]  # each pair once
        tied_a += int(((da == 0) & upper).sum())
        tied_b += int(((db == 0) & upper).sum())
        strict += int(((da != 0) & (db != 0) & upper).sum())
        rev += int(((da * db < 0) & upper).sum())
    n = len(a)
    return {"pairs": n * (n - 1) // 2, "tied_2024": tied_a, "tied_rescored": tied_b,
            "strictly_ordered_by_both": strict, "reversed": rev,
            "reversed_share_of_all_pairs": rev / (n * (n - 1) // 2) if n > 1 else float("nan")}


def main():
    import pandas as pd
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi, hf_hub_download
    from scipy.stats import kendalltau
    api = HfApi()
    tree = [f.path for f in api.list_repo_tree(RESULTS, repo_type="dataset", revision=RESULTS_REV,
                                               recursive=True, expand=False) if f.path.endswith(".json")]
    latest = {}
    for p in tree:
        model = "/".join(p.split("/")[:2])
        if model not in latest or p > latest[model]:  # ISO timestamps sort lexically
            latest[model] = p

    import time
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    def old_score(item):
        model, path = item
        local = LOCAL / "results_2025-01-31" / path
        if not local.exists():  # files already fetched are read without touching the network
            for attempt in range(8):
                try:
                    hf_hub_download(RESULTS, path, repo_type="dataset", revision=RESULTS_REV,
                                    local_dir=LOCAL / "results_2025-01-31")
                    break
                except (HfHubHTTPError, LocalEntryNotFoundError) as e:
                    # the hub allows 5,000 requests per 5 minutes; a rate-limited metadata call
                    # surfaces as LocalEntryNotFoundError, so both are retried after a pause
                    if attempt == 7 or (isinstance(e, HfHubHTTPError) and "429" not in str(e)):
                        raise
                    time.sleep(75)
        d = json.load(open(local, encoding="utf-8"))
        v = (d.get("results", {}).get("leaderboard_math_hard") or {}).get("exact_match,none")
        return model, v

    with ThreadPoolExecutor(max_workers=4) as ex:
        old = {m: v for m, v in ex.map(old_score, sorted(latest.items())) if isinstance(v, (int, float))}

    df = pd.read_parquet(hf_hub_download(CONTENTS, CONTENTS_FILE, repo_type="dataset", revision=CONTENTS_REV))
    new = df.dropna(subset=["MATH Lvl 5 Raw"]).groupby("fullname")["MATH Lvl 5 Raw"].max().to_dict()
    both = sorted(set(old) & set(new))
    a = np.array([old[m] for m in both])
    b = np.array([new[m] for m in both])
    roster = sorted(json.load(open(ROSTER, encoding="utf-8"))["per_model"])
    rin = [m for m in roster if m in set(both)]
    ra = np.array([old[m] for m in rin])
    rb = np.array([new[m] for m in rin])
    shift = b - a
    rshift = {m: float(new[m] - old[m]) for m in rin}
    out = {
        "sources": {"old": {"repo": RESULTS, "revision": RESULTS_REV, "field": "leaderboard_math_hard.exact_match,none",
                            "file_rule": "latest results file per model folder"},
                    "new": {"repo": CONTENTS, "revision": CONTENTS_REV, "field": "MATH Lvl 5 Raw",
                            "rule": "max over precision variants per fullname"}},
        "n_models_old": len(old), "n_models_new": len(new), "n_models_both": len(both),
        "board": {"kendall_tau_b": float(kendalltau(a, b).statistic), "discordant_pair_share": discordant_share(a, b),
                  "median_shift": float(np.median(shift)), "share_score_rose": float((shift > 0).mean()),
                  "pair_counts": pair_counts(a, b)},
        "roster": {"n": len(rin), "kendall_tau_b": float(kendalltau(ra, rb).statistic),
                   "discordant_pair_share": discordant_share(ra, rb), "pair_counts": pair_counts(ra, rb),
                   "median_shift": float(np.median(list(rshift.values()))),
                   "median_shift_percentile_on_board": float((shift < np.median(list(rshift.values()))).mean())},
        "roster_shift": rshift,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("roster_shift", "sources")}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(a, b, ".", color="#BBBBBB", ms=2.5, label=f"{len(both)} models with both scores")
    ax.plot(ra, rb, "o", color="#0072B2", ms=4, label=f"the {len(rin)} roster models")
    lim = [0, max(a.max(), b.max()) * 1.03]
    ax.plot(lim, lim, color="#666666", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MATH-Hard accuracy, 2024 verifier (before re-scoring)")
    ax.set_ylabel("MATH-Hard accuracy, Math-Verify (after re-scoring)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", bbox_inches="tight",  # no creation date (time-zone offset)
                    **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
    print("wrote", OUT, "and", FIG)


if __name__ == "__main__":
    main()
