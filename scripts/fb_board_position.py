"""T20 (reduced) -- where the 27-model roster sits on the full board.

The full re-score was stopped by the rule attached to decision D3: scoring one model with the
three verifiers takes minutes on this machine, so ~4,000 models would take weeks. The first of
D3's three questions (why these 27) is answered by scripts/fb_roster_provenance.py. This script
answers the second: where the roster sits in the whole board's distribution, using the
leaderboard's public `contents` table, pinned to its final revision. It holds each model's
displayed MATH-Hard score -- the Math-Verify score after the February 2025 re-scoring -- and its
parameter count.

Run: python scripts/fb_board_position.py
Out: results/flipbudget/board_position.json, manuscript/tmlr/figures/f6_board_position.{pdf,png}
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flipbudget" / "board_position.json"
FIG = ROOT / "manuscript" / "tmlr" / "figures" / "f6_board_position"
ROSTER = ROOT / "results" / "flipbudget" / "e4_mathhard_per_model.json"
REPO, REVISION, FILE = "open-llm-leaderboard/contents", "9c09a7cae43334062a82cb164f2ef255013dafa2", \
    "data/train-00000-of-00001.parquet"


def main():
    import pandas as pd
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path)
    # "fullname" is the hub id; "MATH Lvl 5 Raw" is the displayed accuracy as a fraction
    name_col, math_col = "fullname", "MATH Lvl 5 Raw"
    params_col = "#Params (B)"
    df = df[[name_col, math_col] + ([params_col] if params_col else [])].dropna(subset=[math_col])
    # one row per model: some models were evaluated more than once (precision variants); keep the max
    best = df.groupby(name_col, as_index=False)[math_col].max()
    roster = sorted(json.load(open(ROSTER, encoding="utf-8"))["per_model"])
    scores = best.set_index(name_col)[math_col]
    found = [m for m in roster if m in scores.index]
    board = np.sort(scores.values)
    pct = {m: float((board < scores[m]).mean()) for m in found}  # share of board strictly below
    out = {
        "source": {"repo": REPO, "revision": REVISION, "file": FILE, "name_column": name_col,
                   "math_column": math_col},
        "n_board_models_with_math": int(len(board)),
        "n_roster": len(roster), "n_roster_found": len(found),
        "roster_not_found": sorted(set(roster) - set(found)),
        "board_math_quantiles": {q: float(np.quantile(board, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "roster_percentile_median": float(np.median(list(pct.values()))),
        "roster_percentile_min": float(min(pct.values())), "roster_percentile_max": float(max(pct.values())),
        "roster_in_top_quartile": int(sum(p >= 0.75 for p in pct.values())),
        "roster_in_bottom_quartile": int(sum(p < 0.25 for p in pct.values())),
        "roster_percentiles": pct,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "roster_percentiles"}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.hist(board, bins=60, color="#BBBBBB", edgecolor="white", lw=0.3,
            label=f"all {len(board)} models with a MATH-Hard score")
    ymax = ax.get_ylim()[1]
    for m in found:
        ax.plot([scores[m]] * 2, [0, ymax * 0.18], color="#0072B2", lw=0.9)
    ax.plot([], [], color="#0072B2", lw=0.9, label=f"the {len(found)} roster models")
    ax.set_xlabel("displayed MATH-Hard accuracy (after the February 2025 re-scoring)")
    ax.set_ylabel("models")
    ax.legend(frameon=False, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", bbox_inches="tight",  # no creation date (time-zone offset)
                    **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))
    print("wrote", OUT, "and", FIG)


if __name__ == "__main__":
    main()
