"""Is each post-re-scoring score on the board actually a re-graded score?

The board-wide reversal statistic (fb_board_verifier_shift.py) pairs each model's 2024 score with
its score in the final public table, taken as the maximum over the model's precision rows. The
February 2025 re-grade rewrote results files in the per-model details repositories (task version
1.0 -> 3.0, roster check in fb_verifier_provenance.py). A public audit (leaderboard discussion
#1171) reports that some displayed rows were never re-graded. This script tests that for every
model in the comparison, without relying on that audit's attribution:

  1. list the results files in the model's details repository and the date of the last commit to
     each (a file last committed on the re-grade day is a re-graded file);
  2. read the MATH-Hard aggregate and the task version of every re-graded file;
  3. call the model's displayed score attributable when it equals the aggregate of a re-graded
     file of the same model (to 1e-9), and record every displayed precision row that matches none.

The reversal statistic is then recomputed on the attributable models only.

Run: python scripts/fb_board_row_provenance.py     Out: results/flipbudget/board_row_provenance.json
Network: about two Hub requests per model plus one download per re-graded file; resumable.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fb_board_verifier_shift import CONTENTS, CONTENTS_FILE, CONTENTS_REV, LOCAL, discordant_share, pair_counts  # noqa: E402

OUT = ROOT / "results" / "flipbudget" / "board_row_provenance.json"
SHIFT = ROOT / "results" / "flipbudget" / "board_verifier_shift.json"
CACHE = LOCAL / "row_provenance_cache.json"
REGRADE_DAY = "2025-02-13"
TOL = 1e-9
# 4 workers ran without a single 429; 12 drew hundreds (the API endpoints allow less than downloads)
WORKERS = int(os.environ.get("FB_WORKERS", "4"))


def snapshot_scores():
    """Jan-31 score per model, with the shift script's file rule (latest results file per folder)."""
    root = LOCAL / "results_2025-01-31"
    latest = {}
    for p in root.rglob("*.json"):
        rel = p.relative_to(root).as_posix()
        model = "/".join(rel.split("/")[:2])
        if model not in latest or rel > latest[model]:
            latest[model] = rel
    old = {}
    for model, rel in latest.items():
        d = json.load(open(root / rel, encoding="utf-8"))
        v = (d.get("results", {}).get("leaderboard_math_hard") or {}).get("exact_match,none")
        if isinstance(v, (int, float)):
            old[model] = float(v)
    return old


def displayed_rows():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download(CONTENTS, CONTENTS_FILE, repo_type="dataset", revision=CONTENTS_REV))
    df = df.dropna(subset=["MATH Lvl 5 Raw"])
    rows = {}
    for name, precision, value in df[["fullname", "Precision", "MATH Lvl 5 Raw"]].itertuples(index=False, name=None):
        rows.setdefault(name, []).append({"precision": precision, "value": float(value)})
    return rows


def with_retry(fn, *a, **k):
    # a rate-limited call can surface as HfHubHTTPError or, for downloads, LocalEntryNotFoundError
    for attempt in range(12):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001 -- only 429s are retried; everything else is re-raised
            if ("429" not in str(e) and "Too Many" not in str(e)) or attempt == 11:
                raise
            time.sleep(75)


def probe(api, model):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, HFValidationError, RepositoryNotFoundError
    repo = f"open-llm-leaderboard/{model.replace('/', '__')}-details"
    prefix = model.replace("/", "__")
    try:
        # list_repo_tree is a lazy generator: materialise it inside the retry so paging 429s are retried
        tree = with_retry(lambda: list(api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset")))
        names = [f.path for f in tree if f.path.split("/")[-1].startswith("results_") and f.path.endswith(".json")]
        if not names:
            return {"status": "no results files"}
        info = with_retry(api.get_paths_info, repo, names, repo_type="dataset", expand=True)
        commits = None
        files = []
        for i in info:
            day = str(i.last_commit.date)[:10]
            entry = {"file": i.path.split("/")[-1], "last_commit_day": day, "regraded": day == REGRADE_DAY}
            if entry["regraded"]:
                d = json.load(open(with_retry(hf_hub_download, repo, i.path, repo_type="dataset"), encoding="utf-8"))
                entry["math_hard"] = (d.get("results", {}).get("leaderboard_math_hard") or {}).get("exact_match,none")
                entry["dtype"] = (d.get("config") or {}).get("model_dtype")
                entry["versions"] = sorted({str(v) for k, v in (d.get("versions") or {}).items()
                                            if k.startswith("leaderboard_math_") and k.endswith("_hard")})
                # the same file just before the re-grade: same run, same responses, old grader
                if commits is None:
                    commits = [c.commit_id for c in with_retry(api.list_repo_commits, repo, repo_type="dataset")]
                entry["math_hard_before"] = None
                if i.last_commit.oid in commits:
                    for rev in commits[commits.index(i.last_commit.oid) + 1:]:
                        try:
                            b = json.load(open(with_retry(hf_hub_download, repo, i.path, repo_type="dataset",
                                                          revision=rev), encoding="utf-8"))
                        except EntryNotFoundError:
                            continue
                        entry["math_hard_before"] = (b.get("results", {}).get("leaderboard_math_hard") or {}).get(
                            "exact_match,none")
                        break
            files.append(entry)
        return {"status": "checked", "files": files}
    except (RepositoryNotFoundError, EntryNotFoundError, HFValidationError) as e:
        # HFValidationError: the details repo name would exceed the Hub's 96-character limit
        return {"status": f"unavailable: {type(e).__name__}"}
    except (HfHubHTTPError, OSError, ValueError) as e:  # recorded as an error and retried on the next run
        return {"status": f"error: {type(e).__name__}: {str(e)[:80]}"}


def share(models, old, new):
    a = np.array([old[m] for m in models])
    b = np.array([new[m] for m in models])
    pc = pair_counts(a, b)
    return {"models": len(models), "reversed_pairs": pc["reversed"], "strict_pairs": pc["strictly_ordered_by_both"],
            "reversed_share": discordant_share(a, b)}


def main():
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    old, rows = snapshot_scores(), displayed_rows()
    both = sorted(set(old) & set(rows))
    expected = json.load(open(SHIFT, encoding="utf-8"))["n_models_both"]
    assert len(both) == expected, (len(both), expected)
    new = {m: max(r["value"] for r in rows[m]) for m in both}  # the shift script's rule

    cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    todo = [m for m in both if m not in cache or cache[m]["status"].startswith("error")]
    api = HfApi()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n, (m, res) in enumerate(zip(todo, ex.map(lambda m: probe(api, m), todo)), 1):
            cache[m] = res
            if n % 100 == 0:
                json.dump(cache, open(CACHE, "w", encoding="utf-8"))
                print(f"{n}/{len(todo)}", flush=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))

    per_model, stale_rows = {}, []
    for m in both:
        c = cache[m]
        reg = [f["math_hard"] for f in c.get("files", []) if f.get("regraded") and isinstance(f.get("math_hard"), (int, float))]
        if c["status"] != "checked":
            cls = "unavailable"
        elif not reg:
            cls = "no re-graded file"
        elif any(abs(new[m] - v) <= TOL for v in reg):
            cls = "attributable"
        else:
            cls = "displayed score matches no re-graded file"
        for r in rows[m]:
            if c["status"] == "checked" and not any(abs(r["value"] - v) <= TOL for v in reg):
                stale_rows.append({"model": m, "precision": r["precision"], "value": r["value"]})
        per_model[m] = {"class": cls, "old": old[m], "displayed": new[m], "n_rows": len(rows[m]),
                        "regraded_values": reg, "probe": c}
    classes = {}
    for v in per_model.values():
        classes[v["class"]] = classes.get(v["class"], 0) + 1
    attributable = [m for m in both if per_model[m]["class"] == "attributable"]
    # same-run pairs: the re-graded file whose aggregate is displayed, before and after its rewrite
    before, after = {}, {}
    for m in attributable:
        for f in per_model[m]["probe"]["files"]:
            if f.get("regraded") and isinstance(f.get("math_hard_before"), (int, float)) \
                    and abs(f["math_hard"] - new[m]) <= TOL:
                before[m], after[m] = float(f["math_hard_before"]), float(f["math_hard"])
                break
    same_run = sorted(before)
    snapshot_is_same_run = sum(abs(old[m] - before[m]) <= TOL for m in same_run)
    out = {"regrade_day": REGRADE_DAY, "n_models": len(both), "classes": classes,
           "same_run_pairs": len(same_run), "snapshot_score_equals_same_run_before": snapshot_is_same_run,
           "displayed_rows_matching_no_regraded_file": len(stale_rows),
           "versions_seen": sorted({v for p in per_model.values() for f in p["probe"].get("files", [])
                                    for v in f.get("versions", [])}),
           "reversal": {"all": share(both, old, new), "attributable_only": share(attributable, old, new),
                        "same_run": share(same_run, before, after)},
           "stale_rows": stale_rows, "per_model": per_model}
    out["same_run_detail"] = same_run_detail(same_run, before, after)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_model", "stale_rows")}, indent=1))
    plot_same_run(same_run, before, after, out["same_run_detail"]["roster_models"])


ROSTER = ROOT / "results" / "flipbudget" / "e4_mathhard_per_model.json"
FIG = ROOT / "manuscript" / "tmlr" / "figures" / "f8_board_same_run"


def same_run_detail(models, before, after):
    """Kendall tau-b and pair counts for the same-run pairs, on the board and on the roster."""
    from scipy.stats import kendalltau
    roster = [m for m in sorted(json.load(open(ROSTER, encoding="utf-8"))["per_model"]) if m in set(models)]
    res = {"roster_models": roster}
    for name, ms in (("board", models), ("roster", roster)):
        a = np.array([before[m] for m in ms])
        b = np.array([after[m] for m in ms])
        res[name] = {"n": len(ms), "kendall_tau_b": float(kendalltau(a, b).statistic),
                     "discordant_pair_share": discordant_share(a, b), "pair_counts": pair_counts(a, b),
                     "share_score_rose": float((b > a).mean())}
    return res


def plot_same_run(models, before, after, roster):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    a = np.array([before[m] for m in models])
    b = np.array([after[m] for m in models])
    ra = np.array([before[m] for m in roster])
    rb = np.array([after[m] for m in roster])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(a, b, ".", color="#BBBBBB", ms=2.5, label=f"{len(models)} models, same run before and after")
    ax.plot(ra, rb, "o", color="#0072B2", ms=4, label=f"the {len(roster)} roster models among them")
    lim = [0, max(a.max(), b.max()) * 1.03]
    ax.plot(lim, lim, color="#666666", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MATH-Hard accuracy, 2024 verifier (before the re-grade)")
    ax.set_ylabel("MATH-Hard accuracy, Math-Verify (after the re-grade)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", bbox_inches="tight",  # no creation date (time-zone offset)
                    **({"metadata": {"CreationDate": None}} if ext == "pdf" else {}))


if __name__ == "__main__":
    main()
