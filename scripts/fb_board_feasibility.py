"""T10 -- can the whole board be re-scored? A metadata-only feasibility spike.

For a seeded random sample of the leaderboard's per-model detail datasets (catalogue from the
earlier study's selection record), queries the Hugging Face API for: whether the dataset still
exists, whether it is gated (gated datasets need a per-dataset terms acceptance that only the
account holder can give), whether it holds MATH-Hard sample files, and their total size. Nothing
is downloaded. The projection to the full catalogue is a proportion with a Wilson interval.

Run: python scripts/fb_board_feasibility.py   Out: results/flipbudget/board_feasibility.json
"""
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KS = Path(os.environ.get("FB_COMPANION_REPO", ROOT.parent / "knowledgeshift"))
CATALOGUE = KS / "results" / "analysis" / "leaderboard_details_catalogue.json"
OUT = ROOT / "results" / "flipbudget" / "board_feasibility.json"
N_SAMPLE, SEED = 120, 20260926


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    api = HfApi()
    catalogue = json.load(open(CATALOGUE, encoding="utf-8"))
    sample = random.Random(SEED).sample(catalogue, N_SAMPLE)
    rows = []
    for entry in sample:
        rid, row = entry["id"], {"id": entry["id"]}
        t0 = time.time()
        try:
            info = api.dataset_info(rid, files_metadata=True)
            row["gated"] = info.gated
            math_files = [s for s in (info.siblings or []) if "samples_leaderboard_math_" in s.rfilename]
            subjects = {s.rfilename.split("samples_leaderboard_math_")[1].split("_hard")[0] for s in math_files}
            # latest file per subject is what the census uses; size of those only
            latest = {}
            for s in math_files:
                subj = s.rfilename.split("samples_leaderboard_math_")[1].split("_hard")[0]
                if subj not in latest or s.rfilename > latest[subj].rfilename:
                    latest[subj] = s
            row["math_subjects"] = len(subjects)
            row["math_bytes_latest"] = sum((s.size or 0) for s in latest.values())
        except Exception as e:  # missing, disabled, or access-restricted
            row["error"] = type(e).__name__
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
    n = len(rows)
    ok = [r for r in rows if "error" not in r]
    open_ = [r for r in ok if not r.get("gated")]
    complete = [r for r in open_ if r.get("math_subjects") == 7]
    mb = [r["math_bytes_latest"] / 1e6 for r in complete]
    share = len(complete) / n
    lo, hi = wilson(len(complete), n)
    out = {
        "catalogue_size": len(catalogue), "sample_size": n, "seed": SEED,
        "reachable": len(ok), "not_gated": len(open_), "not_gated_with_all_7_math_subjects": len(complete),
        "gated_values_seen": sorted({str(r.get("gated")) for r in ok}),
        "share_usable": share, "share_usable_wilson95": [lo, hi],
        "projected_usable_models": [round(lo * len(catalogue)), round(share * len(catalogue)),
                                    round(hi * len(catalogue))],
        "math_mb_per_usable_model_median": sorted(mb)[len(mb) // 2] if mb else None,
        "projected_download_gb": round(share * len(catalogue) * (sum(mb) / len(mb) if mb else 0) / 1e3, 1),
        "rows": rows,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
