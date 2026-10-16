"""G1.2 + G1.4 -- K=3 census of MATH-Hard verdicts and the prespecified coverage table.

One row per (model, item) for the 27-model roster: the published per-item `exact_match`
(em), the re-executed recorded leaderboard scorer (em_rec), this project's boxed verifier
in both semantics (sb_string = historical Windows behaviour, sb_sympy = faithful), and
Math-Verify (mv). Rules: G1_DATA_RULES.md (committed before this script was run).

Responses: latest-timestamp samples file per (model, subject) from knowledgeshift's
cache_l1; a file absent there is downloaded from the same HF details repo (revision
recorded). Scoring runs in a pebble process pool; each verifier call is one task with a
60 s kill deadline (verifiers.py applies the original 1 s / 5 s deadlines by timing).

Run: python scripts/fb_census_k3.py            (~15 min, 18 workers)
Out: results/flipbudget/census_k3.json.gz, results/flipbudget/coverage_table.json
"""
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from pebble import ProcessPool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flipbudget import verifiers  # noqa: E402

CACHE = Path(os.environ.get("FB_CACHE_L1", ROOT.parent / "knowledgeshift" / "results" / "cache_l1"))
ROSTER = ROOT / "results" / "flipbudget" / "e4_mathhard_per_model.json"
KEY = ROOT / "results" / "analysis" / "mathhard_labelling_key.json"
OUT = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
COVERAGE = ROOT / "results" / "flipbudget" / "coverage_table.json"
SUBJECTS = ["algebra", "counting_and_prob", "geometry", "intermediate_algebra",
            "num_theory", "prealgebra", "precalculus"]
N_ITEMS = 1324
MAX_MISSING = 0.05
KILL_SECONDS = 60
WORKERS = 18


def latest_files(model):
    """{subject: (path, source)}; HF download for subjects missing from the cache."""
    d = CACHE / f"open-llm-leaderboard_{model.replace('/', '__')}-details"
    local = sorted(os.listdir(d)) if d.is_dir() else []
    out = {}
    for s in SUBJECTS:
        c = [f for f in local if f.startswith(f"samples_leaderboard_math_{s}_hard_")]
        if c:
            out[s] = (d / c[-1], "cache_l1")
    missing = [s for s in SUBJECTS if s not in out]
    if missing:
        import truststore
        truststore.inject_into_ssl()
        from huggingface_hub import HfApi, hf_hub_download
        repo = f"open-llm-leaderboard/{model.replace('/', '__')}-details"
        api = HfApi()
        rev = api.dataset_info(repo).sha
        names = api.list_repo_files(repo, repo_type="dataset", revision=rev)
        for s in missing:
            c = sorted(n for n in names if f"samples_leaderboard_math_{s}_hard_" in n)
            if c:
                out[s] = (Path(hf_hub_download(repo, c[-1], repo_type="dataset", revision=rev)),
                          f"hf:{repo}@{rev}")
    return out


def load_rows(models):
    rows, sources = [], {}
    for m in models:
        files = latest_files(m)
        sources[m] = {s: [p.name, src] for s, (p, src) in files.items()}
        for s, (p, _) in files.items():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                doc = r.get("doc") or {}
                resp = r["resps"][0][0] if r.get("resps") and isinstance(r["resps"][0], list) else None
                filt = r.get("filtered_resps") or [None]
                filt = filt[0][0] if isinstance(filt[0], list) else filt[0]
                rows.append({"model": m, "item_id": r.get("doc_hash"), "subject": s,
                             "gold": doc.get("answer"), "solution": doc.get("solution"), "problem": doc.get("problem"),
                             "response": resp, "filtered_equals_resps": filt == resp,
                             "em": r.get("exact_match")})
    return rows, sources


def score(rows):
    """Adds em_rec / sb_* / mv fields. One pool task per (row, verifier)."""
    jobs = []
    with ProcessPool(max_workers=WORKERS) as pool:
        for i, r in enumerate(rows):
            if not r["scored"]:
                continue
            jobs.append((i, "em_rec", pool.schedule(verifiers.em_recorded, (r["response"], r["gold"]),
                                                    timeout=KILL_SECONDS)))
            jobs.append((i, "sb", pool.schedule(verifiers.sb, (r["response"], r["gold"]), timeout=KILL_SECONDS)))
            jobs.append((i, "mv", pool.schedule(verifiers.mv, (r["response"], r["solution"]), timeout=KILL_SECONDS)))
        t0 = time.time()
        for n, (i, name, fut) in enumerate(jobs, 1):
            try:
                res, killed = fut.result(), False
            except FutureTimeout:
                res, killed = {"verdict": 0, "timed_out": True}, True
            r = rows[i]
            r[name] = res["verdict"]
            r[f"{name}_timeout"] = bool(res.get("timed_out"))
            r[f"{name}_killed"] = killed
            if name == "sb":
                r["sb_found"] = res.get("found", False)
            if name == "mv":
                r["mv_pred_parsed"] = res.get("pred_parsed", False)
                r["mv_gold_parsed"] = res.get("gold_parsed", False)
            if n % 10000 == 0:
                print(f"  {n}/{len(jobs)} tasks, {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        if r["scored"]:  # sb_string: exact match after normalisation only (no is_equiv)
            r["sb_string"] = int(verifiers.sb(r["response"], r["gold"], mode="string")["verdict"])
            r["sb_sympy"] = r.pop("sb")


def coverage(rows, models):
    by = defaultdict(set)
    for r in rows:
        if r["scored"]:
            by[r["model"]].add(r["item_id"])
    table = {}
    for m in models:
        n_scored = len(by[m])
        mr = 1 - n_scored / N_ITEMS
        table[m] = {"n_total": N_ITEMS, "n_scored": n_scored, "n_missing": N_ITEMS - n_scored,
                    "missing_rate": mr, "dropped": mr > MAX_MISSING}
    kept = [m for m in models if not table[m]["dropped"]]
    common = set.intersection(*(by[m] for m in kept)) if kept else set()
    for m in models:
        table[m]["common_item_coverage"] = len(common & by[m]) / N_ITEMS
    return table, kept, sorted(common)


def main():
    models = sorted(json.load(open(ROSTER, encoding="utf-8"))["per_model"])
    labeled = {(k["model"], k["item_id"]) for k in json.load(open(KEY, encoding="utf-8"))}
    rows, sources = load_rows(models)
    for r in rows:
        r["scored"] = bool(isinstance(r["response"], str) and r["gold"] and r["item_id"] and r["em"] in (0, 1))
        r["already_labeled"] = (r["model"], r["item_id"]) in labeled
    dup = [k for k, c in Counter((r["model"], r["item_id"]) for r in rows).items() if c > 1]
    assert not dup, f"duplicate (model, item) rows: {dup[:5]}"
    print(f"{len(models)} models, {len(rows)} rows loaded, {sum(r['scored'] for r in rows)} scored")
    table, kept, common = coverage(rows, models)
    json.dump({"rule": "G1_DATA_RULES.md", "n_items": N_ITEMS, "max_missing_rate": MAX_MISSING,
               "models_kept": kept, "models_dropped": [m for m in models if table[m]["dropped"]],
               "n_common_items": len(common), "per_model": table, "sources": sources},
              open(COVERAGE, "w", encoding="utf-8"), indent=1)
    print(f"coverage: kept {len(kept)}/{len(models)}, common items {len(common)}")
    score(rows)
    common_set = set(common)
    keep = ("model", "item_id", "subject", "em", "em_rec", "sb_string", "sb_sympy", "mv", "sb_found",
            "mv_pred_parsed", "mv_gold_parsed", "em_rec_timeout", "sb_timeout", "mv_timeout",
            "em_rec_killed", "sb_killed", "mv_killed", "filtered_equals_resps", "already_labeled", "scored")
    out = [{k: r.get(k) for k in keep} | {"common": r["item_id"] in common_set} for r in rows]
    json.dump({"n_rows": len(out), "models": models, "rows": out}, gzip.open(OUT, "wt", encoding="utf-8"),
              separators=(",", ":"))
    print("wrote", OUT, "and", COVERAGE)


if __name__ == "__main__":
    main()
