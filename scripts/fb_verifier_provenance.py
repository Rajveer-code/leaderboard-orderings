"""G1.1 -- provenance of the leaderboard's per-item `exact_match`, by evidence.

For every roster model, from the HF details repo and the K=3 census:
  1. commit history: last commit of each math samples file and of the matching results JSON;
  2. the results JSON as originally uploaded (parent of its last commit) and as now published;
  3. the recorded `process_results` source, compared byte-for-byte with upstream lm-eval
     3c8db1bb7b, plus recorded git_hash, task versions, generation kwargs;
  4. re-execution: recorded scorer (em_rec) vs the published per-item field (em), per item;
     mean(em) vs the original aggregate; mean(mv) vs the rescored (current) aggregate.
Agreement alone is never treated as provenance (plan v2, G1.1); it is reported next to the
documentary evidence.

Run after scripts/fb_census_k3.py:  python scripts/fb_verifier_provenance.py
Out: results/flipbudget/verifier_provenance.json
"""
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
COVERAGE = ROOT / "results" / "flipbudget" / "coverage_table.json"
UPSTREAM = ROOT / "provenance" / "verifier" / "lmeval_leaderboard_math_utils_3c8db1bb7b.py"
OUT = ROOT / "results" / "flipbudget" / "verifier_provenance.json"
TOL = 1e-9


def upstream_process_results():
    src = UPSTREAM.read_text(encoding="utf-8")
    return re.search(r"def process_results.*?\n(?=\n\ndef )", src, re.S).group(0).strip()


def model_evidence(api, model, sample_files):
    repo = f"open-llm-leaderboard/{model.replace('/', '__')}-details"
    prefix = model.replace("/", "__") + "/"
    stamp = sample_files["algebra"][0].split("_hard_")[1].removesuffix(".json")
    res_path = f"{prefix}results_{stamp}.json"
    paths = [prefix + f for f, _ in sample_files.values()] + [res_path]
    info = {i.path: i for i in api.get_paths_info(repo, paths, repo_type="dataset", expand=True)}
    commits = [c.commit_id for c in api.list_repo_commits(repo, repo_type="dataset")]  # newest first
    last = info[res_path].last_commit
    def load(rev):
        return json.load(open(hf_hub_download(repo, res_path, repo_type="dataset", revision=rev), encoding="utf-8"))

    now, orig, parent = load(last.oid), None, None
    for rev in commits[commits.index(last.oid) + 1:]:  # nearest older revision holding the file
        try:
            orig, parent = load(rev), rev
            break
        except EntryNotFoundError:
            continue
    if orig is None:  # file created by its last commit: no pre-rescoring version exists
        return {"repo": repo, "results_file": res_path, "results_original_revision": None,
                "results_last_commit": {"id": last.oid, "date": str(last.date), "title": last.title},
                "samples_last_commit": {p.removeprefix(prefix): str(info[p].last_commit.date) for p in paths[:-1]},
                "git_hash": now.get("git_hash"), "aggregate_original": None,
                "aggregate_now": {k.removeprefix("leaderboard_math_").removesuffix("_hard"): v["exact_match,none"]
                                  for k, v in now["results"].items()
                                  if k.startswith("leaderboard_math_") and k.endswith("_hard")}}
    up = upstream_process_results()
    cfgs = {k: v for k, v in orig["configs"].items() if k.startswith("leaderboard_math_") and k.endswith("_hard")}
    cfgs_now = {k: v for k, v in now["configs"].items() if k in cfgs}

    def agg(d):
        return {k.removeprefix("leaderboard_math_").removesuffix("_hard"): v["exact_match,none"]
                for k, v in d["results"].items() if k in cfgs}

    return {
        "repo": repo, "results_file": res_path,
        "samples_last_commit": {p.removeprefix(prefix): str(info[p].last_commit.date) for p in paths[:-1]},
        "results_last_commit": {"id": last.oid, "date": str(last.date), "title": last.title},
        "results_original_revision": parent,
        "git_hash": orig.get("git_hash"), "git_hash_now": now.get("git_hash"),
        "transformers_version": orig.get("transformers_version"),
        "recorded_process_results_equals_upstream_3c8db1bb7b": all(
            c["process_results"].strip() == up for c in cfgs.values()),
        "config_unchanged_by_rescoring": all(cfgs_now[k] == cfgs[k] for k in cfgs),
        "task_version_original": sorted({orig["versions"][k] for k in cfgs}),
        "task_version_now": sorted({now["versions"][k] for k in cfgs}),
        "generation_kwargs": next(iter(cfgs.values()))["generation_kwargs"],
        "aggregate_original": agg(orig), "aggregate_now": agg(now),
    }


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    sources = json.load(open(COVERAGE, encoding="utf-8"))["sources"]
    rows = [r for r in census["rows"] if r["scored"]]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["model"]][r["subject"]].append(r)
    api = HfApi()
    per_model, summary = {}, defaultdict(int)
    for m in census["models"]:
        ev = model_evidence(api, m, sources[m])
        subj = {}
        for s, rs in by[m].items():
            n = len(rs)
            subj[s] = {"n": n, "mean_em": sum(r["em"] for r in rs) / n,
                       "mean_em_rec": sum(r["em_rec"] for r in rs) / n,
                       "mean_mv": sum(r["mv"] for r in rs) / n,
                       "em_vs_em_rec_disagree": sum(r["em"] != r["em_rec"] for r in rs)}
            subj[s]["em_matches_original_aggregate"] = (
                None if ev["aggregate_original"] is None else abs(subj[s]["mean_em"] - ev["aggregate_original"][s]) < TOL)
            subj[s]["mv_matches_current_aggregate"] = abs(subj[s]["mean_mv"] - ev["aggregate_now"][s]) < TOL
        ev["per_subject"] = subj
        per_model[m] = ev
        summary["models"] += 1
        summary["models_without_pre_rescoring_results"] += ev["aggregate_original"] is None
        summary["process_results_equals_upstream"] += ev.get("recorded_process_results_equals_upstream_3c8db1bb7b", False)
        summary["config_unchanged_by_rescoring"] += ev.get("config_unchanged_by_rescoring", False)
        summary["samples_never_modified_after_2024"] += all(d < "2025" for d in ev["samples_last_commit"].values())
        summary["subjects"] += len(subj)
        summary["subjects_em_equals_original_aggregate"] += sum(bool(v["em_matches_original_aggregate"]) for v in subj.values())
        summary["subjects_mv_equals_current_aggregate"] += sum(v["mv_matches_current_aggregate"] for v in subj.values())
        summary["items"] += sum(v["n"] for v in subj.values())
        summary["items_em_vs_em_rec_disagree"] += sum(v["em_vs_em_rec_disagree"] for v in subj.values())
        print(m, ev["git_hash"], ev["results_last_commit"]["date"][:10],
              {s: (v["em_vs_em_rec_disagree"], v["em_matches_original_aggregate"], v["mv_matches_current_aggregate"])
               for s, v in subj.items()}, flush=True)
    summary["git_hashes"] = sorted({v["git_hash"] for v in per_model.values()})
    summary["rescoring_commit_dates"] = sorted({v["results_last_commit"]["date"][:10] for v in per_model.values()})
    json.dump({"summary": dict(summary), "per_model": per_model}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(summary), indent=1))


if __name__ == "__main__":
    main()
