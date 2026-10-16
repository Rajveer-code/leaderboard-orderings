"""Pull the full per-model item roster from knowledgeshift's raw cache_l1, to unblock
T-C's item-level audit-expansion worklist (previously blocked -- see
AUDIT_EXPANSION_DESIGN.md's "T-C target" section).

READS from the companion project only. Never writes to it, never merges repos --
matches this project's existing snapshot-with-provenance convention
(results/analysis/*.json are already one-time snapshots pulled from knowledgeshift;
this is another one, just item-level instead of pre-aggregated).

Reuses knowledgeshift's OWN canonical stratum-derivation logic exactly as
scripts/62_mathhard_frame.py calls it -- same score_boxed comparator, same
credited/wrong_parsed/unparsed split, same latest-file-per-subject selection.
Not reimplemented: imported directly from knowledgeshift/scripts/19_l1_rescore_leaderboard.py,
so a bug fix there is not silently missed here.

PATHS block (per project convention -- one place to change if the companion repo moves):
"""
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

KNOWLEDGESHIFT_ROOT = Path(r"D:\Projects\knowledgeshift")
CACHE_L1 = KNOWLEDGESHIFT_ROOT / "results" / "cache_l1"
HARNESS_ROOT = KNOWLEDGESHIFT_ROOT / "vendor" / "lm-evaluation-harness"
L1_SCRIPT = KNOWLEDGESHIFT_ROOT / "scripts" / "19_l1_rescore_leaderboard.py"

SUBJECTS = ["algebra", "counting_and_prob", "geometry", "intermediate_algebra",
            "num_theory", "prealgebra", "precalculus"]


def _stub_optional_imports():
    # 19_l1_rescore_leaderboard.py imports `requests` at module level for its `fetch`
    # helper, which this script never calls (cache_l1 is already on disk). Stub it if
    # missing so the import doesn't fail on an unrelated, unused dependency.
    if "requests" not in sys.modules:
        try:
            import requests  # noqa: F401
        except ImportError:
            stub = types.ModuleType("requests")
            stub.get = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("requests.get stubbed -- fetch() should not be called here"))
            sys.modules["requests"] = stub
    if "datasets" not in sys.modules:
        stub = types.ModuleType("datasets")
        stub.Dataset = object
        sys.modules["datasets"] = stub


def load_l1():
    _stub_optional_imports()
    spec = importlib.util.spec_from_file_location("l1_rescore", L1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subject_of(filename: str) -> str | None:
    for s in SUBJECTS:
        if f"math_{s}_hard" in filename:
            return s
    return None


def decode_model_name(dirname: str) -> str:
    """'open-llm-leaderboard_01-ai__Yi-1.5-34B-Chat-details' -> '01-ai/Yi-1.5-34B-Chat'.
    Only the FIRST '__' is the org/model separator (some model names contain further
    underscores or hyphens but not a second '__' in this roster -- verified below by
    checking every decoded name against the known 27-model roster, not assumed)."""
    name = dirname
    if name.startswith("open-llm-leaderboard_"):
        name = name[len("open-llm-leaderboard_"):]
    if name.endswith("-details"):
        name = name[: -len("-details")]
    return name.replace("__", "/", 1)


def load_roster_models() -> set[str]:
    with open("results/flipbudget/e4_mathhard_per_model.json", encoding="utf-8") as f:
        return set(json.load(f)["per_model"].keys())


def build_population(l1, math, roster_models: set[str]):
    population = []
    matched_models = set()
    dirs = sorted(p for p in CACHE_L1.iterdir() if p.is_dir())
    for i, d in enumerate(dirs, 1):
        model = decode_model_name(d.name)
        if model not in roster_models:
            continue
        matched_models.add(model)
        files = l1.latest_math_files([p.name for p in d.glob("*.json")])
        for fname in files:
            subject = subject_of(fname)
            if subject is None:
                continue
            text_content = (d / fname).read_text(encoding="utf-8", errors="replace")
            for row in l1.rows_of(text_content):
                doc = row.get("doc") or {}
                gold = doc.get("answer")
                resps = row.get("resps") or []
                doc_hash = row.get("doc_hash")
                if not (gold and resps and doc_hash):
                    continue
                text = resps[0][0] if isinstance(resps[0], list) else resps[0]
                if not isinstance(text, str):
                    continue
                score, found = l1.score_boxed(math, text, gold)
                if not found:
                    stratum = "unparsed"
                elif score == 1:
                    stratum = "credited"
                else:
                    stratum = "wrong_parsed"
                population.append({"model": model, "item_id": doc_hash,
                                    "subject": subject, "stratum": stratum})
        print(f"  [{i}/{len(dirs)}] {model}: {len(population)} rows so far", flush=True)
    return population, matched_models


def load_audit_key():
    with open("results/analysis/mathhard_labelling_key.json", encoding="utf-8") as f:
        key = json.load(f)
    audited = {}
    for r in key:
        audited[(r["model"], r["item_id"])] = r["stratum"]
    return audited


if __name__ == "__main__":
    print("=" * 70)
    print("Pulling full per-model item roster from knowledgeshift/results/cache_l1")
    print("=" * 70)
    if not CACHE_L1.is_dir():
        raise SystemExit(f"cache_l1 not found at {CACHE_L1} -- companion repo path wrong?")

    roster_models = load_roster_models()
    print(f"Target roster: {len(roster_models)} models (from e4_mathhard_per_model.json)")

    l1 = load_l1()
    math = l1.load_math_utils(HARNESS_ROOT)
    print("Loaded knowledgeshift's canonical comparator (score_boxed) and helpers.\n")

    population, matched_models = build_population(l1, math, roster_models)
    print()
    missing = roster_models - matched_models
    print(f"Models matched in cache_l1: {len(matched_models)} of {len(roster_models)}")
    if missing:
        print(f"  MISSING from cache_l1 (cannot pull roster for these): {sorted(missing)}")
    else:
        print("  [OK] all 27 roster models found in cache_l1 -- no silent gaps")

    print(f"\nTotal population rows: {len(population)}")

    # ---- Item-sharing check: is every item really scored by (nearly) every model? ----
    by_item = defaultdict(set)
    for r in population:
        by_item[r["item_id"]].add(r["model"])
    n_items = len(by_item)
    share_counts = [len(v) for v in by_item.values()]
    n_shared_all = sum(1 for c in share_counts if c == len(matched_models))
    print(f"\nDistinct items: {n_items}")
    print(f"Items scored by ALL {len(matched_models)} matched models: {n_shared_all} "
          f"({100*n_shared_all/n_items:.1f}%)")
    print(f"Min/median/max models-per-item: {min(share_counts)}/"
          f"{sorted(share_counts)[len(share_counts)//2]}/{max(share_counts)}")

    # ---- Cross-validation against the CURRENT audit's own recorded stratum ----
    audited = load_audit_key()
    print(f"\nCurrent audit key: {len(audited)} (model,item_id) records")
    n_checked, n_mismatch = 0, 0
    mismatches = []
    pop_lookup = {(r["model"], r["item_id"]): r["stratum"] for r in population}
    for (model, item_id), audit_stratum in audited.items():
        derived = pop_lookup.get((model, item_id))
        if derived is None:
            continue  # not in our pulled population (shouldn't happen if roster matched)
        n_checked += 1
        if derived != audit_stratum:
            n_mismatch += 1
            mismatches.append({"model": model, "item_id": item_id,
                                "audit_stratum": audit_stratum, "derived_stratum": derived})
    print(f"Cross-validated {n_checked} already-audited records against re-derived stratum:")
    print(f"  mismatches: {n_mismatch} ({100*n_mismatch/n_checked:.2f}%)" if n_checked else
          "  mismatches: n/a (0 checked)")
    if n_mismatch == 0 and n_checked > 0:
        print("  [OK] stratum derivation EXACTLY matches the existing audit key on every")
        print("  overlapping record -- safe to trust for NOT-yet-audited items too.")
    elif n_mismatch > 0:
        print("  [WARNING] derivation does not perfectly match the existing audit key.")
        print("  Do NOT build a worklist on this until reconciled. Sample mismatches:")
        for m in mismatches[:5]:
            print(f"    {m}")

    out = {
        "n_population_rows": len(population),
        "n_distinct_items": n_items,
        "n_matched_models": len(matched_models),
        "missing_models": sorted(missing),
        "n_items_shared_by_all_models": n_shared_all,
        "share_count_min_median_max": [min(share_counts), sorted(share_counts)[len(share_counts)//2], max(share_counts)],
        "cross_validation": {
            "n_checked": n_checked, "n_mismatch": n_mismatch,
            "mismatches": mismatches[:50],
        },
        "population": population,
    }
    out_path = Path("results/analysis/mathhard_full_roster.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
