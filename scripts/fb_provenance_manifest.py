"""Machine-readable provenance manifest (item 17): for every result JSON in
results/flipbudget/, find the script that writes it (by grepping scripts for
the literal output path -- reliable since every script in this project
writes its own JSON path as a literal string, verified convention) and the
most recent git commit that touched it. Human-label sources are flagged
explicitly since they are the one non-reproducible-from-code input.

Run: python scripts/fb_provenance_manifest.py
"""
import json
import re
import subprocess
from pathlib import Path

RESULTS_DIR = Path("results/flipbudget")
SCRIPTS_DIR = Path("scripts")

HUMAN_LABEL_INPUTS = {
    "tc_real_analysis.json": ["labeling/tc_expansion_l1_completed.csv",
                               "labeling/tc_expansion_l2_answers.csv",
                               "labeling/tc_expansion_l3_answers.csv"],
    "tc_alpha_beta_update.json": ["labeling/tc_expansion_l1_completed.csv"],
    "is_equiv_bug_verification.json": ["labeling/tc_expansion_l1_completed.csv"],
}


SELF = "fb_provenance_manifest.py"

# Written via an ad hoc inline verification snippet during the session
# (not a standalone script file), honestly noted rather than left as an
# unexplained gap.
MANUAL_OVERRIDE = {
    # naive string-match picks a CONSUMER (open() for reading) before the
    # true producer for these -- corrected by hand, checked against a
    # direct grep for the literal open(..., "w") call.
    "tc_correlated_error.json": "scripts/tc_correlated_error.py",
}

AD_HOC_WRITTEN = {
    "ea_dominance_corrected.json": "inline snippet re-deriving E-A single-model "
        "dominance under bounded_single_model_extrema (EVIDENCE_TABLE.md item A2)",
    "ea_dominance_pairwise_corrected.json": "inline snippet re-deriving E-A pairwise "
        "dominance from reconciliation_four_layers.json's per-pair widths (EVIDENCE_TABLE.md item A1)",
    "ifeval_label_quality_check_round2.json": "scripts/fb_ifeval_label_quality_check.py, "
        "run with CLI args pointing at the round-2 judged file (not the hardcoded default path)",
    "ifeval_per_model.json": "inline snippet during EB_IFEVAL_STATUS.md's per-model breakdown",
    "kill_criterion_6_post_tc.json": "inline snippet re-verifying pooling spread on "
        "the T-C-updated pool (THEORY_EXTENSIONS.md §6)",
}


def find_writer_script(json_relpath):
    pattern = json_relpath.replace("\\", "/")
    fname = pattern.split("/")[-1]
    if fname in MANUAL_OVERRIDE:
        return MANUAL_OVERRIDE[fname]
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        if script.name == SELF:
            continue
        try:
            text = script.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern in text or fname in text:
            return str(script).replace("\\", "/")
    return AD_HOC_WRITTEN.get(fname)


def git_last_commit(path):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H|%ai|%s", "--", path],
            capture_output=True, text=True, timeout=10,
        )
        line = out.stdout.strip()
        if not line:
            return None
        h, date, msg = line.split("|", 2)
        return {"commit": h, "date": date, "message": msg}
    except Exception:
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("Provenance manifest: results/flipbudget/*.json")
    print("=" * 70)

    manifest = []
    n_no_writer = 0
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if jf.name == "PROVENANCE_MANIFEST.json":
            continue
        rel = str(jf).replace("\\", "/")
        writer = find_writer_script(rel)
        if writer is None:
            n_no_writer += 1
        commit = git_last_commit(rel)
        human_inputs = HUMAN_LABEL_INPUTS.get(jf.name)
        entry = {
            "output": rel, "size_bytes": jf.stat().st_size,
            "writer_script": writer,
            "last_commit": commit,
            "human_label_inputs": human_inputs,
        }
        manifest.append(entry)
        flag = "" if writer else "  [NO WRITER SCRIPT FOUND -- CHECK MANUALLY]"
        print(f"  {jf.name:<50} <- {writer or '???'}{flag}")

    print(f"\nTotal result files: {len(manifest)}")
    print(f"Files with no writer script found by this method: {n_no_writer}")
    print(f"Files with a human-label input (non-reproducible-from-code alone): "
          f"{sum(1 for m in manifest if m['human_label_inputs'])}")

    with open("results/flipbudget/PROVENANCE_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump({"n_results": len(manifest), "n_no_writer_found": n_no_writer,
                    "entries": manifest}, f, indent=2)
    print("\nWritten to results/flipbudget/PROVENANCE_MANIFEST.json")
