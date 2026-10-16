"""Redesigned 458-row audit (TC_EXPANSION_PREREGISTRATION.md Amendment 3).

Design (strata: model x joint verifier cell; unlabeled rows only):
  229 rows from the two DISAGREEMENT cells (score_boxed-only, exact_match-only),
      allocated across (model, cell) strata in proportion to stratum size;
  229 rows from the CONCORDANT cells, spread evenly over model x cell -- a
      per-model check of the concordant-homogeneity assumption the joint
      analysis rests on (Amendment 3), and blinding for the disagreement rows.
Inclusion probabilities are recorded per row for design-weighted analysis.
Outputs: labeling/audit_v2_l1.csv (blinded) and
results/analysis/audit_v2_provenance.json (NOT for distribution).
Run: python scripts/fb_disagreement_sample.py
"""
import csv
import json
import random
import urllib.request
from collections import defaultdict

import truststore
from huggingface_hub import get_token

truststore.inject_into_ssl()

SEED = 20260919
N_DISAGREE, N_CONC = 229, 229


def cell_of(r):
    sb, em = r["score_boxed_stratum"] == "credited", r["exact_match"] == 1
    return {(1, 1): "both_credited", (0, 0): "both_rejected",
            (1, 0): "score_boxed_only", (0, 1): "exact_match_only"}[(int(sb), int(em))]


def allocate(sizes, total):
    """Largest-remainder proportional allocation, capped at stratum size."""
    tot = sum(sizes.values())
    raw = {k: total * v / tot for k, v in sizes.items()}
    n = {k: min(sizes[k], int(raw[k])) for k in sizes}
    for k in sorted(sizes, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if sum(n.values()) >= total:
            break
        if n[k] < sizes[k]:
            n[k] += 1
    return n


def main():
    pop = json.load(open("results/analysis/tc_expansion_population.json", encoding="utf-8"))["population"]
    models = sorted({r["model"] for r in pop})
    strata = defaultdict(list)
    for r in pop:
        if not r["already_labeled"] and r["exact_match"] is not None:
            strata[(r["model"], cell_of(r))].append(r)
    rng = random.Random(SEED)
    dis = {k: len(v) for k, v in strata.items() if k[1] in ("score_boxed_only", "exact_match_only")}
    n_alloc = allocate(dis, N_DISAGREE)
    conc = {(m, c): len(strata[(m, c)]) for m in models for c in ("both_credited", "both_rejected")
            if strata[(m, c)]}
    per = {k: 0 for k in conc}
    order = sorted(conc)
    i = 0
    while sum(per.values()) < N_CONC:  # capacity-aware round-robin over model x concordant cell
        k = order[i % len(order)]
        if per[k] < conc[k]:
            per[k] += 1
        i += 1
    n_alloc.update(per)
    sample = []
    for k, n in sorted(n_alloc.items()):
        for r in rng.sample(strata[k], n):
            sample.append(r | {"cell": k[1], "pi": n / len(strata[k])})
    print(f"sampled {len(sample)} rows; disagreement {sum(v for k, v in n_alloc.items() if k in dis)}")

    token = get_token()
    hdr = {"Authorization": f"Bearer {token}"}
    need = defaultdict(set)
    for r in sample:
        need[(r["model"], r["subject"])].add(r["item_id"])
    full = {}
    for (model, subject), ids in need.items():
        enc = model.replace("/", "__")
        meta = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://huggingface.co/api/datasets/open-llm-leaderboard/{enc}-details", headers=hdr), timeout=30))
        fname = sorted(s["rfilename"] for s in meta["siblings"]
                       if f"samples_leaderboard_math_{subject}_hard" in s["rfilename"])[-1]
        text = urllib.request.urlopen(urllib.request.Request(
            f"https://huggingface.co/datasets/open-llm-leaderboard/{enc}-details/resolve/main/{fname}",
            headers=hdr), timeout=60).read().decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("doc_hash") in ids:
                resp = rec["resps"][0][0] if isinstance(rec["resps"][0], list) else rec["resps"][0]
                doc = rec.get("doc") or {}
                full[(model, rec["doc_hash"])] = (doc.get("problem") or "", resp)
    rng.shuffle(sample)  # blind order
    rows, prov = [], []
    for r in sample:
        prob, resp = full.get((r["model"], r["item_id"]), ("", ""))
        if not resp:
            continue
        uid = f"{rng.getrandbits(64):016x}"
        rows.append({"uid": uid, "problem": prob, "response": resp, "your_answer": ""})
        prov.append({"uid": uid, "model": r["model"], "item_id": r["item_id"], "subject": r["subject"],
                     "cell": r["cell"], "inclusion_prob": r["pi"], "gold": r["gold"]})
    with open("labeling/audit_v2_l1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "problem", "response", "your_answer"])
        w.writeheader()
        w.writerows(rows)
    json.dump({"seed": SEED, "allocation": {f"{k[0]}|{k[1]}": v for k, v in n_alloc.items()},
               "stratum_sizes": {f"{k[0]}|{k[1]}": len(v) for k, v in strata.items()},
               "n_final": len(rows), "rows": prov},
              open("results/analysis/audit_v2_provenance.json", "w", encoding="utf-8"), indent=1)
    print(f"wrote labeling/audit_v2_l1.csv ({len(rows)} rows, blinded) and provenance")


if __name__ == "__main__":
    main()
