"""T17 -- a fourth, learned verifier: xFinder extraction scored with the benchmark's own routine.

Decision D6: test whether the result depends on the three rule-based verifiers, not crown a
fourth. xFinder-qwen1505 (a fine-tuned extractor) reads each response and returns the answer
it commits to, or abstains. Its extraction is then judged by the same harness routine that
decides the reader-derived reference, so the fourth verifier differs from the others in its
extraction step only, as they differ from each other.

The earlier study already ran xFinder on 35,613 of our 35,748 responses, greedily, with the
model's upstream prompt, under two system prompts: "train" (what the checkpoint was fine-tuned
against; primary) and "release" (the shipped default; sensitivity). Those raw extractions are
reused; they are re-scored here because that study judged them with its own comparator. The
remaining 135 responses come from subject files this project fetched later, and are generated
here with that study's own functions (phase `generate`, run in its GPU environment), so every
response gets a fourth verdict and none is imputed.

Phases:
  python scripts/fb_xfinder_verifier.py worklist     -> results/flipbudget/xfinder_worklist.json
  <gpu python> scripts/fb_xfinder_verifier.py generate -> results/flipbudget/xfinder_supplement.jsonl
  python scripts/fb_xfinder_verifier.py score        -> results/flipbudget/xfinder_verdicts.json.gz
"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KS = Path(os.environ.get("FB_COMPANION_REPO", ROOT.parent / "knowledgeshift"))
EXTRACTIONS = KS / "results" / "xfinder" / "extractions.jsonl"
SCRIPT80 = KS / "scripts" / "80_xfinder_extract.py"
CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
WORKLIST = ROOT / "results" / "flipbudget" / "xfinder_worklist.json"
SUPPLEMENT = ROOT / "results" / "flipbudget" / "xfinder_supplement.jsonl"
VERDICTS = ROOT / "results" / "flipbudget" / "xfinder_verdicts.json.gz"
ARMS = ("train", "release")
KILL_SECONDS = 60


def abstained(raw) -> bool:
    """The earlier study's definition: empty, or xFinder's own abstention token, loosely matched."""
    s = str(raw or "").strip().lower().strip("[]() ")
    return not s or s == "no valid answer"


def read_extractions():
    out = {}
    for path in (EXTRACTIONS, SUPPLEMENT):
        if path.exists():
            for line in open(path, encoding="utf-8"):
                d = json.loads(line)
                out[(d["arm"], d["model"], d["doc_hash"])] = d["raw_output"]
    return out


def worklist():
    sys.path.insert(0, str(ROOT / "scripts"))
    from fb_census_k3 import load_rows
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    have = read_extractions()
    need = [(r["model"], r["item_id"]) for r in census["rows"]]
    missing = sorted({mi for mi in need for arm in ARMS if (arm, *mi) not in have})
    rows, _ = load_rows(census["models"])
    text = {(r["model"], r["item_id"]): r for r in rows}
    items = [{"model": m, "doc_hash": i, "problem": text[(m, i)]["problem"], "response": text[(m, i)]["response"]}
             for m, i in missing]
    json.dump(items, open(WORKLIST, "w", encoding="utf-8"), indent=1)
    print(f"{len(items)} responses lack an xFinder extraction in at least one arm -> {WORKLIST}")


def generate():
    """Run in the earlier study's GPU environment. Uses its functions unchanged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("xf80", SCRIPT80)
    xf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xf)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = json.load(open(WORKLIST, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(xf.MODEL_ID, use_fast=False, trust_remote_code=True)  # upstream's own
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(xf.MODEL_ID, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to("cuda").eval()
    stop_ids = xf.resolve_stop_ids(tok)
    done = {(json.loads(l)["arm"], json.loads(l)["model"], json.loads(l)["doc_hash"])
            for l in open(SUPPLEMENT, encoding="utf-8")} if SUPPLEMENT.exists() else set()
    with open(SUPPLEMENT, "a", encoding="utf-8") as f:
        for arm, system in xf.ARMS.items():
            pend = [it for it in items if (arm, it["model"], it["doc_hash"]) not in done]
            for b in range(0, len(pend), 8):
                batch = [(it["model"], it["doc_hash"], xf.format_prompt(system, it["problem"], it["response"]))
                         for it in pend[b:b + 8]]
                for model_name, doc_hash, n_tok, over, text, ms in xf.run_batch_with_oom_retry(model, tok, batch,
                                                                                                stop_ids):
                    f.write(json.dumps({"arm": arm, "model": model_name, "doc_hash": doc_hash, "raw_output": text,
                                        "prompt_tokens": n_tok, "over_train_len": over, "elapsed_ms": ms}) + "\n")
                f.flush()
            print(f"[{arm}] generated {len(pend)}")


def score():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from concurrent.futures import TimeoutError as FutureTimeout

    from pebble import ProcessPool

    from fb_census_k3 import load_rows
    from flipbudget import verifiers
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    ref_rows = load_rows(census["models"][:1])[0]
    gold = {r["item_id"]: r["gold"] for r in ref_rows}
    ext = read_extractions()
    keys = [(r["model"], r["item_id"]) for r in census["rows"]]
    missing = [(arm, *k) for k in keys for arm in ARMS if (arm, *k) not in ext]
    if missing:
        raise SystemExit(f"{len(missing)} extractions missing; run the worklist and generate phases first")
    out = {arm: {} for arm in ARMS}
    t0 = time.time()
    with ProcessPool(max_workers=12) as pool:
        jobs = []
        for arm in ARMS:
            for m, i in keys:
                raw = ext[(arm, m, i)]
                if abstained(raw):
                    out[arm][f"{m}|{i}"] = {"verdict": 0, "found": False, "timed_out": False}
                else:
                    jobs.append((arm, m, i, pool.schedule(verifiers.human_correct, (str(raw), gold[i], "sympy"),
                                                          timeout=KILL_SECONDS)))
        for n, (arm, m, i, fut) in enumerate(jobs, 1):
            try:
                v, killed = fut.result(), False
            except FutureTimeout:
                v, killed = 0, True
            out[arm][f"{m}|{i}"] = {"verdict": int(v), "found": True, "timed_out": killed}
            if n % 10000 == 0:
                print(f"  {n}/{len(jobs)} scored, {time.time() - t0:.0f}s", flush=True)
    json.dump({"arms": list(ARMS), "primary_arm": "train", "extractor": "IAAR-Shanghai/xFinder-qwen1505",
               "judged_by": "verifiers.human_correct(..., 'sympy'), the reader-reference routine",
               "verdicts": out}, gzip.open(VERDICTS, "wt", encoding="utf-8"))
    for arm in ARMS:
        v = out[arm].values()
        print(f"{arm}: credited {sum(x['verdict'] for x in v)} of {len(v)}, abstained {sum(not x['found'] for x in v)}")
    print("wrote", VERDICTS)


if __name__ == "__main__":
    {"worklist": worklist, "generate": generate, "score": score}[sys.argv[1]]()
