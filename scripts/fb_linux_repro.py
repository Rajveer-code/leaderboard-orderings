"""T24 -- reproduce the census verdicts on Linux, where the original SIGALRM guards exist.

The census was computed on Windows with the timing substitute described in
src/flipbudget/verifiers.py. On Linux the verifiers run as their authors wrote them. This script
scores one model per call (so a CI matrix fits the runners' time limit) and then compares every
verdict with the committed census.

  python scripts/fb_linux_repro.py score MODEL OUTDIR   -> OUTDIR/<model>.json
  python scripts/fb_linux_repro.py compare OUTDIR       -> exit 1 on any unexplained difference

Needs a Hugging Face token with access to the leaderboard's detail datasets when the local
generation cache is absent (FB_CACHE_L1), as on a CI runner.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
FIELDS = ("em_rec", "sb_sympy", "sb_string", "mv")
TIMEOUT_OF = {"em_rec": "em_rec_timeout", "sb_sympy": "sb_timeout", "sb_string": "sb_timeout", "mv": "mv_timeout"}


def score_one(model, outdir):
    from fb_census_k3 import load_rows, score
    rows, sources = load_rows([model])
    for r in rows:
        r["scored"] = bool(isinstance(r["response"], str) and r["gold"] and r["item_id"] and r["em"] in (0, 1))
    score(rows)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out = {r["item_id"]: {f: r.get(f) for f in FIELDS + ("em_rec_timeout", "sb_timeout", "mv_timeout")}
           for r in rows if r["scored"]}
    json.dump({"model": model, "sources": sources, "verdicts": out},
              open(Path(outdir) / (model.replace("/", "__") + ".json"), "w", encoding="utf-8"))
    print(f"{model}: {len(out)} responses scored")


def compare(outdir):
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    ref = {(r["model"], r["item_id"]): r for r in census["rows"]}
    n, diff, timed = 0, [], 0
    for p in sorted(Path(outdir).glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        for item, v in d["verdicts"].items():
            r = ref.get((d["model"], item))
            if r is None:
                continue
            for f in FIELDS:
                n += 1
                if int(v[f]) != int(r[f]):
                    # a verdict decided by a timeout on either platform is expected to be fragile
                    t = TIMEOUT_OF[f]
                    if v.get(t) in (True, "True") or r.get(t) in (True, "True"):
                        timed += 1
                    else:
                        diff.append((d["model"], item, f, v[f], r[f]))
    print(f"compared {n} verdicts: {len(diff)} unexplained differences, {timed} at a timeout")
    for x in diff[:20]:
        print("  ", x)
    return 1 if diff else 0


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score_one(sys.argv[2], sys.argv[3])
    else:
        sys.exit(compare(sys.argv[2]))
