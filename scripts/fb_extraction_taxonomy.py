"""T21 + T22 -- what the disagreements are, computed rather than labelled.

T21, template compliance. The MATH-Hard prompt demonstrates an answer template ("Final Answer:
The final answer is X. I hope it is correct."), and the 2024 scorer extracts only from it. So
whether a response follows the template is a property of its text, decided here by that
scorer's own extraction function (verifiers.em_template_found). It is reported for every
response, and crossed with the reader-derived reference on the audited responses, so that
answer recovery, template compliance, extraction and equivalence are separated.

T22, a mechanical taxonomy of the discordant responses, from the extraction state each
verifier records. Every discordant response falls in exactly one class:
  runtime     a verifier timed out or was killed;
  extraction  otherwise, some verifier that says 0 found no answer to judge -- sub-classed by
              which: template (2024 scorer), box (boxed verifier), parse (Math-Verify);
  judgement   otherwise, every verifier that says 0 did extract an answer, so the disagreement
              is over which answer the response gives or whether it matches the reference.
No label is needed; where the reader-derived reference exists, its credit rate per class is
reported alongside. The gallery takes one seeded random example per class, not a chosen one.

Run: python scripts/fb_extraction_taxonomy.py
Out: results/flipbudget/extraction_taxonomy.json, manuscript/tmlr/taxonomy_table.tex,
     manuscript/tmlr/gallery_table.tex
"""
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import verifiers as V  # noqa: E402
from fb_census_k3 import load_rows  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
AUDIT = ROOT / "results" / "flipbudget" / "audit_k3_map.json"
OUT = ROOT / "results" / "flipbudget" / "extraction_taxonomy.json"
TABLE = ROOT / "manuscript" / "tmlr" / "taxonomy_table.tex"
GALLERY = ROOT / "manuscript" / "tmlr" / "gallery_table.tex"
FIELDS = ("em", "sb_sympy", "mv")
SEED = 20260926
TAIL = 240  # characters of each gallery response shown


def classify(r, template_found):
    if any(r.get(f"{v}_{k}") in (True, "True") for v in ("em_rec", "sb", "mv") for k in ("timeout", "killed")):
        return "runtime"
    failed = []
    if int(r["em"]) == 0 and not template_found:
        failed.append("template")
    if int(r["sb_sympy"]) == 0 and r.get("sb_found") in (False, "False"):
        failed.append("box")
    if int(r["mv"]) == 0 and r.get("mv_pred_parsed") in (False, "False"):
        failed.append("parse")
    return "extraction:" + "+".join(failed) if failed else "judgement"


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
           "_": r"\_", "%": r"\%", "^": r"\^{}", "~": r"\~{}"}
    return "".join(rep.get(ch, ch) for ch in s.replace("\n", " "))


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    rows = {(r["model"], r["item_id"]): r for r in census["rows"]}
    text, _ = load_rows(census["models"])
    resp = {(t["model"], t["item_id"]): t["response"] for t in text}
    tmpl = {k: bool(isinstance(resp.get(k), str) and V.em_template_found(resp[k])) for k in rows}
    cell = {k: "".join(str(r[f]) for f in FIELDS) for k, r in rows.items()}

    # T21 at board level: template compliance for every response
    n = len(rows)
    credited_any = [k for k in rows if "1" in cell[k]]
    others_not_em = [k for k in rows if int(rows[k]["em"]) == 0 and (int(rows[k]["sb_sympy"]) or int(rows[k]["mv"]))]
    board = {
        "n_responses": n,
        "template_compliant": sum(tmpl.values()),
        "credited_by_some_verifier": len(credited_any),
        "credited_by_some_verifier_noncompliant": sum(not tmpl[k] for k in credited_any),
        "credited_by_boxed_or_mv_not_2024": len(others_not_em),
        "credited_by_boxed_or_mv_not_2024_noncompliant": sum(not tmpl[k] for k in others_not_em),
        "compliance_by_cell": {c: [sum(tmpl[k] for k in rows if cell[k] == c), sum(cell[k] == c for k in rows)]
                               for c in sorted(set(cell.values()))},
    }

    # T22: taxonomy of the discordant responses
    disc = [k for k in rows if cell[k] not in ("000", "111")]
    cls = {k: classify(rows[k], tmpl[k]) for k in disc}
    top = Counter(c.split(":")[0] for c in cls.values())
    sub = Counter(c for c in cls.values() if c.startswith("extraction"))
    by_cell = defaultdict(Counter)
    for k in disc:
        by_cell[cell[k]][cls[k].split(":")[0]] += 1

    # the reader-derived reference where it exists (existing audit), crossed with compliance
    audit = json.load(open(AUDIT, encoding="utf-8"))["items"]
    cross = Counter()
    per_class_reader = defaultdict(lambda: [0, 0])
    for it in audit:
        k = (it["model"], it["item_id"])
        if k not in rows:
            continue
        y = 0 if it["indeterminate"] else int(it["ystar"]["sympy"])
        key = ("reader_correct" if y else "reader_incorrect", "compliant" if tmpl[k] else "noncompliant")
        cross[key + ("n",)] += 1
        for f in FIELDS:
            cross[key + (f,)] += int(rows[k][f])
        if k in cls:
            per_class_reader[cls[k].split(":")[0]][0] += y
            per_class_reader[cls[k].split(":")[0]][1] += 1
    audited = {"|".join(k): v for k, v in sorted(cross.items())}

    # gallery: one seeded random discordant response per class that has members
    rng = random.Random(SEED)
    gallery = []
    for c in ("extraction:template", "extraction:box", "extraction:parse", "judgement", "runtime"):
        members = sorted(k for k in disc if cls[k] == c or (c == "runtime" and cls[k] == "runtime"))
        if not members:
            continue
        k = rng.choice(members)
        gallery.append({"class": c, "model": k[0], "item_id": k[1], "cell": cell[k], "template": tmpl[k],
                        "response_tail": resp[k][-TAIL:]})

    out = {"board": board, "taxonomy": {"n_discordant": len(disc), "top": dict(top), "extraction_sub": dict(sub),
                                        "by_cell": {c: dict(v) for c, v in sorted(by_cell.items())}},
           "audited_cross": audited, "reader_credit_by_class": {c: v for c, v in per_class_reader.items()},
           "gallery": gallery, "seed": SEED}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    names = {"extraction:template": "template sentence missing (2024 verifier finds nothing)",
             "extraction:box": "no boxed answer (boxed verifier finds nothing)",
             "extraction:parse": "unparseable (Math-Verify finds nothing)"}
    lines = []
    for c, v in sorted(sub.items(), key=lambda kv: -kv[1]):
        lines.append(f"\\quad {tex_escape(names.get(c, c.replace('extraction:', '')))} & {v} \\\\")
    with open(TABLE, "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/fb_extraction_taxonomy.py -- do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Why the verifiers disagree: every discordant response, "
                "classified from what each verifier's extraction step recorded, with no human labels. "
                "\\emph{Extraction}: a verifier that gave no credit found no answer. \\emph{Judgement}: "
                "every verifier that gave no credit extracted an answer. \\emph{Runtime}: a verifier "
                "timed out. The classes are exclusive.}\n\\label{tab:taxonomy}\n\\small\n")
        f.write("\\begin{tabular}{lr}\n\\toprule\nclass & responses \\\\\n\\midrule\n")
        f.write(f"extraction & {top.get('extraction', 0)} \\\\\n" + "\n".join(lines) + "\n")
        f.write(f"judgement & {top.get('judgement', 0)} \\\\\nruntime & {top.get('runtime', 0)} \\\\\n")
        f.write(f"\\midrule\nall discordant & {len(disc):,} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
                .replace(",", "{,}", 1))  # thousands separator as in the text: 1{,}159
    with open(GALLERY, "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/fb_extraction_taxonomy.py -- do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\n\\caption{One discordant response per class, drawn at random "
                "with a fixed seed rather than chosen, showing its last characters. The cell is the verdict "
                "string (2024 verifier, boxed verifier, Math-Verify).}\n\\label{tab:gallery}\n\\small\n")
        f.write("\\begin{tabular}{p{0.2\\textwidth}cp{0.62\\textwidth}}\n\\toprule\n"
                "class & cell & end of the response \\\\\n\\midrule\n")
        for g in gallery:
            f.write(f"{tex_escape(g['class'].replace('extraction:', 'extraction: '))} & {g['cell']} & "
                    f"\\texttt{{\\footnotesize {tex_escape(g['response_tail'])}}} \\\\\n\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("gallery",)}, indent=1)[:2500])
    print("wrote", OUT, TABLE, GALLERY)


if __name__ == "__main__":
    main()
