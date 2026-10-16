# Labeling the v3 MATH-Hard audit — instructions

Read this fully before labeling. The measurement is only valid if a person reads each
response and types the answer. No AI tool may produce, suggest, pre-fill or edit any label.
Batches that fail the quality gate are excluded, which has happened four times before.

## Which file is yours

| Labeler | Sheet | Rows | Batches |
|---|---|---|---|
| Primary | `labeling/audit_v3_l1.csv` | 458 | 10 (9 × 50, 1 × 8) |
| Second labeler | `labeling/audit_v3_l2_overlap.csv` | 92 | 2 (50 + 42) |

The second labeler works independently: do not look at the primary labeler's answers, and
do not discuss items. Neither labeler opens anything under `results/` or `provenance/`.

## The tool

1. Open `labeling/app/label.html` in Chrome by double-clicking it. It works offline.
2. Choose your sheet with the file picker, pick the next batch, and press **Start batch**.
3. For each item, read the problem and the model's response, then record what final answer
   the response commits to. Enter or **Save** stores it and moves on. There is no going back
   within a batch.
4. At the end of the batch, press **Download CSV** and save the file into `labeling/exports/`
   without renaming it. The file name carries the batch id and the export hash.

Progress is kept in the browser if the tab closes mid-batch; reopen the same sheet and
batch to continue. Pasting is blocked and attempts are logged, so type every answer.

## The task, exactly

You are transcribing, not solving. Do not check correctness, simplify, or convert forms:
`\frac{1}{2}` stays `\frac{1}{2}`, and `x=2` stays `x=2`. If the final answer is inside
`\boxed{...}`, type what is inside the box, without the `\boxed{}` wrapper.

- **NONE** — the response never commits to a final answer. For example: it is cut off,
  refuses, hedges between options without choosing, or you cannot tell what it claims.
- **CONTRADICTORY** — it states two different final answers, both presented as final, with
  no resolution between them.
- If the response states an answer and later corrects it explicitly, transcribe the
  corrected one.

## Pace

A careful read takes 15–60 seconds per item. Batches read much faster than that are
excluded by the gate. The full primary sheet takes about 4 hours; take breaks between
batches, not in the middle of one.

## After each batch

Check the batch (or send the file to be checked):

```bash
python scripts/fb_audit_v3_gate.py labeling/exports/<file>.csv
```

A passing batch prints `PASS`. A failing batch is copied to
`provenance/excluded_labeling_attempts/` and is not used.
