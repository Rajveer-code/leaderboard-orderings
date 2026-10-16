# How to fill `tc_expansion2_l1.csv` — complete instructions

Read this whole document before opening the spreadsheet. It replaces the
short version in `labeling/README.md` for this specific file — that section
still applies, this expands it with worked examples for every case that has
actually caused a problem before.

## Why this document is this long

Two earlier attempts at this same file were rejected — not because the
answers looked wrong, but because the *pattern* of how they were produced
looked automated: near-perfect agreement with the existing scorers,
mechanical bulk edits between two submissions seven minutes apart, zero use
of the markers below that a real reader hits regularly. The full story is
in `TC_EXPANSION2_LABELING_QUALITY_ISSUE.md` if you want it. The practical
upshot: this file exists to measure something specific — whether the
automated scorer agrees with what a human actually reads in the response —
and that measurement is only real if the labels come from actually reading
458 responses, at the pace that takes.

## The file

`labeling/tc_expansion2_l1.csv` — 458 rows, 4 columns: `uid`, `problem`,
`response`, `your_answer`. Confirmed intact before this was written: 458
rows, 458 unique `uid`s, `your_answer` empty in all of them.

**Only ever type in `your_answer`.** Never touch `uid`, `problem`, or
`response` — `uid` is how your row gets matched back to the right item; if
it's edited, duplicated, reordered, or deleted, that row becomes unusable.
Don't sort or filter the sheet. Save as CSV (not .xlsx) and don't rename
the file.

## The task, exactly

Each row has a math problem and a model's response to it. Read the
response and decide: **what final answer does this response actually
commit to?** Then type that answer into `your_answer`, transcribed exactly
as written — same notation, same form.

You are transcribing, not solving. Don't check whether the answer is
mathematically correct, don't simplify it, don't convert its form.
`\frac{1}{2}` stays `\frac{1}{2}`, not `0.5`. `x=2` stays `x=2`, not `2`.
`3\sqrt{2}` stays `3\sqrt{2}`, not `4.24`. If the response's final answer is
inside `\boxed{...}`, copy exactly what's inside the box, braces and all
the LaTeX included, but without the `\boxed{}` wrapper itself (e.g.
`\boxed{7}` → type `7`; `\boxed{\frac{1}{2}}` → type `\frac{1}{2}`).

## The three markers, and exactly when each applies

**`NONE`** — the response never commits to a final answer.
- It's cut off mid-sentence, mid-derivation, before reaching a conclusion.
- It explicitly refuses or fails to solve the problem.
- It hedges between options and never picks one ("the answer could be
  either $1/2$ or $1/3$ depending on interpretation" — and stops there).
- You genuinely cannot tell what, if anything, it's claiming as final.
  Don't force a guess in either direction — `NONE` is the honest answer
  when the text doesn't clearly commit to one.

**`CONTRADICTORY`** — the response states two *different* final answers,
both presented as final, with no resolution between them. Example: the
derivation concludes "So the answer is 12" in one paragraph, and a later,
separate paragraph — not a correction of the first, a second independent
conclusion — says "Final answer: 15." Neither retracts the other. That's
`CONTRADICTORY`, not a pick-one situation.

**A specific value** — everything else, including the two cases people
tend to mishandle:
- **Self-correction is not a contradiction.** "Wait, I made an error —
  it's actually 15, not 12" is *one* commitment (15), arrived at by
  correcting itself along the way. Type `15`. The tell that separates this
  from `CONTRADICTORY` above: does the response itself acknowledge the
  earlier value was wrong? If yes, it's a correction (use the corrected
  value). If both values are asserted as simply true with no acknowledgment
  either was wrong, it's `CONTRADICTORY`.
- **Cut off after committing.** If the response states "the answer is 7"
  and then the text is cut off before elaborating further, the commitment
  already happened — type `7`, not `NONE`. `NONE` is for when nothing was
  ever stated, not for when something was stated and then the text ends.
- **Hedges, then picks.** "It could be $1/2$ or $1/3$... on reflection,
  $1/2$ is correct" → type `1/2`. The hedge doesn't matter once it resolves.

## Notation edge cases

- Multiple equivalent forms given together ("$x = 2$, i.e. $x=2.0$") →
  transcribe the *first* form the response commits to as its answer, not
  every restatement.
- A units label attached to the number ("the answer is 5 dollars") →
  include it as written: `5 dollars` (or `\$5` if that's how it's written)
  — don't strip units, don't add them if absent. Match what the response
  actually says.
- Set or interval notation ("$x \in \{2, 5\}$", "$(-\infty, 3]$") →
  transcribe verbatim, brackets and all.
- A list of several correct values (e.g. all solutions to a quadratic) →
  transcribe the full list as given, in the order given.

## What NOT to do

- **No AI tool anywhere in this task** — not to "help transcribe," not to
  double-check a reading, not for anything. This is not a formality: the
  entire point of this audit is measuring whether a human's reading agrees
  with the automated scorer. Running the text through any LLM first
  reintroduces the exact dependency this file exists to check against, and
  produces a signature the quality gate is specifically built to catch (see
  `TC_EXPANSION2_LABELING_QUALITY_ISSUE.md`).
- No web lookups, no solving the problem yourself, no using outside
  knowledge of what the "correct" answer should be. Judge only the text in
  front of you.
- Don't try to guess which model produced a response or use that as a
  signal. You don't have that information and it isn't relevant.
- Don't average, round, or "improve" an answer the response actually
  states in a messier form.

## Pacing — read this before you start

Real reading time for 458 rows: most rows (roughly 80%) are quick — a short
response with an obvious final answer, 10–20 seconds each. The rest need
real reading — a longer derivation, a self-correction to trace, a
borderline hedge-vs-commit call — call it 1–3 minutes each. Total: several
hours, not several minutes. (For calibration: the two rejected attempts
each claimed to finish all 458 in under ten minutes — that pace is itself
what triggered the review, independent of the answers given.)

**Work in batches, not one continuous sitting:**
1. Split the 458 rows into roughly 8 sessions of ~55–60 rows each — under
   an hour per session at a sustainable pace.
2. After each batch, save your progress. You don't need to submit
   incrementally unless you want a mid-way sanity check — but if you do,
   run the quality gate on the partial file (see below) so you catch a
   systematic misunderstanding after 60 rows, not after 458.
3. Take a real break between batches. Fatigue is when the self-correction
   vs. contradiction distinction starts getting missed.

## Before you send it back — a 5-item self-check

1. Every row in `your_answer` is filled (or deliberately `NONE`/
   `CONTRADICTORY`) — no blanks left from rows you meant to come back to.
2. You used `CONTRADICTORY` at least a handful of times if you genuinely
   encountered it, and didn't quietly resolve every ambiguous case to a
   single value just to avoid using the marker. (Zero `CONTRADICTORY`
   uses across all 458 rows is itself one of the signals the quality gate
   checks — not because using it is required, but because a real read of
   458 varied model responses turning up literally none of that pattern is
   unusual.)
3. Spot-check five rows you remember as genuinely hard — re-read them
   fresh. Do you still agree with what you wrote?
4. `uid` column is untouched — same values, same order, same count (458).
5. File is saved as `.csv`, same filename, only `your_answer` changed from
   the original.

## What happens after you submit

`scripts/fb_tc_expansion2_analysis.py --csv labeling/tc_expansion2_l1.csv --i-confirm-human-labeled`
runs a quality gate before anything else: duplicate-uid check, row-count
check, unmatched-uid check, an answer-corruption heuristic, and the
statistical-signature checks that caught the two earlier submissions
(near-total agreement with the automated scorers in the cells where they
already agree, near-zero agreement in the cell where they're expected to
mostly disagree, `CONTRADICTORY` usage). This isn't a test you can fail by
reading carefully and answering honestly — it's built to catch the specific
automated-substitution pattern, not to second-guess a genuine, careful human
read. If your real answers happen to disagree with the automated scorers
often, that's a legitimate, informative result — say so plainly if asked
about it, don't tune your answers toward what you think the scorer
"should" say.
