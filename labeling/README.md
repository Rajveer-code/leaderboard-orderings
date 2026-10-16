# Labeling instructions

Two spreadsheets, two different tasks. Read the section for the one you were
given. Don't look anything up, don't try to guess who wrote a response —
judge only from the text in front of you.

Only ever type in the last column. Leave every other column exactly as it
is — `uid` in particular is how your row gets matched back to the right
item; if it gets edited, duplicated, or deleted, that row can't be used.
Don't reorder, sort, or filter rows. Save as CSV when done (not .xlsx) and
send it back with the same filename it came with.

---

## `tc_expansion_l1.csv` — math problems

Each row has a math problem and a model's response to it. Your column is
`your_answer`.

**What to do:** find the final answer the response commits to, and type it
exactly as written — same notation, same form. `\frac{1}{2}` stays
`\frac{1}{2}`, not `0.5`. `x=2` stays `x=2`, not `2`. You're transcribing what
the text says, not solving the problem yourself and not simplifying it.

- **Response never gives a final answer** (cut off, refuses, never
  concludes) → type `NONE`.
- **Response states two different final answers with no resolution between
  them** (not a correction, just two contradicting claims both presented as
  final) → type `CONTRADICTORY`.
- **Response corrects itself** ("wait, that's wrong, actually it's X") → type
  the final, corrected value. That's one commitment, not a contradiction.
- **Response hedges between options and never picks one** ("it could be
  1/2 or 1/3") → `NONE`. If it hedges and then picks one, type the pick.
- **Response is cut off mid-way** → use whatever answer was already stated
  before the cutoff. If nothing was stated yet, `NONE`.
- **You genuinely can't tell** what it commits to → `NONE`. Don't force a
  guess.

Long response cells look intimidating in a spreadsheet — widen the column or
turn on wrap-text rather than skimming.

---

## `tc_expansion2_l1.csv` — math problems, round 2 (458 rows)

**Same task as `tc_expansion_l1.csv` above, same column (`your_answer`),
same rules.** This is a fresh set of 458 problems (no overlap with the ones
already done), sized for a properly-powered estimate instead of the small
first round.

**Full instructions with worked examples for every edge case (self-correction,
hedging, cutoff, notation, units, pacing, the pre-submission checklist):
`labeling/HUMAN_LABELING_INSTRUCTIONS_tc_expansion2.md`. Read that file in
full before starting this one — it supersedes the short version here.**

---

## `ifeval_audit_l1.csv` — instruction-following (redo — read this fully before starting)

An earlier pass on this same sheet was checked against the real automated
grader and found wrong on 89% of the rows marked compliant, concentrated
entirely in one pattern: **countable or exact-format requirements were not
actually checked, just eyeballed.** Example of what went wrong last time —
a prompt said *"Don't contain the letter 'p'"* and a response containing
"**exp**ert", "to**p**-notch" was marked compliant anyway, because it read
as a reasonable, well-written answer overall. It isn't compliant — it
visibly contains the forbidden letter. That's the failure mode this redo
has to avoid.

Each row has a prompt (which may contain one or more specific instructions
— e.g. "write X, and don't use commas") and a model's response to it. Your
column is `your_judgment`.

**What to do:** decide whether the response actually does what the prompt's
instructions ask for. **"Does it read like a good answer" is not the
question — "does it provably satisfy every specific requirement stated" is.**

**If any instruction in the prompt is countable or checkable against an
exact rule, you must actually check it, not estimate it from a read-through:**

- *Forbidden or required letter/word* ("don't use the letter X", "must
  include the word Y") → search the response for it (Ctrl+F). One
  occurrence of a forbidden letter/word fails the instruction, however good
  the response otherwise reads.
- *No commas / no other specific punctuation* → scan for that character
  specifically. One comma fails it.
- *Exact word / sentence / paragraph count* ("at least 300 words", "exactly
  3 paragraphs") → count them. Close doesn't count — check the actual
  number against the actual requirement.
- *Exact format* (JSON, a title in `<<double angle brackets>>`, quotation
  marks around the whole response, a specific end phrase, a postscript
  starting with a specific marker) → check the literal structure character
  by character, not "does it look roughly like that."
- *Choose exactly one of N given fixed options* → the response must
  literally be one of those options, verbatim. A paraphrase or a discussion
  of the options is not a match.

**Then:**
- **Response provably satisfies every instruction** → `YES`.
- **Response fails at least one instruction** (single-instruction prompt:
  this means it fails outright) → `NO`.
- **Prompt has multiple instructions and the response satisfies some but
  not all** → `PARTIAL`.

Judge only against what the prompt actually asked for — not whether the
response is well-written, correct on facts, or something you'd personally
prefer. A well-written response that breaks one stated rule is still `NO`
(or `PARTIAL` if other instructions in the same prompt are satisfied).
