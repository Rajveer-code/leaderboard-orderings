# Operational definitions behind four reported quantities

Each definition below is worded as in the paper's appendices, and
`scripts/fb_verify_definitions.py` re-implements it from this wording, independently of the
production code, and checks that it reproduces the reported number
(`results/analysis/definition_checks.json`).

## Template compliance (Section 5.1; Appendix I)

A response is template compliant when the 2024 verifier's extraction step finds an answer in it.
That step appends the string `I hope it is correct.` to the response and searches the result with
Python's `re.search` for

    Final Answer: The final answer is(.*?). I hope it is correct.

The dots are unescaped and match any character. The response is compliant when the search
succeeds; the captured group is the answer the verifier then judges. Computed for all 35,748
responses with the scoring function recorded in the leaderboard's result files.

## The discordant taxonomy (Table 2; Appendix I)

For every discordant response (cell other than 000 and 111), apply in order; the first rule that
applies decides:

1. **runtime** if any of the three verifiers timed out or was stopped on the response;
2. **extraction** if at least one verifier that gave no credit found no answer, where "found no
   answer" means: the 2024 verifier gave 0 and the response is not template compliant (sub-class
   `template`); the boxed verifier gave 0 and the response contains no `\boxed{}` (`box`); or
   Math-Verify gave 0 and could not parse an answer (`parse`). The sub-class lists every verifier
   for which this holds, joined by `+`;
3. **judgement** otherwise: every verifier that gave no credit extracted an answer.

The classes are exclusive and exhaustive by construction.

## The board-wide reversal statistic (Section 6; Appendix G)

- a_m: model m's MATH-Hard score in `open-llm-leaderboard/results` at revision `c6b22601`
  (31 January 2025, the last revision before the February 2025 re-scoring), from the model's
  most recent results file.
- b_m: its displayed score in `open-llm-leaderboard/contents` at revision `9c09a7ca`
  (20 March 2025, the final public table), the largest value when a model is listed at more than
  one numerical precision.
- For the M models with both scores, a pair {i, j} is reversed when (a_i - a_j)(b_i - b_j) < 0.
- Reported share = reversed pairs / pairs with (a_i - a_j)(b_i - b_j) != 0. Pairs tied under
  either score enter neither count; the share with ties counted as not reversed is reported beside
  it. Kendall's tau-b uses its standard tie correction.

The other board count, 3,751, is the maintainers' own count of models re-scored in February 2025
(`kydlicek2025fixing`); 4,497 is the number of models with a score in the final public table.

## xFinder in the four-verifier analysis (Section 5.4; Appendix F)

- An extraction that is empty or reads "no valid answer" (ignoring case, brackets and
  parentheses) is an abstention and scores 0; so does an extraction whose comparison with the gold
  answer is stopped after 60 seconds. Every other extraction is judged by the equivalence routine
  that decides the reader verdict.
- The verdict is appended to the three-verifier string, giving 16 cells. 0000 and 1111 are
  concordant (the pooling assumption applies to them); the other 14 are discordant and enter each
  model's range at [0, 1] until the registered audit supplies labels.
- Primary arm: the system prompt xFinder was fine-tuned with; sensitivity arm: its shipped default.
