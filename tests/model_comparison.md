# Model comparison — full 11-question eval run

Same agent code, same system prompt, same questions, same dataset
(Sephora product catalog, 8,494 products). The only variable is the
model. Opus results: [eval_results.md](eval_results.md). Sonnet
results: [eval_results_sonnet.md](eval_results_sonnet.md).

| Metric | Claude Opus 4.8 (`claude-opus-4-8`) | Claude Sonnet 5 (`claude-sonnet-5`) |
|---|---|---|
| Score (hand-graded) | 11/11 | 11/11 |
| Total wall-clock time | not recorded | 82s |
| Input tokens | not recorded | 5,267 (+ 6,129 cache writes, 122,580 cache reads) |
| Output tokens | not recorded | 3,857 |
| Estimated cost per full run | not recorded | ~$0.09 |

Notes:

- The Opus run predates the timing/token instrumentation added to
  `run_eval.py`, so those numbers were not captured and are not
  guessed here. A future Opus re-run would record them automatically.
- Sonnet cost uses introductory Claude Sonnet 5 pricing ($2 input /
  $10 output per million tokens, through 2026-08-31; standard is
  $3/$15), with cache writes at 1.25x and cache reads at 0.1x the
  input rate.
- Earlier hand-graded Opus history (10.5 → 10.5 → 11 across prompt
  iterations) is summarized in the project README's Evaluation
  section.
