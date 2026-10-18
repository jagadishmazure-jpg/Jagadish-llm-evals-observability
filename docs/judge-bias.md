# Judge bias probes

`evals/judges/bias.py` checks whether a pairwise judge can be trusted before
its verdicts feed a regression gate. Run it whenever you change the judge
model, the prompt or the temperature.

## What is measured

Each pair is judged twice: `(A, B)` and then `(B, A)`. Each verdict is mapped
back to content:

| Verdicts (A first, B first) | Meaning |
|---|---|
| `a, b` or `b, a` | content-stable: the same answer won both times |
| `tie, tie` | content-stable tie |
| `a, a` or `b, b` | order flip: the same **slot** won both times (position bias) |
| `tie, a`, `b, tie`, ... | order flip: the verdict changed with order |

- **Position probe** (`POSITION_PAIRS`): one answer covers the reference and
  the other does not. Biased = same slot won both times.
- **Verbosity probe** (`VERBOSITY_PAIRS`): both answers cover the reference
  equally, and the long one only adds filler. Biased = the longer answer won
  **in both orders**. A pair that flips with order is reported as an order
  flip, not as length bias. Equal-length pairs are not applicable and are left
  out of the denominator.

Both reports give `bias_rate`, `bias_ci` (95% Wilson) and `order_flip_rate`.
`passes` requires both rates to be at or below `MAX_ACCEPTABLE_BIAS_RATE`
(0.10). `confidently_passes` also requires the upper end of the interval to be
at or below 0.10.

## Fixes in this version

The first version had two errors that made a biased judge look better and
blamed the wrong cause:

1. Equal-length pairs counted in the verbosity denominator. An always-longer
   judge scored 1/10 when nine of the ten pairs had equal length, and passed.
   Now the denominator is applicable pairs only.
2. The verbosity probe judged one order only, so an always-first judge was
   reported as length-biased whenever the long answer sat in slot A. Now it
   shows as 40/40 order flips and 0 length bias. It still fails, for the right
   reason.

## Why 40 pairs and a Wilson interval

At n = 6, 0 biased pairs gives a 95% upper bound of 0.39, which a coin-flip
judge could produce. The Wilson score interval stays sound at small n and at
0 or 1, where the normal approximation collapses to a zero-width interval.
For 0 biased pairs, the upper bound is under 0.10 from n = 35. The probe sets
have 40 items each, so the upper bound is 0.088.

## Real results (from `examples/demo.py`, re-checked in CI)

| Judge | Position | Verbosity |
|---|---|---|
| Offline stub (coverage-based) | 0/40, CI 0.00-0.09, 0 flips: PASS | 0/40, CI 0.00-0.09, 0 flips: PASS |
| Always picks slot A | 40/40, 40 flips: FAIL | 0/40 length, 40 flips: FAIL |
| Always picks the longer answer | 16/40, 16 flips: FAIL | 40/40, 0 flips: FAIL |

Only the stub's position result is about the stub itself. It scores keyword
coverage, so 0 bias is expected by construction. The rigged rows show that
the probes catch bias and attribute it correctly. A real LLM judge would give
the numbers that matter.

## Not covered

Self-preference (a judge favouring its own model family), style or format
bias, and domain shift. The probe items are short general-knowledge questions,
so build probe sets from your own data before trusting a judge there.
