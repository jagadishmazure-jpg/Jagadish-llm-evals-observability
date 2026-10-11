# Metrics overview

This project ships three metric families. All of them run offline on data you
already have — judge scores, token counts, and finished spans.

## Quality metrics (`evals.metrics.quality`)

| Metric | Function | Meaning |
|---|---|---|
| Accuracy | `accuracy(scores, threshold=0.5)` | Fraction of judge scores at or above the threshold. Empty input → `0.0`. |
| Pass@k | `pass_at_k(n, c, k)` | Unbiased estimator (Chen et al., 2021) of the probability that at least one of `k` samples is correct, given `c` correct out of `n`. Returns `1.0` when fewer than `k` samples are wrong. |

Use accuracy for single-shot evals and regression suites; use pass@k when you
sample multiple completions per prompt (e.g. temperature > 0 or agent retries).

## Cost metrics (`evals.metrics.cost`)

- `TokenUsage(prompt_tokens, completion_tokens)` — immutable token counts;
  supports `+` and rejects negatives.
- `total_usage(usages)` — sums a sequence of usages.
- `estimate_cost_usd(usage, model, pricing)` — USD estimate from a per-1M-token
  pricing table `{"model": (input_rate, output_rate)}`. Raises `KeyError` for
  unknown models so silent mispricing can't happen.

Pricing defaults are illustrative; always pass your provider's real table.

## Latency metrics (`evals.metrics.cost.latency_stats`)

`latency_stats(latencies_ms)` returns `LatencyStats` with `count`, `mean_ms`,
`min_ms`, `max_ms`, `p50_ms`, `p95_ms`. Percentiles use linear interpolation.
Empty input and negative samples raise `ValueError`.

## Trace analytics (`observability.analytics`)

| Function | Groups by | Produces |
|---|---|---|
| `aggregate_by_model(spans, pricing)` | `model` span attribute | calls, tokens, USD cost, latency stats |
| `aggregate_by_agent(spans, pricing)` | `agent` span attribute | calls, tokens, USD cost, latency stats |
| `failure_breakdown(spans)` | error type / span name | totals, failure rate, breakdowns |

Spans missing `model`/`agent` attributes are grouped under `"unknown"`.
Latency is only included when spans have finished (`duration_ms` present).
