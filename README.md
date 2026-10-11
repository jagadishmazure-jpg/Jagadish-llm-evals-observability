# Jagadish LLM Evals & Observability

LLM evaluation and observability **as code**: judges, regression suites,
quality/cost/latency metrics, tracing, and trace analytics — all offline-first,
no API keys required.

## What it does

- **Judges** (`evals/judges/`): a `Judge` interface, a 1–5 **rubric judge**
  with weighted criteria, a **pairwise judge** (A/B/tie), and an
  `OfflineStubJudge` — deterministic, keyword-based, for tests.
- **Regression suites** (`evals/suites/`): load a JSONL/JSON dataset, run a
  model function over every case, score with judges, and get an aggregate
  `SuiteResult` (pass rate, token totals, failures).
- **Metrics** (`evals/metrics/`): accuracy, unbiased pass@k, token accounting,
  USD cost estimation, and latency statistics (mean/p50/p95).
- **Tracing** (`observability/tracing/`): `Span`/`Trace` dataclasses, a
  context-manager `Tracer` with automatic error capture, and an in-memory
  exporter.
- **Analytics** (`observability/analytics/`): cost/latency aggregates per
  model and per agent, plus failure breakdowns by error type and span name.

## Quickstart

```python
from evals.judges import OfflineStubJudge, RubricCriterion, RubricJudge
from evals.suites import RegressionSuite, load_dataset
from evals.metrics import accuracy

judge = OfflineStubJudge()
cases = load_dataset("examples/demo.jsonl")

def fake_model(case):
    from evals.suites import ModelOutput
    return ModelOutput(response="4", prompt_tokens=10, completion_tokens=2)

suite = RegressionSuite(cases, fake_model, judges=[judge])
result = suite.run()
print(f"pass rate: {result.pass_rate:.2%}")
print(f"accuracy:  {accuracy([r.judgment.score for r in result.results if r.judgment]):.2%}")
```

Tracing:

```python
from observability.tracing import Tracer, InMemoryExporter
from observability.analytics import aggregate_by_model

exporter = InMemoryExporter()
tracer = Tracer(exporter)
with tracer.start_span("agent.run", agent="researcher", model="demo-llm",
                        prompt_tokens=120, completion_tokens=40):
    with tracer.start_span("llm.call"):
        ...

for model, agg in aggregate_by_model(exporter.spans).items():
    print(model, f"${agg.cost_usd:.4f}", f"p95={agg.latency.p95_ms:.1f}ms")
```

Swap `OfflineStubJudge` for a real LLM-backed judge in production; the
interfaces are designed for that.

## How to run tests

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or plain pip):

```bash
uv sync --group dev
uv run pytest
```

Every test is offline — no network calls, no API keys.

## Project layout

```
evals/
  judges/      # Judge protocol, rubric, pairwise, offline stub
  suites/      # dataset loader + regression suite runner
  metrics/     # quality, cost, latency
observability/
  tracing/     # spans, tracer context manager, exporters
  analytics/   # per-model/per-agent aggregates, failure breakdown
tests/         # 100+ offline pytest tests
docs/          # metrics overview
```

## License

MIT — see [LICENSE](LICENSE).
