# Jagadish LLM Evals & Observability

[![CI](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

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

Methodology notes: [docs/metrics.md](docs/metrics.md).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest -q
```

```python
from evals.judges import OfflineStubJudge
from evals.suites import RegressionSuite, load_dataset

suite = RegressionSuite(load_dataset("examples/demo.jsonl"),
                        my_model_fn, judges=[OfflineStubJudge()])
result = suite.run()
print(f"pass rate: {result.pass_rate:.2%}")
```

Swap `OfflineStubJudge` for a real LLM-backed judge in production; the
`Judge` interface is unchanged.

## Project structure

```
evals/
  judges/      Judge interface, rubric judge, pairwise judge, OfflineStubJudge
  suites/      regression suites: datasets, model functions, SuiteResult
  metrics/     accuracy, pass@k, token accounting, cost, latency stats
observability/
  tracing/     Span/Trace, Tracer context manager, in-memory exporter
  analytics/   per-model/per-agent cost+latency aggregates, failure breakdowns
examples/      runnable dataset (demo.jsonl)
docs/          metrics.md
tests/         111 offline pytest tests
```

## Tests & CI

**111 tests, all passing, fully offline.** GitHub Actions runs the suite on
every push and pull request (Python 3.12, via `uv`). Releases ship to PyPI
from version tags (`v*`) via trusted publishing — see
`.github/workflows/release.yml` (add manually; workflow files can't be
pushed via the GitHub API).

## Portfolio

Part of an agentic-AI engineering portfolio by
[jagadishmazure-jpg](https://github.com/jagadishmazure-jpg):

| Repo | What it is | Tests |
|---|---|---|
| [Jagadish-multi-agent-orchestration](https://github.com/jagadishmazure-jpg/Jagadish-multi-agent-orchestration) | Orchestration patterns as code | 124 |
| [Jagadish-agent-security-redteam](https://github.com/jagadishmazure-jpg/Jagadish-agent-security-redteam) | Red-teaming suite: attack catalog, layered defenses, risk scoring | 206 |
| **This repo** | LLM evals + observability as code | 111 |
| [Jagadish-ingredion-agentic-ai](https://github.com/jagadishmazure-jpg/Jagadish-ingredion-agentic-ai) | End-to-end multi-agent platform (Microsoft Agent Framework 1.0) + live demo | 564 |

## License

MIT — see [LICENSE](LICENSE).
