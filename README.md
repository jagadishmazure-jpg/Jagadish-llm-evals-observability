# Jagadish-llm-evals-observability

[![ci](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability/actions/workflows/ci.yml)
[![codeql](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability/actions/workflows/codeql.yml/badge.svg)](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability/actions/workflows/codeql.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**LLM evaluation and observability as code: judges (rubric, pairwise, offline stub), regression suites, quality and cost metrics, nested tracing, and per-agent cost and failure analytics. Offline, no API keys.**

> Personal learning and portfolio project. Everything runs offline on synthetic,
> fictional data with no API keys. Nothing is deployed. Numbers in this README
> come from real runs and are re-checked by CI (`scripts/render_docs.py --check`).

## Recruiter summary

- Shows how to put an LLM feature under test: a dataset, a model function, judges, a pass threshold, and token and cost accounting.
- Real run: a careful candidate passes 3/3 and a sloppy one 1/3 (one wrong answer, one timeout caught as a failure); a traced agent run shows tokens and cost per agent and the failing span.
- Judge bias probes: position and verbosity, 40 pairs each, both orders judged, order-flip rate and 95% Wilson interval reported. The stub judge scores 0/40 on both (CI upper bound 0.09); rigged always-first and always-longer judges fail for the right reason.
- Deterministic offline tests; CI with lint, CodeQL, gitleaks and documented-output drift checks.

## Purpose

Without evals, "it worked in the demo" is the only quality signal, and without
traces nobody can say which agent or tool is slow, costly or failing. This repo
gives both as small libraries with stable interfaces, so a deterministic stub
judge in tests can be swapped for an LLM judge later. Methodology:
[docs/metrics.md](docs/metrics.md).

## Architecture

```mermaid
flowchart LR
    DS["dataset<br/>JSONL / JSON"] --> RS["RegressionSuite"]
    MF["model_fn(case)"] --> RS
    J["Judges<br/>rubric, pairwise, stub"] --> RS
    J --> BP["bias probes<br/>position, verbosity<br/>both orders, Wilson CI"]
    RS --> SR["SuiteResult<br/>pass rate, tokens, failures"]
    SR --> M["metrics<br/>accuracy, pass@k, cost, latency"]
    APP["agent code"] --> T["Tracer<br/>nested spans"] --> EX["InMemoryExporter"]
    EX --> AN["analytics<br/>per model / per agent, failures"]
```

## How it works

- **Judges** (`evals/judges/`): a `Judge` protocol; `RubricJudge` validates 1-5
  scores per weighted criterion; `PairwiseJudge` returns A, B or tie;
  `OfflineStubJudge` scores keyword coverage of the reference.
- **Bias probes** (`evals/judges/bias.py`, `probe_sets.py`): every pair is judged
  in both orders. Position bias is the same slot winning both times. A pair is
  length-biased only if the longer answer wins in both orders, so slot
  preference is never blamed on length. Equal-length pairs are left out of the
  denominator. Each report gives the bias rate with a 95% Wilson interval and
  the order-flip rate, and a judge passes only if both are at or below 0.10.
  The 40-item probe sets are large enough that 0/40 has an upper bound (0.088)
  under that threshold. Details: [docs/judge-bias.md](docs/judge-bias.md).
- **Suites** (`evals/suites/`): load cases, run the model function, catch model
  exceptions as failures, apply judges at a threshold, and summarise.
- **Metrics** (`evals/metrics/`): accuracy, unbiased pass@k, token totals, cost
  from a per-1M-token price table, latency mean, p50 and p95.
- **Tracing** (`observability/tracing/`): `Tracer.start_span` context manager with
  parent links and automatic error capture.
- **Analytics** (`observability/analytics/`): cost and latency per model or agent,
  failure counts by error type and span name.

## Real code and output

```python
"""Compare two candidate 'models' on the demo set, trace a small agent run, and probe a pairwise judge for bias."""
from pathlib import Path

from evals.judges import (
    POSITION_PAIRS,
    VERBOSITY_PAIRS,
    OfflineStubJudge,
    PairwiseJudge,
    position_swap_probe,
    verbosity_probe,
)
from evals.metrics.cost import TokenUsage, estimate_cost_usd
from evals.metrics.quality import pass_at_k
from evals.suites import RegressionSuite, load_dataset
from evals.suites.runner import ModelOutput
from observability.analytics.aggregates import aggregate_by_agent, failure_breakdown
from observability.tracing.context import Tracer
from observability.tracing.exporters import InMemoryExporter

PRICING = {"model-small": (0.50, 1.50), "model-large": (5.00, 15.00)}  # illustrative USD per 1M tokens

cases = load_dataset(Path(__file__).with_name("demo.jsonl"))
ANSWERS = {"math-1": "The answer is 4.", "math-2": "10 / 2 = 5", "geo-1": "Paris is the capital of France."}


def careful(case):
    return ModelOutput(response=ANSWERS[case.id], prompt_tokens=40, completion_tokens=12)


def sloppy(case):
    if case.id == "math-2":
        return ModelOutput(response="It is 20.", prompt_tokens=40, completion_tokens=4)
    if case.id == "geo-1":
        raise TimeoutError("upstream timeout")
    return careful(case)


for name, fn, model in (("careful", careful, "model-large"), ("sloppy", sloppy, "model-small")):
    r = RegressionSuite(cases, fn, judges=[OfflineStubJudge()]).run()
    usage = TokenUsage(r.total_prompt_tokens, r.total_completion_tokens)
    cost = estimate_cost_usd(usage, model=model, pricing=PRICING).cost_usd
    print(f"{name:<8} pass rate {r.pass_rate:.0%} ({r.passed}/{r.total}), "
          f"tokens {usage.total_tokens}, est. cost ${cost:.6f} on {model}")
    for f in r.failures():
        print(f"         failed {f.case_id}: {f.error or 'judge score ' + format(f.judgment.score, '.2f')}")

print(f"pass@1 with 3 of 10 samples correct: {pass_at_k(10, 3, 1):.2f}; pass@5: {pass_at_k(10, 3, 5):.3f}")

exporter = InMemoryExporter()
tracer = Tracer(exporter)
with tracer.start_span("agent.run", agent="planner", model="model-large", prompt_tokens=300, completion_tokens=80):
    with tracer.start_span("llm.call", agent="researcher", model="model-small", prompt_tokens=900, completion_tokens=150):
        pass
    try:
        with tracer.start_span("tool.search", agent="researcher"):
            raise ConnectionError("search API unavailable")
    except ConnectionError:
        pass
for key, agg in sorted(aggregate_by_agent(exporter.spans, pricing=PRICING).items()):
    print(f"agent {key:<10} calls {agg.calls}, tokens {agg.prompt_tokens + agg.completion_tokens}, "
          f"est. cost ${agg.cost_usd:.6f}")
fb = failure_breakdown(exporter.spans)
print(f"spans {fb.total_spans}, failed {fb.failed_spans} ({fb.failure_rate:.0%}), by error {fb.by_error_type}, "
      f"by span {fb.by_span_name}")

# Bias probes: 40 pairs each, both orders judged, Wilson 95% CI.
honest = PairwiseJudge(OfflineStubJudge().pairwise_prefer)
slot_a = PairwiseJudge(lambda p, a, b, ref: "a")
longer = PairwiseJudge(lambda p, a, b, ref: "a" if len(a) >= len(b) else "b")
for label, judge in (("stub judge", honest), ("always-first judge", slot_a), ("always-longer judge", longer)):
    print(f"{label}:")
    print(f"  {position_swap_probe(judge, POSITION_PAIRS).summary()}")
    print(f"  {verbosity_probe(judge, VERBOSITY_PAIRS).summary()}")
```

Output of `python examples/demo.py` (real run, re-checked in CI):

<!-- run: python examples/demo.py -->
```text
careful  pass rate 100% (3/3), tokens 156, est. cost $0.001140 on model-large
sloppy   pass rate 33% (1/3), tokens 96, est. cost $0.000064 on model-small
         failed math-2: judge score 0.00
         failed geo-1: TimeoutError: upstream timeout
pass@1 with 3 of 10 samples correct: 0.30; pass@5: 0.917
agent planner    calls 1, tokens 380, est. cost $0.002700
agent researcher calls 2, tokens 1050, est. cost $0.000675
spans 3, failed 1 (33%), by error {'ConnectionError': 1}, by span {'tool.search': 1}
stub judge:
  position-swap: 0/40 biased (rate 0.00, 95% CI 0.00-0.09), order flips 0/40 (0.00), threshold 0.10 -> PASS
  verbosity: 0/40 biased (rate 0.00, 95% CI 0.00-0.09), order flips 0/40 (0.00), threshold 0.10 -> PASS
always-first judge:
  position-swap: 40/40 biased (rate 1.00, 95% CI 0.91-1.00), order flips 40/40 (1.00), threshold 0.10 -> FAIL
  verbosity: 0/40 biased (rate 0.00, 95% CI 0.00-0.09), order flips 40/40 (1.00), threshold 0.10 -> FAIL
always-longer judge:
  position-swap: 16/40 biased (rate 0.40, 95% CI 0.26-0.55), order flips 16/40 (0.40), threshold 0.10 -> FAIL
  verbosity: 40/40 biased (rate 1.00, 95% CI 0.91-1.00), order flips 0/40 (0.00), threshold 0.10 -> FAIL
```
<!-- /run -->

## How to run

```bash
git clone https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability.git
cd Jagadish-llm-evals-observability
python -m pip install pytest      # no runtime dependencies (uv users: uv sync --group dev)
python -m pytest -q          # offline, no keys
python examples/demo.py
python scripts/render_docs.py --check   # documented output still matches
```

Test suite size (real collection, re-checked in CI):

<!-- run: python scripts/count_tests.py -->
```text
136 tests collected
```
<!-- /run -->

## Guardrails

- A model exception becomes a recorded failure; the suite never crashes mid-run.
- Rubric scores outside 1-5 or of the wrong type raise instead of being averaged in.
- Spans record errors automatically, so failures show up in analytics.
- A pairwise judge must pass the position and verbosity probes (bias and order-flip rates both at most 0.10) before its verdicts are trusted.
- Unknown models fall back to default pricing in analytics instead of crashing; pass an explicit table for billing.

## Limitations

- The only built-in judge is a keyword-coverage stub; LLM judges are an interface, not an implementation.
- Prices in the demo are illustrative, and the default table has a single placeholder rate.
- The bias probes cover position and verbosity only. Self-preference and other judge biases are not probed. The probe items are short general-knowledge questions, so a real judge also needs probes in its own domain.
- Per-agent cost is priced using the first span's model in each group.
- Spans are kept in memory; no OpenTelemetry export yet.
- No PyPI release; install from source.

## Built vs planned

| Capability | Status | Where |
|---|---|---|
| Judge protocol, rubric, pairwise, stub | Built | `evals/judges/` |
| Judge bias probes (position, verbosity) with Wilson CI | Built | `evals/judges/bias.py` |
| Regression suite runner | Built | `evals/suites/` |
| Quality, token, cost and latency metrics | Built | `evals/metrics/` |
| Tracer and in-memory exporter | Built | `observability/tracing/` |
| Cost, latency and failure analytics | Built | `observability/analytics/` |
| LLM-backed judge | Planned | docs/implementation-guide.md |
| OpenTelemetry / Application Insights exporter | Planned | not started |
| PyPI package | Planned | not started |

## Implementation guide

The step-by-step path from this offline build to an Azure deployment is in
[docs/implementation-guide.md](docs/implementation-guide.md). None of it has been
run; it is a plan.

## Related repositories

| Repo | How it relates |
|---|---|
| [Jagadish-agent-security-redteam](https://github.com/jagadishmazure-jpg/Jagadish-agent-security-redteam) | Security suite that fits the same regression pattern |
| [Jagadish-multi-agent-orchestration](https://github.com/jagadishmazure-jpg/Jagadish-multi-agent-orchestration) | Agents whose runs these traces describe |
| [Jagadish-agentic-rag](https://github.com/jagadishmazure-jpg/Jagadish-agentic-rag) | RAG-specific metrics (faithfulness, recall) |
| [Jagadish-dspy-prompt-optimizer](https://github.com/jagadishmazure-jpg/Jagadish-dspy-prompt-optimizer) | Uses a metric to choose prompts |
| [Jagadish-azure-finops](https://github.com/jagadishmazure-jpg/Jagadish-azure-finops) | Cost tracking at the cloud level |

## Documentation

- [Metrics methodology](docs/metrics.md)
- [Judge bias probes](docs/judge-bias.md)
- [Implementation guide (Azure)](docs/implementation-guide.md)
- [Interview points](docs/interview-points.md)
- [Security policy](SECURITY.md)

## Project layout

```
evals/
  judges/      Judge protocol, rubric, pairwise, stub, bias probes + probe sets
  suites/      datasets, runner, SuiteResult
  metrics/     quality, cost, latency
observability/
  tracing/     Span, Tracer, exporters
  analytics/   aggregates, failure breakdown
examples/      demo.jsonl + demo.py (the run shown above)
docs/          metrics.md
tests/         offline pytest suite
```

## License

MIT, see [LICENSE](LICENSE). All data is synthetic; every organisation and person is fictional.
