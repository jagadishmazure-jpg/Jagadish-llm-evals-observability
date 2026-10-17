"""Compare two candidate 'models' on the demo set, then trace a small agent run and analyse it."""
from pathlib import Path

from evals.judges import OfflineStubJudge
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
