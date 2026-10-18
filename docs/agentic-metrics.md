# Agentic metrics: scoring how the agent worked, not just what it said

Outcome metrics ("did it get the right answer?") miss the most expensive
agent failures: 12 tool calls where 3 would do, the right tools in a
nonsense order, a good plan that was never followed. These seven metrics
score the **trajectory** (the path) and the **behavior** (tools, arguments,
conclusions) deterministically — no LLM judge, no API keys, identical runs
always produce identical scores.

Methodology mapping to [DeepEval](https://github.com/confident-ai/deepeval)
(metrics re-implemented here in original, dependency-free code):

| This repo | DeepEval equivalent | What it catches |
|---|---|---|
| `task_completion_score` | `TaskCompletionMetric` | Wrong or missing outcome in the final answer |
| `step_efficiency_score` | `StepEfficiencyMetric` | Unnecessary tool calls (cost, latency) |
| `plan_quality_score` | `PlanQualityMetric` | Bad tool sequences: re-fetches, actions before lookups, calls after the outcome was decided |
| `plan_adherence_score` | `PlanAdherenceMetric` | Ignoring the agent's own declared plan |
| `tool_correctness_score` | `ToolCorrectnessMetric` | Wrong tool selection (set-overlap F1) |
| `argument_correctness_score` | `ArgumentCorrectnessMetric` | Right tool, wrong arguments |
| `answer_correctness_score` | `GEval` (answer correctness) | Wrong logical conclusion, even with word overlap |

## The contract: trajectories and behavior cases

```python
from evals.trajectory import Trajectory, TrajectoryStep

traj = Trajectory(
    steps=[
        TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
        TrajectoryStep("calculate_refund", {"username": "u-1042"}),
    ],
    final_answer="A refund ticket for $45 has been issued.",
    declared_plan=["look up purchase history", "calculate refund amount"],
)
```

```python
from evals.behavior import BehaviorCase

case = BehaviorCase(
    input="u-1042 wants a refund for the trail backpack",
    expected_answer="A refund ticket was issued for $45.",
    expected_tools=["get_purchase_history", "calculate_refund", "issue_ticket"],
    expected_arguments={"issue_ticket": {"username": "u-1042", "amount": 45}},
)
```

`BehaviorCase` is the "expected behavior" contract: stricter than an
answer-only gold set, because it pins down which tools should run with
which arguments. Every metric takes a `threshold` and returns a
`MetricResult(score, passed, reason)` with a human-readable reason.

## What each metric actually checks

**Task completion** extracts claims from the expected behavior and scores
the fraction covered by the final answer (60% of a claim's content tokens
must appear; case/punctuation-insensitive, stopwords ignored). A
paraphrase passes; a missing conclusion fails.

**Step efficiency** is `necessary calls / total calls` against a minimal
tool set. The classic catch: an agent that calls a date tool it never
needed. Duplicates are not penalized here — that is plan quality's job.

**Plan quality** applies structural heuristics, each costing 0.25:
redundant re-fetch (same tool, identical arguments), action before
information (e.g. issuing a refund with no lookup first), and calls after
a terminal action (tools run once the outcome was already decided).

**Plan adherence** checks the declared plan against execution: a planned
step counts as followed when half its content tokens appear across tool
names and reasoning summaries. A good plan that was ignored scores low
even when the answer was right.

**Tool correctness** is set-overlap F1: missing tools hurt recall, extra
tools hurt precision. **Argument correctness** is the per-tool argument
match rate, averaged over expected tools; values compare canonically, so
`45` never accidentally equals `"45"`. **Answer correctness** is logical
conclusion match — not exact string match: scrambled word order with the
same claims still passes.

## Deterministic vs LLM judge: when to use which

Use these deterministic metrics when the expectation can be written down:
the right tools, the right arguments, the required claims. They are cheap,
stable in CI, and their failures are unambiguous.

Reach for an LLM judge (the `Judge` protocol in `evals/judges/`) when the
expectation is genuinely open-ended: tone, helpfulness, plan creativity.
When you do, keep the deterministic metrics as the regression floor — the
LLM judge should never be the only thing standing between a bad agent
and production.

## Closing the loop

`evals/improvement.py` turns scores into fixes:

1. `FailureAnalysis` groups failing (case, metric) records by reason
   signature — quoted values and numbers are abstracted, so "unnecessary:
   'get_today'" and "unnecessary: 'check_date'" surface as one pattern.
2. Fix the agent, re-run, and `ComparisonReport(...).to_markdown()`
   renders the per-metric delta table for the PR.

See `examples/agentic_metrics_demo.py` for the loop running end to end on
a fictional refund agent.
