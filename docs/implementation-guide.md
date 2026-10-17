# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| LLM judge | Azure OpenAI model behind the `Judge` protocol | Calibrate on human labels |
| Suites | Azure AI Foundry evaluations, or a CI job on every prompt/model change |  |
| Tracing | OpenTelemetry to Application Insights | Keep span attribute names |
| Analytics | Log Analytics / KQL workbooks over spans |  |
| Pricing | Azure retail prices for the deployed models |  |

## Steps

1. Implement an LLM judge against Azure OpenAI and calibrate it on 50 to 100 human-labelled cases.
2. Add an OpenTelemetry exporter that sends spans to Application Insights.
3. Run the regression suite in CI on every prompt or model change and fail the build under a threshold.
4. Load real prices per deployment and alert on cost per request.

## Guardrails to keep when going live

- A model exception becomes a recorded failure; the suite never crashes mid-run.
- Rubric scores outside 1-5 or of the wrong type raise instead of being averaged in.
- Spans record errors automatically, so failures show up in analytics.
- Unknown models fall back to default pricing in analytics instead of crashing; pass an explicit table for billing.

## What to measure before calling it production-ready

- Pass rate per release on a fixed gold set.
- Judge agreement with human labels.
- Cost and p95 latency per agent per day.
