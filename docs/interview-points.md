# Interview points

Short, honest talking points for Jagadish-llm-evals-observability. Every number quoted is from the
real run shown in the README.

- Evals turn quality into a regression test; the judge is just a pluggable interface.
- Errors count as failures, not crashes: the sloppy candidate's timeout shows up as a failed case.
- pass@k uses the unbiased estimator: 3 of 10 correct gives 0.30 at k=1 and 0.917 at k=5.
- Honest gap: only a stub judge ships today.

## Likely follow-up questions

- How do you trust an LLM judge? (Measure agreement with human labels and watch for position bias in pairwise.)
- What goes on a span? (Model, agent, tokens, error type; never raw PII.)
- How big should a gold set be? (Start at 50 to 100 real cases and grow from failures.)
