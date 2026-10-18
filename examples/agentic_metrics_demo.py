"""Score a fictional refund agent on all 7 deterministic agentic metrics.

Runs three recorded trajectories (successful refund, expired-window refund
with a wasteful duplicate tool call, unknown user) through the trajectory
and behavior metrics, then shows the improvement loop: failure analysis
finds the pattern, a fix is applied, and the comparison report measures
the delta. Fully offline and deterministic.
"""
from evals.behavior import (
    BehaviorCase,
    answer_correctness_score,
    argument_correctness_score,
    tool_correctness_score,
)
from evals.improvement import ComparisonReport, FailureAnalysis, ScoredCase
from evals.trajectory import (
    Trajectory,
    TrajectoryStep,
    plan_adherence_score,
    plan_quality_score,
    step_efficiency_score,
    task_completion_score,
)

CASES = [
    {
        "id": "refund-ok",
        "task": "refund the trail backpack for u-1042",
        "behavior": "A refund ticket was issued for $45.",
        "minimal": {"get_purchase_history", "calculate_refund", "issue_ticket"},
        "case": BehaviorCase(
            input="u-1042 wants a refund for the trail backpack",
            expected_answer="A refund ticket was issued for $45.",
            expected_tools=["get_purchase_history", "calculate_refund", "issue_ticket"],
            expected_arguments={
                "get_purchase_history": {"username": "u-1042"},
                "calculate_refund": {"username": "u-1042", "items": ["trail backpack"]},
                "issue_ticket": {"username": "u-1042", "amount": 45},
            },
        ),
        "trajectory": Trajectory(
            steps=[
                TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
                TrajectoryStep(
                    "calculate_refund",
                    {"username": "u-1042", "items": ["trail backpack"]},
                ),
                TrajectoryStep("issue_ticket", {"username": "u-1042", "amount": 45}),
            ],
            final_answer="A refund ticket for $45 has been issued.",
            declared_plan=[
                "look up purchase history",
                "calculate refund amount",
                "issue refund ticket",
            ],
        ),
    },
    {
        "id": "refund-expired",
        "task": "check the water bottle refund request",
        "behavior": "No refund was issued because the purchase is outside the 30-day window.",
        "minimal": {"get_purchase_history", "get_today"},
        "case": BehaviorCase(
            input="u-1042 wants a refund for the water bottle",
            expected_answer="No refund was issued because the purchase is outside the 30-day window.",
            expected_tools=["get_purchase_history", "get_today"],
            expected_arguments={
                "get_purchase_history": {"username": "u-1042"},
                "get_today": {},
            },
        ),
        "trajectory": Trajectory(
            steps=[
                TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
                TrajectoryStep("get_today", {}),
                TrajectoryStep("get_today", {}),
            ],
            final_answer="No refund: the water bottle was purchased 45 days ago, outside the 30-day window.",
            declared_plan=["look up purchase history", "get today's date"],
        ),
    },
]


def score_all(spec):
    case, traj = spec["case"], spec["trajectory"]
    tools = [s.tool_name for s in traj.steps]
    args = {s.tool_name: dict(s.arguments) for s in traj.steps}
    return {
        "task_completion": task_completion_score(traj, spec["behavior"]),
        "step_efficiency": step_efficiency_score(traj, spec["minimal"]),
        "plan_quality": plan_quality_score(traj, spec["task"]),
        "plan_adherence": plan_adherence_score(traj),
        "tool_correctness": tool_correctness_score(tools, case.expected_tools),
        "argument_correctness": argument_correctness_score(args, case.expected_arguments),
        "answer_correctness": answer_correctness_score(traj.final_answer, case.expected_answer),
    }


def main():
    all_results = {}
    for spec in CASES:
        results = score_all(spec)
        all_results[spec["id"]] = results
        passed = sum(1 for r in results.values() if r.passed)
        print(f"case {spec['id']}: {passed}/7 metrics passed")
        for name, result in results.items():
            flag = "PASS" if result.passed else "FAIL"
            print(f"  {name:<20} {result.score:.2f} {flag:<4} {result.reason}")

    # Improvement loop on the wasteful case.
    records = [
        ScoredCase(case_id=cid, metric_name=name, result=result)
        for cid, results in all_results.items()
        for name, result in results.items()
    ]
    patterns = FailureAnalysis(records).top_patterns()
    print(f"\ntop failure pattern: {patterns[0].describe()}")
    print(f"  e.g. {patterns[0].example_reason}")

    expired = CASES[1]
    fixed_traj = Trajectory(
        steps=[
            TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
            TrajectoryStep("get_today", {}),
        ],
        final_answer=expired["trajectory"].final_answer,
        declared_plan=expired["trajectory"].declared_plan,
    )
    fixed_spec = dict(expired, trajectory=fixed_traj)
    after = score_all(fixed_spec)
    report = ComparisonReport(
        before={m: [r.score] for m, r in all_results["refund-expired"].items()},
        after={m: [r.score] for m, r in after.items()},
    )
    print("\nbefore/after comparison (duplicate get_today removed):")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
