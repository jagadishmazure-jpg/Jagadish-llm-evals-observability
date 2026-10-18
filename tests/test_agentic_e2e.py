"""End-to-end: a fictional refund agent scored on all 7 agentic metrics.

Scenario: "Cinder & Oak Supply Co." is a fictional online retailer. Its
refund agent has four tools (get_purchase_history, get_today,
calculate_refund, issue_ticket) and a 30-day refund policy. Three cases run
through every metric; a fourth test closes the improvement loop by fixing
the inefficiency the metrics surface.
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

METRICS = [
    "task_completion",
    "step_efficiency",
    "plan_quality",
    "plan_adherence",
    "tool_correctness",
    "argument_correctness",
    "answer_correctness",
]


def run_all_7(case, trajectory, minimal_tools, task_description, expected_behavior):
    actual_tools = [s.tool_name for s in trajectory.steps]
    actual_args = {s.tool_name: dict(s.arguments) for s in trajectory.steps}
    return {
        "task_completion": task_completion_score(trajectory, expected_behavior),
        "step_efficiency": step_efficiency_score(trajectory, minimal_tools),
        "plan_quality": plan_quality_score(trajectory, task_description),
        "plan_adherence": plan_adherence_score(trajectory),
        "tool_correctness": tool_correctness_score(actual_tools, case.expected_tools),
        "argument_correctness": argument_correctness_score(
            actual_args, case.expected_arguments
        ),
        "answer_correctness": answer_correctness_score(
            trajectory.final_answer, case.expected_answer
        ),
    }


def case_ok():
    case = BehaviorCase(
        input="u-1042 wants a refund for the trail backpack",
        expected_answer="A refund ticket was issued for $45.",
        expected_tools=["get_purchase_history", "calculate_refund", "issue_ticket"],
        expected_arguments={
            "get_purchase_history": {"username": "u-1042"},
            "calculate_refund": {"username": "u-1042", "items": ["trail backpack"]},
            "issue_ticket": {"username": "u-1042", "amount": 45},
        },
        expected_plan=[
            "look up purchase history",
            "calculate refund amount",
            "issue refund ticket",
        ],
    )
    trajectory = Trajectory(
        steps=[
            TrajectoryStep(
                "get_purchase_history",
                {"username": "u-1042"},
                "look up purchase history",
            ),
            TrajectoryStep(
                "calculate_refund",
                {"username": "u-1042", "items": ["trail backpack"]},
                "calculate refund amount",
            ),
            TrajectoryStep(
                "issue_ticket",
                {"username": "u-1042", "amount": 45},
                "issue refund ticket",
            ),
        ],
        final_answer="A refund ticket for $45 has been issued and should arrive soon.",
        declared_plan=list(case.expected_plan),
    )
    return case, trajectory


def case_expired():
    """Agent is correct but wasteful: get_today is called twice identically."""
    case = BehaviorCase(
        input="u-1042 wants a refund for the water bottle",
        expected_answer="No refund was issued because the purchase is outside the 30-day window.",
        expected_tools=["get_purchase_history", "get_today"],
        expected_arguments={
            "get_purchase_history": {"username": "u-1042"},
            "get_today": {},
        },
    )
    trajectory = Trajectory(
        steps=[
            TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
            TrajectoryStep("get_today", {}),
            TrajectoryStep("get_today", {}),
        ],
        final_answer=(
            "No refund: the water bottle was purchased 45 days ago, "
            "outside the 30-day window."
        ),
        declared_plan=["look up purchase history", "get today's date"],
    )
    return case, trajectory


def case_invalid_user():
    case = BehaviorCase(
        input="u-999 wants a refund for the book",
        expected_answer=(
            "The username is not valid. Ask the user to check the username "
            "and send it again before giving refund information."
        ),
        expected_tools=["get_purchase_history"],
        expected_arguments={"get_purchase_history": {"username": "u-999"}},
    )
    trajectory = Trajectory(
        steps=[TrajectoryStep("get_purchase_history", {"username": "u-999"})],
        final_answer=(
            "I could not find account u-999, so I cannot verify the purchase. "
            "No refund was issued. Please check the username and send it again."
        ),
        declared_plan=["look up purchase history"],
    )
    return case, trajectory


def test_successful_refund_passes_all_seven():
    case, trajectory = case_ok()
    results = run_all_7(
        case,
        trajectory,
        minimal_tools=set(case.expected_tools),
        task_description="refund the trail backpack for u-1042",
        expected_behavior="A refund ticket was issued for $45.",
    )
    assert set(results) == set(METRICS)
    for name, result in results.items():
        assert result.passed, f"{name} should pass: {result.reason}"


def test_expired_refund_fails_only_plan_quality():
    case, trajectory = case_expired()
    results = run_all_7(
        case,
        trajectory,
        minimal_tools={"get_purchase_history", "get_today"},
        task_description="check the water bottle refund request",
        expected_behavior="No refund was issued because the purchase is outside the 30-day window.",
    )
    assert not results["plan_quality"].passed
    assert "redundant re-fetch" in results["plan_quality"].reason
    for name, result in results.items():
        if name != "plan_quality":
            assert result.passed, f"{name} should pass: {result.reason}"


def test_invalid_user_fails_answer_metrics():
    # The agent never says the username itself is invalid: a real behavioral
    # gap the metrics catch, mirroring a common agent-eval finding.
    case, trajectory = case_invalid_user()
    results = run_all_7(
        case,
        trajectory,
        minimal_tools={"get_purchase_history"},
        task_description="handle a refund request from u-999",
        expected_behavior=(
            "The username is identified as invalid and the user is asked "
            "to provide a valid username."
        ),
    )
    assert not results["answer_correctness"].passed
    assert not results["task_completion"].passed
    for name in (
        "step_efficiency",
        "plan_quality",
        "plan_adherence",
        "tool_correctness",
        "argument_correctness",
    ):
        assert results[name].passed, f"{name} should pass: {results[name].reason}"


def test_improvement_loop_finds_pattern_and_measures_fix():
    case, trajectory = case_expired()
    before = run_all_7(
        case,
        trajectory,
        minimal_tools={"get_purchase_history", "get_today"},
        task_description="check the water bottle refund request",
        expected_behavior="No refund was issued because the purchase is outside the 30-day window.",
    )
    records = [
        ScoredCase(case_id="expired", metric_name=name, result=result)
        for name, result in before.items()
    ]
    analysis = FailureAnalysis(records)
    patterns = analysis.top_patterns()
    assert len(patterns) == 1
    assert patterns[0].metric_name == "plan_quality"
    assert "redundant re-fetch" in patterns[0].signature

    # Fix: drop the duplicate get_today call.
    fixed = Trajectory(
        steps=[
            TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
            TrajectoryStep("get_today", {}),
        ],
        final_answer=trajectory.final_answer,
        declared_plan=trajectory.declared_plan,
    )
    after = run_all_7(
        case,
        fixed,
        minimal_tools={"get_purchase_history", "get_today"},
        task_description="check the water bottle refund request",
        expected_behavior="No refund was issued because the purchase is outside the 30-day window.",
    )
    report = ComparisonReport(
        before={m: [r.score] for m, r in before.items()},
        after={m: [r.score] for m, r in after.items()},
    )
    assert report.delta("plan_quality") == 0.25
    assert report.delta("step_efficiency") == 0.0
    md = report.to_markdown()
    assert "+0.25" in md and "1 metric(s) improved" in md


def test_failure_patterns_cluster_across_tools():
    # Two cases fail step_efficiency on different unnecessary tools: one pattern.
    def wasteful(extra_tool):
        return Trajectory(
            steps=[
                TrajectoryStep("get_purchase_history", {}),
                TrajectoryStep(extra_tool, {}),
            ],
            final_answer="done",
        )

    records = [
        ScoredCase(
            case_id=cid,
            metric_name="step_efficiency",
            result=step_efficiency_score(wasteful(tool), {"get_purchase_history"}),
        )
        for cid, tool in (("c1", "get_today"), ("c2", "check_date"))
    ]
    patterns = FailureAnalysis(records).top_patterns()
    assert len(patterns) == 1
    assert patterns[0].occurrences == 2
    assert patterns[0].case_ids == ("c1", "c2")
