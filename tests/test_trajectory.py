"""Tests for deterministic trajectory metrics (task completion, step efficiency,
plan quality, plan adherence)."""

import pytest

from evals.trajectory import (
    MetricResult,
    Trajectory,
    TrajectoryStep,
    plan_adherence_score,
    plan_quality_score,
    step_efficiency_score,
    task_completion_score,
)


def make_traj(**kwargs):
    defaults = dict(
        steps=[
            TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
            TrajectoryStep("calculate_refund", {"username": "u-1042"}),
            TrajectoryStep("issue_ticket", {"username": "u-1042", "amount": 45}),
        ],
        final_answer="A refund ticket for $45 has been issued.",
        declared_plan=[
            "look up purchase history",
            "calculate refund amount",
            "issue refund ticket",
        ],
    )
    defaults.update(kwargs)
    return Trajectory(**defaults)


# --- MetricResult ----------------------------------------------------------


def test_metric_result_valid():
    r = MetricResult(score=0.75, passed=True, reason="ok")
    assert r.score == 0.75 and r.passed and r.reason == "ok"


def test_metric_result_rejects_bad_score():
    with pytest.raises(ValueError):
        MetricResult(score=1.5, passed=False, reason="x")


def test_metric_result_rejects_empty_reason():
    with pytest.raises(ValueError):
        MetricResult(score=0.5, passed=False, reason="")


# --- task_completion_score -------------------------------------------------


def test_task_completion_happy_path():
    r = task_completion_score(make_traj(), "A refund ticket was issued for $45.")
    assert r.score == 1.0 and r.passed


def test_task_completion_paraphrase_passes():
    traj = make_traj(final_answer="We issued a refund ticket in the amount of $45.")
    r = task_completion_score(traj, "A refund ticket was issued for $45.")
    assert r.score == 1.0 and r.passed


def test_task_completion_partial_credit():
    traj = make_traj(final_answer="A refund ticket was issued.")
    r = task_completion_score(
        traj, "A refund ticket was issued for $45. The customer was notified."
    )
    assert r.score == pytest.approx(0.5)
    assert not r.passed  # below default 0.8
    assert "missing" in r.reason


def test_task_completion_threshold_param():
    traj = make_traj(final_answer="A refund ticket was issued.")
    r = task_completion_score(
        traj, "A refund ticket was issued for $45. The customer was notified.", threshold=0.5
    )
    assert r.score == pytest.approx(0.5) and r.passed


def test_task_completion_empty_expected_raises():
    with pytest.raises(ValueError):
        task_completion_score(make_traj(), "   ")


def test_task_completion_bad_threshold_raises():
    with pytest.raises(ValueError):
        task_completion_score(make_traj(), "a ticket was issued", threshold=2.0)


# --- step_efficiency_score -------------------------------------------------


def test_step_efficiency_all_necessary():
    r = step_efficiency_score(
        make_traj(), {"get_purchase_history", "calculate_refund", "issue_ticket"}
    )
    assert r.score == 1.0 and r.passed


def test_step_efficiency_extra_call_penalized():
    traj = make_traj(
        steps=[
            TrajectoryStep("get_purchase_history", {}),
            TrajectoryStep("get_today", {}),
            TrajectoryStep("calculate_refund", {}),
        ]
    )
    r = step_efficiency_score(traj, {"get_purchase_history", "calculate_refund"})
    assert r.score == pytest.approx(2 / 3)
    assert not r.passed
    assert "'get_today'" in r.reason


def test_step_efficiency_no_calls_with_expectations():
    r = step_efficiency_score(Trajectory(), {"get_purchase_history"})
    assert r.score == 0.0 and not r.passed


def test_step_efficiency_no_calls_no_expectations():
    r = step_efficiency_score(Trajectory(), set())
    assert r.score == 1.0 and r.passed


# --- plan_quality_score ----------------------------------------------------


def test_plan_quality_good_sequence():
    r = plan_quality_score(make_traj(), "refund the backpack")
    assert r.score == 1.0 and r.passed


def test_plan_quality_redundant_refetch():
    traj = make_traj(
        steps=[
            TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
            TrajectoryStep("get_purchase_history", {"username": "u-1042"}),
            TrajectoryStep("issue_ticket", {"username": "u-1042"}),
        ]
    )
    r = plan_quality_score(traj, "issue a refund")
    assert r.score == pytest.approx(0.75)
    assert "redundant re-fetch" in r.reason


def test_plan_quality_action_before_information():
    traj = make_traj(steps=[TrajectoryStep("issue_ticket", {"amount": 45})])
    r = plan_quality_score(traj, "issue a refund")
    assert r.score == pytest.approx(0.75)
    assert "action before information" in r.reason


def test_plan_quality_calls_after_terminal_action():
    traj = make_traj(
        steps=[
            TrajectoryStep("get_purchase_history", {}),
            TrajectoryStep("issue_ticket", {}),
            TrajectoryStep("get_today", {}),
        ]
    )
    r = plan_quality_score(traj, "issue a refund")
    assert r.score == pytest.approx(0.75)
    assert "after terminal action" in r.reason


def test_plan_quality_multiple_penalties_floor():
    traj = make_traj(
        steps=[
            TrajectoryStep("issue_ticket", {}),
            TrajectoryStep("get_today", {}),
            TrajectoryStep("get_today", {}),
            TrajectoryStep("check_weather", {}),
            TrajectoryStep("check_weather", {}),
        ]
    )
    r = plan_quality_score(traj, "issue a refund")
    # action-before-info + trailing calls + 2 redundant re-fetches = 4 penalties
    assert r.score == 0.0 and not r.passed


def test_plan_quality_empty_trajectory():
    r = plan_quality_score(Trajectory(), "do something")
    assert r.score == 0.0 and not r.passed


# --- plan_adherence_score --------------------------------------------------


def test_plan_adherence_full():
    r = plan_adherence_score(make_traj())
    assert r.score == 1.0 and r.passed


def test_plan_adherence_partial():
    traj = make_traj(
        declared_plan=["look up purchase history", "send marketing email"],
        steps=[TrajectoryStep("get_purchase_history", {})],
    )
    r = plan_adherence_score(traj)
    assert r.score == pytest.approx(0.5)
    assert not r.passed


def test_plan_adherence_no_plan():
    r = plan_adherence_score(make_traj(declared_plan=[]))
    assert r.score == 0.0 and not r.passed
    assert "no declared plan" in r.reason


def test_plan_adherence_uses_reasoning_summaries():
    traj = Trajectory(
        steps=[TrajectoryStep("custom_tool", {}, reasoning_summary="verified refund eligibility")],
        declared_plan=["verify refund eligibility"],
    )
    r = plan_adherence_score(traj)
    assert r.score == 1.0
