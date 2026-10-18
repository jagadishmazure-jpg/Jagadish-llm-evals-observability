"""Tests for deterministic behavior metrics (tools, arguments, answers)."""

import pytest

from evals.behavior import (
    BehaviorCase,
    answer_correctness_score,
    argument_correctness_score,
    tool_correctness_score,
)


def test_behavior_case_defaults():
    case = BehaviorCase(input="hi", expected_answer="hello")
    assert case.expected_tools == ()
    assert case.expected_arguments == {}
    assert case.expected_plan == ()


# --- tool_correctness_score ------------------------------------------------


def test_tool_correctness_perfect():
    tools = ["get_purchase_history", "calculate_refund", "issue_ticket"]
    r = tool_correctness_score(tools, tools)
    assert r.score == 1.0 and r.passed


def test_tool_correctness_extra_tool_hurts_precision():
    r = tool_correctness_score(["a", "b", "c"], ["a", "b"])
    # precision 2/3, recall 1.0 -> F1 0.8, exactly at the default threshold
    assert r.score == pytest.approx(0.8)
    assert r.passed
    assert "'c'" in r.reason


def test_tool_correctness_missing_tool_hurts_recall():
    r = tool_correctness_score(["a"], ["a", "b"])
    assert r.score == pytest.approx(2 / 3)
    assert not r.passed
    assert "'b'" in r.reason


def test_tool_correctness_both_empty():
    r = tool_correctness_score([], [])
    assert r.score == 1.0 and r.passed


def test_tool_correctness_nothing_called():
    r = tool_correctness_score([], ["a"])
    assert r.score == 0.0 and not r.passed


def test_tool_correctness_nothing_expected_but_called():
    r = tool_correctness_score(["a"], [])
    assert r.score == 0.0 and not r.passed


# --- argument_correctness_score --------------------------------------------


def test_argument_correctness_perfect():
    args = {"issue_ticket": {"username": "u-1042", "amount": 45}}
    r = argument_correctness_score(args, args)
    assert r.score == 1.0 and r.passed


def test_argument_correctness_wrong_value():
    r = argument_correctness_score(
        {"issue_ticket": {"username": "u-1042", "amount": 40}},
        {"issue_ticket": {"username": "u-1042", "amount": 45}},
    )
    assert r.score == pytest.approx(0.5)
    assert not r.passed
    assert "amount" in r.reason


def test_argument_correctness_missing_tool_scores_zero():
    r = argument_correctness_score(
        {},
        {"issue_ticket": {"username": "u-1042"}},
    )
    assert r.score == 0.0 and not r.passed
    assert "'issue_ticket' was not called" in r.reason


def test_argument_correctness_no_expected():
    r = argument_correctness_score({"a": {"x": 1}}, {})
    assert r.score == 1.0 and r.passed


def test_argument_correctness_extra_actual_tool_not_penalized():
    r = argument_correctness_score(
        {"a": {"x": 1}, "extra": {"y": 2}},
        {"a": {"x": 1}},
    )
    assert r.score == 1.0 and r.passed


def test_argument_correctness_type_strict():
    # 45 != "45": arguments are machine values, so types matter
    r = argument_correctness_score({"a": {"n": "45"}}, {"a": {"n": 45}})
    assert r.score == 0.0 and not r.passed


# --- answer_correctness_score ----------------------------------------------


def test_answer_correctness_exact_match():
    r = answer_correctness_score("A refund was issued.", "A refund was issued.")
    assert r.score == 1.0 and r.passed


def test_answer_correctness_paraphrase_passes():
    r = answer_correctness_score(
        "We issued a refund ticket in the amount of $45.",
        "A refund ticket was issued for $45.",
    )
    assert r.score == 1.0 and r.passed


def test_answer_correctness_word_order_irrelevant():
    # Not an exact string match: same claims in scrambled order still pass.
    r = answer_correctness_score(
        "$45 for issued was ticket refund a",
        "A refund ticket was issued for $45.",
    )
    assert r.score == 1.0 and r.passed


def test_answer_correctness_wrong_conclusion_fails():
    r = answer_correctness_score(
        "A refund ticket for $45 has been issued and should arrive soon.",
        "No refund was issued because the purchase is outside the 30-day window.",
    )
    assert r.score == 0.0 and not r.passed


def test_answer_correctness_empty_expected_raises():
    with pytest.raises(ValueError):
        answer_correctness_score("something", "   ")
