"""Tests for the improvement loop: failure analysis and before/after reports."""

import pytest

from evals.improvement import ComparisonReport, FailureAnalysis, ScoredCase
from evals.trajectory import MetricResult


def scored(case_id, metric, score, threshold=0.8):
    return ScoredCase(
        case_id=case_id,
        metric_name=metric,
        result=MetricResult(
            score=score, passed=score >= threshold, reason=f"reason for {case_id}"
        ),
    )


def eff_fail(case_id, tool):
    return ScoredCase(
        case_id=case_id,
        metric_name="step_efficiency",
        result=MetricResult(
            score=0.66,
            passed=False,
            reason=f"2/3 calls necessary; unnecessary: '{tool}'",
        ),
    )


# --- FailureAnalysis -------------------------------------------------------


def test_failures_filters_passes():
    fa = FailureAnalysis([scored("c1", "m", 1.0), scored("c2", "m", 0.2)])
    assert [r.case_id for r in fa.failures()] == ["c2"]


def test_failure_rate():
    fa = FailureAnalysis(
        [scored("c1", "m", 1.0), scored("c2", "m", 0.2), scored("c3", "m", 0.1)]
    )
    assert fa.failure_rate("m") == pytest.approx(2 / 3)


def test_failure_rate_unknown_metric_raises():
    fa = FailureAnalysis([scored("c1", "m", 1.0)])
    with pytest.raises(ValueError):
        fa.failure_rate("nope")


def test_metrics_lists_all():
    fa = FailureAnalysis([scored("c1", "b", 1.0), scored("c2", "a", 0.0)])
    assert fa.metrics() == ["a", "b"]


def test_top_patterns_groups_by_signature():
    fa = FailureAnalysis(
        [
            eff_fail("c1", "get_today"),
            eff_fail("c2", "check_date"),
            scored("c3", "plan_quality", 0.5),
        ]
    )
    patterns = fa.top_patterns()
    assert patterns[0].metric_name == "step_efficiency"
    assert patterns[0].occurrences == 2
    assert patterns[0].case_ids == ("c1", "c2")
    assert "{}" in patterns[0].signature  # tool name abstracted away


def test_top_patterns_tiebreak_deterministic():
    fa = FailureAnalysis(
        [
            ScoredCase("c1", "b", MetricResult(0.1, False, "zeta")),
            ScoredCase("c2", "a", MetricResult(0.1, False, "alpha")),
        ]
    )
    patterns = fa.top_patterns()
    assert [p.metric_name for p in patterns] == ["a", "b"]


def test_top_patterns_empty_when_no_failures():
    fa = FailureAnalysis([scored("c1", "m", 1.0)])
    assert fa.top_patterns() == []


def test_top_patterns_n_validated():
    fa = FailureAnalysis([scored("c1", "m", 0.0)])
    with pytest.raises(ValueError):
        fa.top_patterns(n=0)


def test_pattern_describe():
    fa = FailureAnalysis([eff_fail("c1", "get_today"), eff_fail("c2", "get_today")])
    desc = fa.top_patterns()[0].describe()
    assert "step_efficiency" in desc and "c1" in desc and "c2" in desc


# --- ComparisonReport ------------------------------------------------------


def test_delta_positive():
    rep = ComparisonReport(
        before={"step_efficiency": [0.66, 0.66]},
        after={"step_efficiency": [1.0, 1.0]},
    )
    assert rep.delta("step_efficiency") == pytest.approx(0.34)


def test_delta_negative():
    rep = ComparisonReport(before={"m": [0.9]}, after={"m": [0.4]})
    assert rep.delta("m") == pytest.approx(-0.5)


def test_delta_missing_side_raises():
    rep = ComparisonReport(before={"m": [0.9]}, after={})
    with pytest.raises(ValueError):
        rep.delta("m")


def test_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        ComparisonReport(before={"m": [1.5]}, after={"m": [0.5]})


def test_to_markdown_table_and_summary():
    rep = ComparisonReport(
        before={"step_efficiency": [0.66, 0.66], "plan_quality": [1.0]},
        after={"step_efficiency": [1.0, 1.0], "plan_quality": [1.0]},
    )
    md = rep.to_markdown()
    assert "| step_efficiency | 0.66 | 1.00 | +0.34 |" in md
    assert "| plan_quality | 1.00 | 1.00 | +0.00 |" in md
    assert "1 metric(s) improved, 0 regressed" in md


def test_to_markdown_missing_side_is_na():
    rep = ComparisonReport(before={"m": [0.5]}, after={"other": [0.9]})
    md = rep.to_markdown()
    assert "| m | n/a | n/a | n/a | n/a | n/a |" in md


def test_custom_thresholds_drive_pass_rates():
    rep = ComparisonReport(
        before={"m": [0.7]},
        after={"m": [0.7]},
        thresholds={"m": 0.6},
    )
    md = rep.to_markdown()
    assert "100%" in md  # 0.7 passes the custom 0.6 threshold on both sides
