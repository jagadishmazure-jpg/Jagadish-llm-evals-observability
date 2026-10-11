"""Tests for the rubric judge."""

import pytest

from evals.judges import (
    OfflineStubJudge,
    RubricCriterion,
    RubricJudge,
)

STUB = OfflineStubJudge()


def make_judge(**kwargs):
    criteria = [
        RubricCriterion("accuracy", weight=2.0),
        RubricCriterion("clarity", weight=1.0),
    ]
    return RubricJudge(criteria, STUB.rubric_scorer, **kwargs)


def test_criterion_requires_name():
    with pytest.raises(ValueError):
        RubricCriterion("")


def test_criterion_requires_positive_weight():
    with pytest.raises(ValueError):
        RubricCriterion("x", weight=0)
    with pytest.raises(ValueError):
        RubricCriterion("x", weight=-1)


def test_judge_requires_criteria():
    with pytest.raises(ValueError):
        RubricJudge([], STUB.rubric_scorer)


def test_judge_rejects_duplicate_names():
    with pytest.raises(ValueError):
        RubricJudge(
            [RubricCriterion("a"), RubricCriterion("a")],
            STUB.rubric_scorer,
        )


def test_perfect_response_scores_five():
    judge = make_judge()
    result = judge.judge("q", "the cat sat", reference="the cat sat")
    assert result.weighted_score == pytest.approx(5.0)
    assert result.normalized_score == pytest.approx(1.0)
    assert len(result.scores) == 2


def test_empty_response_scores_one():
    judge = make_judge()
    result = judge.judge("q", "", reference="the cat sat")
    assert result.weighted_score == pytest.approx(1.0)
    assert result.normalized_score == pytest.approx(0.0)


def test_weighted_average_respects_weights():
    def scorer(criterion, prompt, response, reference):
        return {"accuracy": 5, "clarity": 1}[criterion.name]

    judge = RubricJudge(
        [RubricCriterion("accuracy", weight=2.0), RubricCriterion("clarity", weight=1.0)],
        scorer,
    )
    result = judge.judge("q", "r")
    # (2*5 + 1*1) / 3 = 11/3
    assert result.weighted_score == pytest.approx(11 / 3)
    assert result.normalized_score == pytest.approx((11 / 3 - 1) / 4)


def test_score_for_lookup():
    judge = make_judge()
    result = judge.judge("q", "the cat sat", reference="the cat sat")
    assert result.score_for("accuracy") == 5
    assert result.score_for("clarity") == 5


def test_score_for_unknown_criterion_raises():
    judge = make_judge()
    result = judge.judge("q", "x", reference="y")
    with pytest.raises(KeyError):
        result.score_for("nope")


def test_out_of_range_score_rejected():
    judge = RubricJudge([RubricCriterion("a")], lambda *a: 6)
    with pytest.raises(ValueError):
        judge.judge("q", "r")


def test_zero_score_rejected():
    judge = RubricJudge([RubricCriterion("a")], lambda *a: 0)
    with pytest.raises(ValueError):
        judge.judge("q", "r")


def test_non_int_score_rejected():
    judge = RubricJudge([RubricCriterion("a")], lambda *a: 4.5)
    with pytest.raises(TypeError):
        judge.judge("q", "r")


def test_bool_score_rejected():
    judge = RubricJudge([RubricCriterion("a")], lambda *a: True)
    with pytest.raises(TypeError):
        judge.judge("q", "r")


def test_stub_rubric_scorer_scales_with_coverage():
    criterion = RubricCriterion("accuracy")
    assert STUB.rubric_scorer(criterion, "q", "the cat sat", "the cat sat") == 5
    assert STUB.rubric_scorer(criterion, "q", "", "the cat sat") == 1
    mid = STUB.rubric_scorer(criterion, "q", "the cat", "the cat sat on the mat")
    assert 1 < mid < 5


def test_criteria_property():
    judge = make_judge()
    assert [c.name for c in judge.criteria] == ["accuracy", "clarity"]
