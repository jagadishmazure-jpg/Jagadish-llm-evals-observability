"""Tests for the Judge interface and the offline stub judge."""

import pytest

from evals.judges import Judge, Judgment, OfflineStubJudge


def test_judgment_defaults():
    j = Judgment(score=0.75)
    assert j.score == 0.75
    assert j.label == ""
    assert j.rationale == ""
    assert j.metadata == {}


def test_judgment_is_frozen():
    j = Judgment(score=1.0)
    with pytest.raises(AttributeError):
        j.score = 0.0  # type: ignore[misc]


def test_stub_satisfies_judge_protocol():
    assert isinstance(OfflineStubJudge(), Judge)


def test_stub_full_coverage_scores_one():
    judge = OfflineStubJudge()
    result = judge.judge("q", "the cat sat on the mat", reference="the cat sat")
    assert result.score == pytest.approx(1.0)
    assert result.label == "pass"


def test_stub_partial_coverage():
    judge = OfflineStubJudge()
    result = judge.judge("q", "the cat", reference="the cat sat on the mat")
    assert result.score == pytest.approx(0.4)
    assert result.label == "fail"


def test_stub_empty_response_scores_zero():
    judge = OfflineStubJudge()
    result = judge.judge("q", "   ", reference="something")
    assert result.score == 0.0
    assert result.label == "fail"


def test_stub_no_reference_nonempty_passes():
    judge = OfflineStubJudge()
    result = judge.judge("q", "any answer")
    assert result.score == 1.0
    assert result.label == "pass"


def test_stub_no_reference_empty_fails():
    judge = OfflineStubJudge()
    result = judge.judge("q", "")
    assert result.score == 0.0
    assert result.label == "fail"


def test_stub_is_deterministic():
    judge = OfflineStubJudge()
    first = judge.judge("q", "the cat sat", reference="the cat")
    second = judge.judge("q", "the cat sat", reference="the cat")
    assert first == second


def test_stub_rationale_mentions_offline():
    judge = OfflineStubJudge()
    result = judge.judge("q", "x", reference="y")
    assert "offline stub" in result.rationale
    assert result.metadata["judge"] == "offline-stub"
