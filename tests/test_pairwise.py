"""Tests for the pairwise judge."""

import pytest

from evals.judges import OfflineStubJudge, PairwiseJudge, PairwiseResult

STUB = OfflineStubJudge()


def test_better_coverage_wins():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    result = judge.judge("q", "the cat sat", "the cat", reference="the cat sat")
    assert result.winner == "a"


def test_worse_coverage_loses():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    result = judge.judge("q", "the cat", "the cat sat", reference="the cat sat")
    assert result.winner == "b"


def test_equal_coverage_ties():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    result = judge.judge("q", "the cat", "the cat", reference="the cat sat")
    assert result.winner == "tie"


def test_no_reference_longer_wins():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    result = judge.judge("q", "a much longer response", "short")
    assert result.winner == "a"


def test_no_reference_empty_loses():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    result = judge.judge("q", "", "something")
    assert result.winner == "b"


def test_no_reference_equal_length_ties():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    result = judge.judge("q", "abc", "def")
    assert result.winner == "tie"


def test_invalid_winner_rejected():
    judge = PairwiseJudge(lambda *a: "c")
    with pytest.raises(ValueError):
        judge.judge("q", "a", "b")


def test_result_rejects_bad_winner():
    with pytest.raises(ValueError):
        PairwiseResult(winner="c")


def test_result_rejects_bad_confidence():
    with pytest.raises(ValueError):
        PairwiseResult(winner="a", confidence=1.5)


def test_result_defaults():
    result = PairwiseResult(winner="tie")
    assert result.rationale == ""
    assert result.confidence == 1.0


def test_pairwise_is_deterministic():
    judge = PairwiseJudge(STUB.pairwise_prefer)
    first = judge.judge("q", "the cat sat", "the dog", reference="the cat sat")
    second = judge.judge("q", "the cat sat", "the dog", reference="the cat sat")
    assert first == second


def test_custom_prefer_function_used():
    judge = PairwiseJudge(lambda prompt, a, b, ref: "b")
    assert judge.judge("q", "a", "b").winner == "b"
