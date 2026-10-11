"""Tests for quality metrics: accuracy and pass@k."""

import pytest

from evals.metrics import accuracy, pass_at_k


def test_accuracy_all_pass():
    assert accuracy([0.9, 1.0, 0.7]) == pytest.approx(1.0)


def test_accuracy_partial():
    assert accuracy([0.9, 0.1, 0.7, 0.2]) == pytest.approx(0.5)


def test_accuracy_none_pass():
    assert accuracy([0.1, 0.2]) == pytest.approx(0.0)


def test_accuracy_empty_is_zero():
    assert accuracy([]) == 0.0


def test_accuracy_custom_threshold():
    assert accuracy([0.8, 0.9], threshold=0.85) == pytest.approx(0.5)


def test_accuracy_threshold_inclusive():
    assert accuracy([0.5], threshold=0.5) == pytest.approx(1.0)


def test_pass_at_k_all_correct():
    assert pass_at_k(10, 10, 3) == pytest.approx(1.0)


def test_pass_at_k_none_correct():
    assert pass_at_k(10, 0, 1) == pytest.approx(0.0)


def test_pass_at_k_known_value():
    # 1 - C(2,1)/C(4,1) = 0.5
    assert pass_at_k(4, 2, 1) == pytest.approx(0.5)


def test_pass_at_k_known_value_2():
    # 1 - C(2,2)/C(5,2) = 0.9
    assert pass_at_k(5, 3, 2) == pytest.approx(0.9)


def test_pass_at_k_k_exceeds_wrong_count():
    assert pass_at_k(5, 4, 3) == pytest.approx(1.0)


def test_pass_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        pass_at_k(5, 3, 0)


def test_pass_at_k_rejects_bad_n():
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)


def test_pass_at_k_rejects_bad_c():
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, -1, 1)
