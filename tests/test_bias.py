"""Tests for the pairwise judge bias probes.

The stub judge is deterministic and coverage-driven, so on reference-based
pairs with clearly different coverage it must show zero position bias and
zero verbosity bias. Rigged judges prove the probes fail loudly when bias
is actually present.
"""

import pytest

from evals.judges import (
    MAX_ACCEPTABLE_BIAS_RATE,
    POSITION_PAIRS,
    VERBOSITY_PAIRS,
    BiasPair,
    BiasReport,
    OfflineStubJudge,
    PairwiseJudge,
    position_swap_probe,
    verbosity_probe,
    wilson_interval,
)
from evals.judges.probe_sets import FACTS, FILLER
from evals.judges.stub import _coverage, _keywords

STUB = OfflineStubJudge()
JUDGE = PairwiseJudge(STUB.pairwise_prefer)

# Seeded 40-item fixtures (evals/judges/probe_sets.py): POSITION_PAIRS have
# clearly different reference coverage; VERBOSITY_PAIRS have equal coverage
# and differ only in length.


def _always_first(prompt, a, b, ref):
    return "a"


def _always_longer(prompt, a, b, ref):
    return "a" if len(a) >= len(b) else "b"


# ---------- position swap ----------

def test_position_swap_stub_not_biased():
    report = position_swap_probe(JUDGE, POSITION_PAIRS)
    assert report.bias_rate == 0.0
    assert report.pairs == len(POSITION_PAIRS)
    assert report.passes
    assert "position-swap" in report.summary()


def test_position_swap_catches_always_first_judge():
    rigged = PairwiseJudge(_always_first)
    report = position_swap_probe(rigged, POSITION_PAIRS)
    # Fails loudly: every pair follows the slot, not the content.
    assert report.bias_rate == 1.0
    assert not report.passes
    assert all(d.biased for d in report.details)


def test_position_swap_identical_responses_not_biased():
    pairs = [BiasPair("q", "same text here", "same text here", "same text")]
    report = position_swap_probe(JUDGE, pairs)
    assert report.bias_rate == 0.0
    assert report.passes


def test_position_swap_empty_pairs():
    report = position_swap_probe(JUDGE, [])
    assert report.pairs == 0
    assert report.bias_rate == 0.0
    assert report.passes


# ---------- verbosity ----------

def test_verbosity_stub_neutral_with_reference():
    report = verbosity_probe(JUDGE, VERBOSITY_PAIRS)
    assert report.bias_rate == 0.0
    assert report.passes


def test_verbosity_probe_catches_longer_bias():
    rigged = PairwiseJudge(_always_longer)
    report = verbosity_probe(rigged, VERBOSITY_PAIRS)
    assert report.bias_rate == 1.0
    assert not report.passes


def test_verbosity_probe_catches_stub_without_reference():
    # Documents real behavior: the stub's no-reference fallback prefers
    # the longer answer, and the probe flags it.
    pairs = [BiasPair("q", p.response_a, p.response_b)
             for p in VERBOSITY_PAIRS]
    report = verbosity_probe(JUDGE, pairs)
    assert report.bias_rate == 1.0
    assert not report.passes


def test_verbosity_equal_length_not_applicable():
    pairs = [BiasPair("q", "abc", "def", "abc")]
    report = verbosity_probe(JUDGE, pairs)
    assert report.bias_rate == 0.0
    assert "not applicable" in report.details[0].note


# ---------- report semantics ----------

def test_threshold_is_documented_and_shared():
    assert MAX_ACCEPTABLE_BIAS_RATE == 0.10
    assert BiasReport(probe="x", pairs=10, biased_pairs=1).passes
    assert not BiasReport(probe="x", pairs=10, biased_pairs=2).passes


def test_summary_marks_pass_and_fail():
    ok = BiasReport(probe="position-swap", pairs=40, biased_pairs=0)
    bad = BiasReport(probe="position-swap", pairs=40, biased_pairs=40)
    assert ok.summary().endswith("PASS")
    assert bad.summary().endswith("FAIL")


# ---------- order consistency and attribution ----------

def test_verbosity_always_first_is_order_flip_not_length_bias():
    # Old single-order probe blamed slot preference on length; now it is an order flip.
    report = verbosity_probe(PairwiseJudge(_always_first), VERBOSITY_PAIRS)
    assert report.biased_pairs == 0
    assert report.order_flips == report.pairs == len(VERBOSITY_PAIRS)
    assert report.order_flip_rate == 1.0
    assert not report.passes  # still unfit, for the right reason
    assert all("not attributed to length" in d.note for d in report.details)


def test_verbosity_equal_length_excluded_from_denominator():
    mixed = VERBOSITY_PAIRS[:10] + [BiasPair("q", "abc", "def", "abc")] * 10
    report = verbosity_probe(PairwiseJudge(_always_longer), mixed)
    assert report.pairs == 10 and report.not_applicable == 10
    assert report.bias_rate == 1.0  # previously diluted to 0.50
    assert "10 n/a" in report.summary()


def test_verbosity_shorter_preference_is_not_length_bias():
    def always_shorter(prompt, a, b, ref):
        return "a" if len(a) < len(b) else "b"

    report = verbosity_probe(PairwiseJudge(always_shorter), VERBOSITY_PAIRS)
    assert report.biased_pairs == 0 and report.order_flips == 0
    assert all("shorter" in d.note for d in report.details)


def test_position_probe_counts_tie_win_mix_as_order_flip():
    def credits_correct_only_in_slot_a(prompt, a, b, ref):
        return "a" if ref in a.lower() else "tie"

    report = position_swap_probe(PairwiseJudge(credits_correct_only_in_slot_a), POSITION_PAIRS[:4])
    assert report.biased_pairs == 0
    assert report.order_flips == 4
    assert not report.passes


def test_stub_has_no_order_flips_on_either_probe():
    for probe, pairs in ((position_swap_probe, POSITION_PAIRS), (verbosity_probe, VERBOSITY_PAIRS)):
        report = probe(JUDGE, pairs)
        assert report.order_flips == 0 and report.biased_pairs == 0


# ---------- probe set size and confidence ----------

def test_probe_sets_are_large_enough_to_be_conclusive():
    assert len(POSITION_PAIRS) == len(VERBOSITY_PAIRS) == 40
    for probe, pairs in ((position_swap_probe, POSITION_PAIRS), (verbosity_probe, VERBOSITY_PAIRS)):
        report = probe(JUDGE, pairs)
        assert report.confidently_passes
        assert report.bias_ci[1] < MAX_ACCEPTABLE_BIAS_RATE


def test_small_clean_set_passes_but_not_confidently():
    report = position_swap_probe(JUDGE, POSITION_PAIRS[:6])
    assert report.passes and not report.confidently_passes
    assert report.bias_ci[1] > 0.3


def test_probe_set_construction_is_sound():
    for prompt, ref, good, bad in FACTS:
        assert _coverage(good, ref) == 1.0 > _coverage(bad, ref)
        for f in FILLER:
            assert not _keywords(f) & _keywords(ref), (prompt, f)
    # Correct / longer answer alternates between slots.
    assert [p.response_a == FACTS[i][2] for i, p in enumerate(POSITION_PAIRS)][:4] == [True, False, True, False]
    assert [len(p.response_a) > len(p.response_b) for p in VERBOSITY_PAIRS][:4] == [True, False, True, False]


@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 40, 0.0, 0.0876),
    (0, 6, 0.0, 0.3903),
    (1, 10, 0.0179, 0.4042),
    (5, 10, 0.2366, 0.7634),
    (40, 40, 0.9124, 1.0),
])
def test_wilson_interval_known_values(k, n, lo, hi):
    got = wilson_interval(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-4)
    assert got[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_interval_edge_cases():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_summary_reports_ci_and_flips():
    text = position_swap_probe(JUDGE, POSITION_PAIRS).summary()
    assert "95% CI 0.00-0.09" in text and "order flips 0/40" in text
