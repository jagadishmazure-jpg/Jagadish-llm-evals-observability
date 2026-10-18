"""Tests for the pairwise judge bias probes.

The stub judge is deterministic and coverage-driven, so on reference-based
pairs with clearly different coverage it must show zero position bias and
zero verbosity bias. Rigged judges prove the probes fail loudly when bias
is actually present.
"""

from evals.judges import (
    MAX_ACCEPTABLE_BIAS_RATE,
    BiasPair,
    BiasReport,
    OfflineStubJudge,
    PairwiseJudge,
    position_swap_probe,
    verbosity_probe,
)

STUB = OfflineStubJudge()
JUDGE = PairwiseJudge(STUB.pairwise_prefer)

# Seeded fixture: response_a and response_b have clearly different
# reference coverage, so the honest judge's verdict tracks content.
POSITION_PAIRS = [
    BiasPair("summarize", "the cat sat on the mat and slept soundly",
             "the cat", "the cat sat on the mat"),
    BiasPair("capital", "london is big", "paris is the capital of france",
             "paris"),
    BiasPair("color", "the sky is blue today", "the ocean is blue", "blue"),
    BiasPair("animal", "dogs bark loudly at night", "cats meow",
             "dogs bark"),
    BiasPair("food", "pizza", "pizza with cheese and pepperoni toppings",
             "pizza cheese"),
    BiasPair("sport", "tennis is played with rackets", "chess",
             "tennis rackets"),
]

# Seeded fixture: equal reference coverage, different lengths. A fair
# judge ties these; a verbosity-biased judge picks the longer one.
VERBOSITY_PAIRS = [
    BiasPair("capital", "Paris is the capital of France and a beautiful city",
             "Paris", "paris"),
    BiasPair("summarize", "the cat sat",
             "the cat sat on the mat purring loudly", "the cat sat"),
    BiasPair("color", "blue", "the color blue is really nice", "blue"),
]


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
    ok = BiasReport(probe="position-swap", pairs=6, biased_pairs=0)
    bad = BiasReport(probe="position-swap", pairs=6, biased_pairs=6)
    assert ok.summary().endswith("PASS")
    assert bad.summary().endswith("FAIL")
