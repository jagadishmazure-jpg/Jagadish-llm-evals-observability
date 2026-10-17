"""Tests for dataset loading and the regression suite runner."""

import pytest

from evals.judges import OfflineStubJudge
from evals.suites import (
    CaseResult,
    DatasetError,
    EvalCase,
    ModelOutput,
    RegressionSuite,
    load_dataset,
)

STUB = OfflineStubJudge()


def echo_model(case):
    return ModelOutput(response=case.expected or "", prompt_tokens=5, completion_tokens=3)


# --- dataset loading ---


def test_load_jsonl(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text(
        '{"id": "a", "input": "q1", "expected": "r1"}\n'
        '{"id": "b", "input": "q2"}\n'
        "\n"  # blank lines are skipped
    )
    cases = load_dataset(f)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[0].expected == "r1"
    assert cases[1].expected is None


def test_load_json(tmp_path):
    f = tmp_path / "data.json"
    f.write_text('[{"id": "a", "input": "q", "metadata": {"k": "v"}}]')
    cases = load_dataset(f)
    assert cases[0].metadata == {"k": "v"}


def test_load_missing_file(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope.jsonl")


def test_load_bad_extension(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("id\n")
    with pytest.raises(DatasetError):
        load_dataset(f)


def test_load_invalid_jsonl_line(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"id": "a", "input": "q"}\nnot json\n')
    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(f)


def test_load_row_missing_id(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"input": "q"}\n')
    with pytest.raises(DatasetError, match="'id'"):
        load_dataset(f)


def test_load_row_missing_input(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"id": "a"}\n')
    with pytest.raises(DatasetError, match="'input'"):
        load_dataset(f)


def test_load_duplicate_ids(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"id": "a", "input": "q"}\n{"id": "a", "input": "q"}\n')
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(f)


def test_load_json_must_be_list(tmp_path):
    f = tmp_path / "data.json"
    f.write_text('{"id": "a"}')
    with pytest.raises(DatasetError, match="list"):
        load_dataset(f)


def test_load_bad_metadata_type(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"id": "a", "input": "q", "metadata": [1]}\n')
    with pytest.raises(DatasetError, match="metadata"):
        load_dataset(f)


# --- regression suite runner ---


def test_suite_all_pass():
    cases = [EvalCase("1", "q", "answer"), EvalCase("2", "q", "answer")]
    result = RegressionSuite(cases, echo_model, judges=[STUB]).run()
    assert result.total == 2
    assert result.passed == 2
    assert result.failed == 0
    assert result.pass_rate == pytest.approx(1.0)
    assert result.failures() == []


def test_suite_all_fail_on_bad_answers():
    cases = [EvalCase("1", "q", "the cat sat")]
    result = RegressionSuite(
        cases, lambda c: ModelOutput(response="something unrelated"), judges=[STUB]
    ).run()
    assert result.passed == 0
    assert result.failed == 1
    assert result.pass_rate == pytest.approx(0.0)
    assert len(result.failures()) == 1


def test_suite_catches_model_exceptions():
    def boom(case):
        raise RuntimeError("model exploded")

    cases = [EvalCase("1", "q", "x")]
    result = RegressionSuite(cases, boom, judges=[STUB]).run()
    assert result.failed == 1
    failure = result.results[0]
    assert "RuntimeError" in failure.error
    assert failure.output is None
    assert failure.judgment is None


def test_suite_records_model_error_field():
    cases = [EvalCase("1", "q", "x")]
    result = RegressionSuite(
        cases, lambda c: ModelOutput(response="", error="timeout"), judges=[STUB]
    ).run()
    assert result.failed == 1
    assert result.results[0].error == "timeout"


def test_suite_no_judges_passes_on_no_error():
    cases = [EvalCase("1", "q")]
    result = RegressionSuite(cases, lambda c: ModelOutput(response="anything")).run()
    assert result.passed == 1


def test_suite_aggregates_tokens():
    cases = [EvalCase("1", "q"), EvalCase("2", "q")]
    result = RegressionSuite(cases, echo_model).run()
    assert result.total_prompt_tokens == 10
    assert result.total_completion_tokens == 6


def test_suite_threshold_filters_scores():
    cases = [EvalCase("1", "q", "the cat sat on the mat")]
    def partial(c):
        return ModelOutput(response="the cat")  # coverage 0.4

    strict = RegressionSuite(cases, partial, judges=[STUB], pass_threshold=0.9).run()
    lenient = RegressionSuite(cases, partial, judges=[STUB], pass_threshold=0.1).run()
    assert strict.failed == 1
    assert lenient.passed == 1


def test_suite_rejects_bad_threshold():
    with pytest.raises(ValueError):
        RegressionSuite([], echo_model, pass_threshold=1.5)


def test_suite_tracks_latency():
    cases = [EvalCase("1", "q")]
    result = RegressionSuite(cases, echo_model).run()
    assert result.total_latency_ms >= 0.0
    assert result.results[0].latency_ms >= 0.0


def test_suite_exposes_cases():
    cases = [EvalCase("1", "q")]
    suite = RegressionSuite(cases, echo_model)
    assert suite.cases == cases


def test_case_result_shape():
    r = CaseResult(case_id="x", passed=True, judgment=None, output=None)
    assert r.error is None
    assert r.latency_ms == 0.0
