# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.benchmark."""

import pytest

from opensearch_genai_observability_sdk_py.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSummary,
    EvalScore,
    _make_case_id,
    evaluate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRACE_ID = "6ebb9835f43af1552f2cebb9f5165e39"
SPAN_ID = "89829115c2128845"


def _get_spans_by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _get_root_span(exporter, benchmark_name):
    return next(
        s for s in exporter.get_finished_spans() if s.name == f"test_suite_run {benchmark_name}"
    )


def _get_case_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.name == "test_case"]


# ---------------------------------------------------------------------------
# _make_case_id
# ---------------------------------------------------------------------------


class TestMakeCaseId:
    def test_deterministic(self):
        assert _make_case_id("hello") == _make_case_id("hello")

    def test_different_inputs_different_ids(self):
        assert _make_case_id("hello") != _make_case_id("world")

    def test_length_is_16(self):
        assert len(_make_case_id("test")) == 16

    def test_dict_input(self):
        id1 = _make_case_id({"a": 1, "b": 2})
        id2 = _make_case_id({"b": 2, "a": 1})
        assert id1 == id2  # sort_keys=True

    def test_none_input(self):
        result = _make_case_id(None)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# EvalScore dataclass
# ---------------------------------------------------------------------------


class TestEvalScore:
    def test_minimal(self):
        s = EvalScore(name="accuracy", value=0.95)
        assert s.name == "accuracy"
        assert s.value == 0.95
        assert s.label is None
        assert s.explanation is None
        assert s.metadata is None

    def test_full(self):
        s = EvalScore(
            name="relevance",
            value=0.8,
            label="good",
            explanation="Relevant answer",
            metadata={"model": "gpt-4"},
        )
        assert s.label == "good"
        assert s.explanation == "Relevant answer"
        assert s.metadata == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# Benchmark — root span structure
# ---------------------------------------------------------------------------


class TestBenchmarkRootSpan:
    def test_creates_root_span(self, exporter):
        with Benchmark(name="test-bench"):
            pass
        root = _get_root_span(exporter, "test-bench")
        assert root is not None

    def test_root_span_attributes(self, exporter):
        with Benchmark(name="my-bench"):
            pass
        root = _get_root_span(exporter, "my-bench")
        assert root.attributes["test.suite.name"] == "my-bench"
        assert root.attributes["test.suite.run.id"].startswith("run_")
        assert root.attributes["gen_ai.operation.name"] == "evaluation"

    def test_root_span_success_status(self, exporter):
        with Benchmark(name="ok-bench") as b:
            b.log(scores={"a": 1.0})
        root = _get_root_span(exporter, "ok-bench")
        assert root.attributes["test.suite.run.status"] == "success"

    def test_root_span_failure_status(self, exporter):
        with Benchmark(name="fail-bench") as b:
            b.log(error="boom")
        root = _get_root_span(exporter, "fail-bench")
        assert root.attributes["test.suite.run.status"] == "failure"

    def test_metadata_on_root_span(self, exporter):
        with Benchmark(name="meta-bench", metadata={"model": "gpt-4"}):
            pass
        root = _get_root_span(exporter, "meta-bench")
        assert root.attributes["model"] == "gpt-4"

    def test_reserved_metadata_filtered(self, exporter):
        with Benchmark(
            name="reserved-bench",
            metadata={"test.suite.name": "oops", "model": "gpt-4"},
        ):
            pass
        root = _get_root_span(exporter, "reserved-bench")
        # Reserved key should not overwrite SDK attribute
        assert root.attributes["test.suite.name"] == "reserved-bench"
        # Non-reserved key should be present
        assert root.attributes["model"] == "gpt-4"

    def test_custom_test_prefix_allowed(self, exporter):
        """User metadata with test.* prefix is allowed if not an SDK key."""
        with Benchmark(
            name="custom-test",
            metadata={"test.custom_field": "hello"},
        ):
            pass
        root = _get_root_span(exporter, "custom-test")
        assert root.attributes["test.custom_field"] == "hello"

    def test_run_id_property(self):
        b = Benchmark(name="prop-test")
        assert b.run_id.startswith("run_")
        assert b.name == "prop-test"
        b.close()


# ---------------------------------------------------------------------------
# Benchmark.log() — case spans
# ---------------------------------------------------------------------------


class TestBenchmarkLog:
    def test_creates_case_span(self, exporter):
        with Benchmark(name="log-test") as b:
            b.log(input="q1", scores={"a": 1.0})
        cases = _get_case_spans(exporter)
        assert len(cases) == 1

    def test_case_span_attributes(self, exporter):
        with Benchmark(name="attr-test") as b:
            b.log(
                input="What is Python?",
                scores={"accuracy": 0.9},
                case_name="python_q",
            )
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.suite.name"] == "attr-test"
        assert case.attributes["test.suite.run.id"].startswith("run_")
        assert case.attributes["test.case.id"] == _make_case_id("What is Python?")
        assert case.attributes["test.case.name"] == "python_q"
        assert case.attributes["test.case.result.status"] == "pass"

    def test_explicit_case_id(self, exporter):
        with Benchmark(name="id-test") as b:
            b.log(case_id="custom_123", scores={"a": 1.0})
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.id"] == "custom_123"

    def test_auto_case_id_from_input(self, exporter):
        with Benchmark(name="auto-id") as b:
            b.log(input="hello", scores={"a": 1.0})
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.id"] == _make_case_id("hello")

    def test_error_sets_fail_status(self, exporter):
        with Benchmark(name="err-test") as b:
            b.log(input="q1", error="something broke")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.result.status"] == "fail"

    def test_multiple_cases(self, exporter):
        with Benchmark(name="multi") as b:
            b.log(input="q1", scores={"a": 1.0})
            b.log(input="q2", scores={"a": 0.5})
            b.log(input="q3", scores={"a": 0.0})
        cases = _get_case_spans(exporter)
        assert len(cases) == 3

    def test_case_metadata(self, exporter):
        with Benchmark(name="meta") as b:
            b.log(input="q1", scores={"a": 1.0}, metadata={"model": "gpt-4"})
        case = _get_case_spans(exporter)[0]
        assert case.attributes["model"] == "gpt-4"

    def test_case_reserved_metadata_filtered(self, exporter):
        with Benchmark(name="res-meta") as b:
            b.log(
                input="q1",
                scores={"a": 1.0},
                metadata={"gen_ai.operation.name": "oops", "model": "gpt-4"},
            )
        case = _get_case_spans(exporter)[0]
        # Reserved key should not overwrite SDK attribute
        assert case.attributes["gen_ai.operation.name"] == "evaluation"
        assert case.attributes["model"] == "gpt-4"

    def test_log_after_close_raises(self):
        b = Benchmark(name="closed")
        b.close()
        with pytest.raises(RuntimeError, match="already closed"):
            b.log(input="q1")

    def test_double_close_raises(self):
        b = Benchmark(name="double")
        b.close()
        with pytest.raises(RuntimeError, match="already closed"):
            b.close()


# ---------------------------------------------------------------------------
# Benchmark.log() — score events
# ---------------------------------------------------------------------------


class TestBenchmarkScoreEvents:
    def test_score_events_emitted(self, exporter):
        with Benchmark(name="evt-test") as b:
            b.log(input="q1", scores={"accuracy": 0.9, "relevance": 0.8})
        case = _get_case_spans(exporter)[0]
        events = [e for e in case.events if e.name == "gen_ai.evaluation.result"]
        assert len(events) == 2

    def test_score_event_attributes(self, exporter):
        with Benchmark(name="evt-attr") as b:
            b.log(input="q1", scores={"accuracy": 0.95})
        case = _get_case_spans(exporter)[0]
        event = case.events[0]
        assert event.attributes["gen_ai.evaluation.name"] == "accuracy"
        assert event.attributes["gen_ai.evaluation.score.value"] == 0.95

    def test_no_scores_no_events(self, exporter):
        with Benchmark(name="no-score") as b:
            b.log(input="q1")
        case = _get_case_spans(exporter)[0]
        events = [e for e in case.events if e.name == "gen_ai.evaluation.result"]
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Benchmark.log() — record_io
# ---------------------------------------------------------------------------


class TestBenchmarkRecordIO:
    def test_record_io_false_by_default(self, exporter):
        with Benchmark(name="no-io") as b:
            b.log(input="secret", output="answer", expected="truth")
        case = _get_case_spans(exporter)[0]
        assert "test.case.input" not in case.attributes
        assert "test.case.output" not in case.attributes
        assert "test.case.expected" not in case.attributes

    def test_record_io_true(self, exporter):
        with Benchmark(name="with-io", record_io=True) as b:
            b.log(input="question", output="answer", expected="truth")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.input"] == '"question"'
        assert case.attributes["test.case.output"] == '"answer"'
        assert case.attributes["test.case.expected"] == '"truth"'

    def test_record_io_dict_input(self, exporter):
        with Benchmark(name="dict-io", record_io=True) as b:
            b.log(input={"q": "hello"}, output="world")
        case = _get_case_spans(exporter)[0]
        assert '"q"' in case.attributes["test.case.input"]

    def test_record_io_truncation_marker(self, exporter):
        long_input = "x" * 20_000
        with Benchmark(name="trunc-io", record_io=True) as b:
            b.log(input=long_input, output="short")
        case = _get_case_spans(exporter)[0]
        assert case.attributes["test.case.input"].endswith("...[truncated]")
        assert len(case.attributes["test.case.input"]) <= 10_000


# ---------------------------------------------------------------------------
# Benchmark.log() — span links
# ---------------------------------------------------------------------------


class TestBenchmarkSpanLinks:
    def test_trace_and_span_id_creates_link(self, exporter):
        with Benchmark(name="link-test") as b:
            b.log(input="q1", trace_id=TRACE_ID, span_id=SPAN_ID)
        case = _get_case_spans(exporter)[0]
        assert len(case.links) == 1
        link = case.links[0]
        assert format(link.context.trace_id, "032x") == TRACE_ID
        assert format(link.context.span_id, "016x") == SPAN_ID

    def test_trace_id_without_span_id_raises(self, exporter):
        with Benchmark(name="link-no-span") as b:
            with pytest.raises(ValueError, match="Both trace_id and span_id"):
                b.log(input="q1", trace_id=TRACE_ID)

    def test_span_id_without_trace_id_raises(self, exporter):
        with Benchmark(name="link-no-trace") as b:
            with pytest.raises(ValueError, match="Both trace_id and span_id"):
                b.log(input="q1", span_id=SPAN_ID)

    def test_no_trace_id_no_link(self, exporter):
        with Benchmark(name="no-link") as b:
            b.log(input="q1")
        case = _get_case_spans(exporter)[0]
        assert len(case.links) == 0

    def test_invalid_trace_id_no_link(self, exporter):
        with Benchmark(name="bad-link") as b:
            b.log(input="q1", trace_id="not-hex!", span_id=SPAN_ID)
        case = _get_case_spans(exporter)[0]
        assert len(case.links) == 0


# ---------------------------------------------------------------------------
# Benchmark — case spans are children of root
# ---------------------------------------------------------------------------


class TestBenchmarkParentChild:
    def test_case_spans_are_children_of_root(self, exporter):
        with Benchmark(name="parent-test") as b:
            b.log(input="q1", scores={"a": 1.0})
            b.log(input="q2", scores={"a": 0.5})

        root = _get_root_span(exporter, "parent-test")
        cases = _get_case_spans(exporter)

        for case in cases:
            assert case.parent is not None
            assert case.parent.span_id == root.context.span_id
            assert case.context.trace_id == root.context.trace_id


# ---------------------------------------------------------------------------
# Benchmark.close() — summary
# ---------------------------------------------------------------------------


class TestBenchmarkSummary:
    def test_close_returns_summary(self):
        b = Benchmark(name="sum-test")
        b.log(input="q1", scores={"accuracy": 1.0, "relevance": 0.8})
        b.log(input="q2", scores={"accuracy": 0.5, "relevance": 0.9})
        summary = b.close()

        assert isinstance(summary, BenchmarkSummary)
        assert summary.benchmark_name == "sum-test"
        assert summary.total_cases == 2
        assert summary.error_count == 0
        assert summary.scores["accuracy"].avg == 0.75
        assert summary.scores["accuracy"].min == 0.5
        assert summary.scores["accuracy"].max == 1.0
        assert summary.scores["accuracy"].count == 2
        assert summary.scores["relevance"].avg == pytest.approx(0.85)

    def test_summary_with_errors(self):
        b = Benchmark(name="err-sum")
        b.log(input="q1", scores={"a": 1.0})
        b.log(input="q2", error="failed")
        summary = b.close()
        assert summary.error_count == 1
        assert summary.total_cases == 2

    def test_summary_str_format(self):
        b = Benchmark(name="fmt-test")
        b.log(input="q1", scores={"accuracy": 0.9})
        summary = b.close()
        text = str(summary)
        assert "BENCHMARK: fmt-test" in text
        assert "accuracy" in text
        assert "avg=0.90" in text

    def test_prints_summary_to_stdout(self, capsys):
        """close() prints the benchmark summary for developer visibility."""
        with Benchmark(name="print-test") as b:
            b.log(input="q1", scores={"a": 1.0})
        captured = capsys.readouterr()
        assert "BENCHMARK: print-test" in captured.out


# ---------------------------------------------------------------------------
# evaluate() — full runner
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_basic_evaluate(self, exporter):
        def my_task(input_val):
            return f"answer to {input_val}"

        def accuracy(input_val, output, expected) -> EvalScore:
            return EvalScore(
                name="accuracy",
                value=1.0 if expected in output else 0.0,
            )

        result = evaluate(
            name="eval-test",
            task=my_task,
            data=[
                {"input": "q1", "expected": "q1"},
                {"input": "q2", "expected": "q2"},
            ],
            scores=[accuracy],
        )

        assert isinstance(result, BenchmarkResult)
        assert result.summary.total_cases == 2
        assert result.summary.scores["accuracy"].avg == 1.0
        assert len(result.cases) == 2

    def test_evaluate_creates_spans(self, exporter):
        evaluate(
            name="span-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
        )

        root = _get_root_span(exporter, "span-eval")
        assert root is not None
        assert root.attributes["test.suite.name"] == "span-eval"

        case_spans = _get_case_spans(exporter)
        assert len(case_spans) == 1

    def test_evaluate_task_error(self, exporter):
        def failing_task(x):
            raise ValueError("boom")

        result = evaluate(
            name="err-eval",
            task=failing_task,
            data=[{"input": "q1"}],
            scores=[],
        )

        assert result.summary.error_count == 1
        assert result.cases[0].status == "fail"
        assert result.cases[0].error == "boom"

    def test_evaluate_scorer_returns_float(self, exporter):
        def my_scorer(input_val, output, expected):
            return 0.75

        result = evaluate(
            name="float-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[my_scorer],
        )

        assert result.cases[0].scores["my_scorer"] == 0.75

    def test_evaluate_scorer_returns_list(self, exporter):
        def multi_scorer(input_val, output, expected):
            return [
                EvalScore(name="a", value=0.9),
                EvalScore(name="b", value=0.8),
            ]

        result = evaluate(
            name="list-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[multi_scorer],
        )

        assert result.cases[0].scores == {"a": 0.9, "b": 0.8}

    def test_evaluate_record_io(self, exporter):
        evaluate(
            name="io-eval",
            task=lambda x: f"answer: {x}",
            data=[{"input": "question", "expected": "answer"}],
            scores=[],
            record_io=True,
        )

        case_spans = _get_case_spans(exporter)
        case = case_spans[0]
        assert case.attributes["test.case.input"] == '"question"'
        assert case.attributes["test.case.expected"] == '"answer"'
        assert '"answer: question"' in case.attributes["test.case.output"]

    def test_evaluate_case_id_from_data(self, exporter):
        evaluate(
            name="cid-eval",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "my_custom_id"}],
            scores=[],
        )

        case_spans = _get_case_spans(exporter)
        assert case_spans[0].attributes["test.case.id"] == "my_custom_id"

    def test_evaluate_agent_spans_are_children(self, exporter):
        """Agent spans created inside task() should be children of the case span."""
        from opensearch_genai_observability_sdk_py import observe

        @observe(name="my_agent")
        def traced_task(x):
            return f"result: {x}"

        evaluate(
            name="child-eval",
            task=traced_task,
            data=[{"input": "q1"}],
            scores=[],
        )

        spans = exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name == "my_agent")
        case_span = next(s for s in spans if s.name == "test_case")

        # Agent span should be a child of the case span
        assert agent_span.parent is not None
        assert agent_span.parent.span_id == case_span.context.span_id
        assert agent_span.context.trace_id == case_span.context.trace_id

    def test_evaluate_prints_summary_to_stdout(self, capsys):
        """evaluate() prints the summary for developer visibility."""
        evaluate(
            name="print-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
        )
        captured = capsys.readouterr()
        assert "BENCHMARK: print-eval" in captured.out

    def test_evaluate_scorer_error_surfaces(self, exporter):
        """A failing scorer should set status to fail and be in scorer_errors."""

        def bad_scorer(input_val, output, expected):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[bad_scorer],
        )

        assert result.cases[0].status == "fail"
        assert result.cases[0].scorer_errors is not None
        assert any("scorer broke" in e for e in result.cases[0].scorer_errors)

    def test_evaluate_accepts_generator(self, exporter):
        """data can be a generator (single-pass iterable)."""

        def gen():
            yield {"input": "q1"}
            yield {"input": "q2"}

        result = evaluate(
            name="gen-eval",
            task=lambda x: x,
            data=gen(),
            scores=[],
        )

        assert result.summary.total_cases == 2

    def test_evaluate_reserved_metadata_filtered(self, exporter):
        evaluate(
            name="res-meta-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
            metadata={"test.suite.name": "oops", "model": "gpt-4"},
        )

        root = _get_root_span(exporter, "res-meta-eval")
        assert root.attributes["test.suite.name"] == "res-meta-eval"
        assert root.attributes["model"] == "gpt-4"
