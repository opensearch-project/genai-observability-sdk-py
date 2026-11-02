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
from opentelemetry import trace

from opensearch_genai_observability_sdk_py._internal import SDK_VERSION
from opensearch_genai_observability_sdk_py.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSummary,
    EvalScore,
    _make_case_id,
    evaluate,
    scorer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRACE_ID = "6ebb9835f43af1552f2cebb9f5165e39"
SPAN_ID = "89829115c2128845"


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
# Benchmark: run, case and score events (no spans)
# ---------------------------------------------------------------------------

CASE_EVENT = "opensearch.test_case.result"
RUN_EVENT = "opensearch.test_suite_run.result"
EVALUATION = "gen_ai.evaluation.result"
SDK_SCOPE = "opensearch-genai-observability-sdk-py"


@pytest.fixture()
def events(log_exporter):
    """Return a callable listing emitted records with a given event name."""

    def _get(name):
        return [r for r in log_exporter.get_finished_logs() if r.log_record.event_name == name]

    return _get


class TestBenchmarkRun:
    def test_run_event_attributes(self, events):
        with Benchmark(name="my-bench") as b:
            pass
        a = events(RUN_EVENT)[0].log_record.attributes
        assert a["test.suite.name"] == "my-bench"
        assert a["test.suite.run.id"] == b.run_id
        assert b.run_id.startswith("run_")
        assert a["test.suite.run.status"] == "success"
        assert "gen_ai.operation.name" not in a

    def test_failure_status(self, events):
        with Benchmark(name="fail-bench") as b:
            b.log(error="boom")
        a = events(RUN_EVENT)[0].log_record.attributes
        assert a["test.suite.run.status"] == "failure"

    def test_metadata_on_run_event(self, events):
        with Benchmark(name="meta-bench", metadata={"model": "gpt-4"}):
            pass
        assert events(RUN_EVENT)[0].log_record.attributes["model"] == "gpt-4"

    def test_reserved_metadata_filtered(self, events):
        with Benchmark(name="reserved-bench", metadata={"test.suite.name": "oops", "m": "x"}):
            pass
        a = events(RUN_EVENT)[0].log_record.attributes
        assert a["test.suite.name"] == "reserved-bench"
        assert a["m"] == "x"

    def test_custom_test_prefix_allowed(self, events):
        """User metadata with test.* prefix is allowed if not an SDK key."""
        with Benchmark(name="custom-test", metadata={"test.custom_field": "hello"}):
            pass
        assert events(RUN_EVENT)[0].log_record.attributes["test.custom_field"] == "hello"

    def test_properties(self):
        b = Benchmark(name="prop-test")
        assert b.name == "prop-test"
        b.close()


class TestBenchmarkLog:
    def test_case_event_attributes(self, events):
        with Benchmark(name="attr-test") as b:
            b.log(input="What is Python?", scores={"accuracy": 0.9}, case_name="python_q")
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["test.suite.name"] == "attr-test"
        assert a["test.suite.run.id"] == b.run_id
        assert a["test.case.id"] == _make_case_id("What is Python?")
        assert a["test.case.name"] == "python_q"
        assert a["test.case.result.status"] == "pass"

    def test_explicit_case_id(self, events):
        with Benchmark(name="id-test") as b:
            b.log(case_id="custom_123", scores={"a": 1.0})
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.id"] == "custom_123"

    def test_error_is_error_event_and_fail_result(self, events):
        with Benchmark(name="err-test") as b:
            b.log(input="q1", error="something broke")
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["test.case.result.status"] == "error"
        # The error message is not recorded as an attribute
        assert "opensearch.test_case.error" not in a

    def test_multiple_cases(self, events):
        with Benchmark(name="multi") as b:
            b.log(input="q1", scores={"a": 1.0})
            b.log(input="q2", scores={"a": 0.5})
            b.log(input="q3", scores={"a": 0.0})
        assert len(events(CASE_EVENT)) == 3
        assert len(events(EVALUATION)) == 3

    def test_case_reserved_metadata_filtered(self, events):
        with Benchmark(name="res-meta") as b:
            b.log(
                input="q1",
                scores={"a": 1.0},
                metadata={"test.case.result.status": "oops", "model": "gpt-4"},
            )
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["test.case.result.status"] == "pass"
        assert a["model"] == "gpt-4"

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

    def test_trace_id_without_span_id_raises(self):
        with Benchmark(name="link-no-span") as b:
            with pytest.raises(ValueError, match="Both trace_id and span_id"):
                b.log(input="q1", trace_id=TRACE_ID)

    def test_span_id_without_trace_id_raises(self):
        with Benchmark(name="link-no-trace") as b:
            with pytest.raises(ValueError, match="Both trace_id and span_id"):
                b.log(input="q1", span_id=SPAN_ID)

    def test_invalid_ids_emit_without_trace_context(self, events):
        with Benchmark(name="bad-link") as b:
            b.log(input="q1", trace_id="not-hex!", span_id=SPAN_ID)
        lr = events(CASE_EVENT)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0


class TestBenchmarkScoreEvents:
    def test_one_log_event_per_score(self, exporter, eval_events):
        with Benchmark(name="evt-test") as b:
            b.log(input="q1", scores={"accuracy": 0.9, "relevance": 0.8})
        names = sorted(r.log_record.attributes["gen_ai.evaluation.name"] for r in eval_events())
        assert names == ["accuracy", "relevance"]
        assert exporter.get_finished_spans() == ()

    def test_grouping_attributes_copied(self, eval_events):
        with Benchmark(name="evt-group") as b:
            b.log(input="q1", case_id="c1", case_name="greeting", scores={"accuracy": 1.0})
        a = eval_events()[0].log_record.attributes
        assert a["test.suite.name"] == "evt-group"
        assert a["test.suite.run.id"] == b.run_id
        assert a["test.case.id"] == "c1"
        assert a["test.case.name"] == "greeting"
        assert not [k for k in a if k.startswith("opensearch.test_case.")]

    def test_event_parented_to_linked_agent_span(self, eval_events):
        with Benchmark(name="evt-link") as b:
            b.log(input="q1", scores={"accuracy": 1.0}, trace_id=TRACE_ID, span_id=SPAN_ID)
        lr = eval_events()[0].log_record
        assert format(lr.trace_id, "032x") == TRACE_ID
        assert format(lr.span_id, "016x") == SPAN_ID
        assert lr.attributes["opensearch.evaluation.level"] == "span"

    def test_no_scores_no_events(self, eval_events):
        with Benchmark(name="no-score") as b:
            b.log(input="q1")
        assert eval_events() == []


class TestBenchmarkRecordIO:
    def test_record_io_false_by_default(self, events):
        with Benchmark(name="no-io") as b:
            b.log(input="secret", output="answer", expected="truth")
        a = events(CASE_EVENT)[0].log_record.attributes
        assert not [k for k in a if k.startswith("opensearch.test_case.") and "duration" not in k]
        assert "test.case.input" not in a

    def test_record_io_true(self, events):
        with Benchmark(name="with-io", record_io=True) as b:
            b.log(input="question", output="answer", expected="truth")
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["opensearch.test_case.input"] == '"question"'
        assert a["opensearch.test_case.output"] == '"answer"'
        assert a["opensearch.test_case.expected"] == '"truth"'
        assert "test.case.input" not in a

    def test_record_io_dict_input(self, events):
        with Benchmark(name="dict-io", record_io=True) as b:
            b.log(input={"q": "hello"}, output="world")
        assert '"q"' in events(CASE_EVENT)[0].log_record.attributes["opensearch.test_case.input"]

    def test_record_io_truncation_marker(self, events):
        with Benchmark(name="trunc-io", record_io=True) as b:
            b.log(input="x" * 20_000, output="short")
        value = events(CASE_EVENT)[0].log_record.attributes["opensearch.test_case.input"]
        assert value.endswith("...[truncated]")
        assert len(value) <= 10_000


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
        text = str(b.close())
        assert "BENCHMARK: fmt-test" in text
        assert "accuracy" in text
        assert "avg=0.90" in text

    def test_prints_summary_to_stdout(self, capsys):
        """close() prints the benchmark summary for developer visibility."""
        with Benchmark(name="print-test") as b:
            b.log(input="q1", scores={"a": 1.0})
        assert "BENCHMARK: print-test" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Instrumentation scope of SDK events
# ---------------------------------------------------------------------------


class TestEventScope:
    """The scope is the instrumentation library (the SDK); no invented scope attributes."""

    def test_named_scorer_is_an_attribute_and_scope_is_sdk(self, events):
        def judge(i, o, e):
            return 1.0

        evaluate(name="scope", task=lambda x: x, data=[{"input": "q1"}], scores=[judge])
        grade = events(EVALUATION)[0]
        assert grade.log_record.attributes["opensearch.evaluation.evaluator.name"] == "judge"
        for rec in (grade, events(CASE_EVENT)[0], events(RUN_EVENT)[0]):
            assert rec.instrumentation_scope.name == SDK_SCOPE
            assert rec.instrumentation_scope.version == SDK_VERSION
            assert not rec.instrumentation_scope.attributes

    def test_anonymous_scorer_uses_sdk_scope(self, events):
        evaluate(name="anon", task=lambda x: x, data=[{"input": "q1"}], scores=[lambda i, o, e: 1])
        rec = events(EVALUATION)[0]
        assert rec.instrumentation_scope.name == SDK_SCOPE
        assert rec.instrumentation_scope.version == SDK_VERSION

    def test_benchmark_events_use_sdk_scope(self, events):
        with Benchmark(name="b-scope") as b:
            b.log(input="q1", scores={"a": 1.0})
        for name in (EVALUATION, CASE_EVENT, RUN_EVENT):
            rec = events(name)[0]
            assert rec.instrumentation_scope.name == SDK_SCOPE
            assert rec.instrumentation_scope.version == SDK_VERSION


# ---------------------------------------------------------------------------
# evaluate(): full runner
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_basic_evaluate(self):
        def my_task(input_val):
            return f"answer to {input_val}"

        def accuracy(input_val, output, expected) -> EvalScore:
            return EvalScore(name="accuracy", value=1.0 if expected in output else 0.0)

        result = evaluate(
            name="eval-test",
            task=my_task,
            data=[{"input": "q1", "expected": "q1"}, {"input": "q2", "expected": "q2"}],
            scores=[accuracy],
        )

        assert isinstance(result, BenchmarkResult)
        assert result.summary.total_cases == 2
        assert result.summary.scores["accuracy"].avg == 1.0
        assert len(result.cases) == 2

    def test_evaluate_task_error(self):
        def failing_task(x):
            raise ValueError("boom")

        result = evaluate(name="err-eval", task=failing_task, data=[{"input": "q1"}], scores=[])
        assert result.summary.error_count == 1
        assert result.cases[0].status == "fail"
        assert result.cases[0].error == "boom"

    def test_evaluate_scorer_returns_float(self):
        def my_scorer(input_val, output, expected):
            return 0.75

        result = evaluate(
            name="float-eval", task=lambda x: x, data=[{"input": "q1"}], scores=[my_scorer]
        )
        assert result.cases[0].scores["my_scorer"] == 0.75

    def test_evaluate_scorer_returns_list(self):
        def multi_scorer(input_val, output, expected):
            return [EvalScore(name="a", value=0.9), EvalScore(name="b", value=0.8)]

        result = evaluate(
            name="list-eval", task=lambda x: x, data=[{"input": "q1"}], scores=[multi_scorer]
        )
        assert result.cases[0].scores == {"a": 0.9, "b": 0.8}

    def test_evaluate_case_id_from_data(self, events):
        evaluate(
            name="cid-eval",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "my_custom_id"}],
            scores=[],
        )
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.id"] == "my_custom_id"

    def test_evaluate_prints_summary_to_stdout(self, capsys):
        evaluate(name="print-eval", task=lambda x: x, data=[{"input": "q1"}], scores=[])
        assert "BENCHMARK: print-eval" in capsys.readouterr().out

    def test_evaluate_scorer_error_surfaces(self):
        def bad_scorer(input_val, output, expected):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err", task=lambda x: x, data=[{"input": "q1"}], scores=[bad_scorer]
        )
        assert result.cases[0].status == "fail"
        assert result.cases[0].scorer_errors is not None
        assert any("scorer broke" in e for e in result.cases[0].scorer_errors)

    def test_evaluate_emits_events_with_label_and_explanation(self, eval_events):
        def judge(input_val, output, expected):
            return EvalScore(name="relevance", value=0.7, label="relevant", explanation="ok")

        evaluate(
            name="evt-eval",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "c1", "case_name": "first"}],
            scores=[judge],
        )
        rec = eval_events()[0]
        a = rec.log_record.attributes
        assert a["gen_ai.evaluation.name"] == "relevance"
        assert a["gen_ai.evaluation.score.label"] == "relevant"
        assert a["gen_ai.evaluation.explanation"] == "ok"
        assert a["test.suite.name"] == "evt-eval"
        assert a["test.case.id"] == "c1"
        assert a["test.case.name"] == "first"
        assert a["opensearch.evaluation.evaluator.name"] == "judge"
        assert rec.instrumentation_scope.name == SDK_SCOPE

    def test_evaluate_scorer_error_emits_error_event(self, eval_events):
        def bad_scorer(input_val, output, expected):
            raise RuntimeError("scorer broke")

        evaluate(name="err-evt", task=lambda x: x, data=[{"input": "q1"}], scores=[bad_scorer])
        a = eval_events()[0].log_record.attributes
        assert a["gen_ai.evaluation.name"] == "bad_scorer"
        assert a["error.type"] == "RuntimeError"
        assert "gen_ai.evaluation.score.value" not in a

    def test_evaluate_accepts_generator(self):
        def gen():
            yield {"input": "q1"}
            yield {"input": "q2"}

        result = evaluate(name="gen-eval", task=lambda x: x, data=gen(), scores=[])
        assert result.summary.total_cases == 2

    def test_evaluate_reserved_metadata_filtered(self, events):
        evaluate(
            name="res-meta-eval",
            task=lambda x: x,
            data=[{"input": "q1"}],
            scores=[],
            metadata={"test.suite.name": "oops", "model": "gpt-4"},
        )
        a = events(RUN_EVENT)[0].log_record.attributes
        assert a["test.suite.name"] == "res-meta-eval"
        assert a["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Offline runs: agent trace + case and run events
# ---------------------------------------------------------------------------

GROUPING = ("test.suite.name", "test.suite.run.id", "test.case.id", "test.case.name")


def _harness_spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.name.startswith("test_suite_run ")
        or s.name.startswith("test_case ")
        or s.name.startswith("evaluation ")
    ]


def _traced_agent():
    from opensearch_genai_observability_sdk_py import observe

    @observe(name="agent")
    def agent(x):
        return _tool(x)

    @observe(name="tool")
    def _tool(x):
        if x == "boom":
            raise ValueError("task broke")
        return f"answer: {x}"

    return agent


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


class TestEvaluateEvents:
    def test_no_harness_spans(self, exporter):
        evaluate(
            name="no-harness",
            task=_traced_agent(),
            data=[{"input": "q1"}, {"input": "q2"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert _harness_spans(exporter) == []
        agents = _spans_named(exporter, "agent")
        assert len(agents) == 2
        assert all(a.parent is None for a in agents)
        # Each case's agent run is its own trace
        assert agents[0].context.trace_id != agents[1].context.trace_id

    def test_root_span_stamped_children_not(self, exporter):
        result = evaluate(
            name="stamp",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1", "case_name": "first"}],
            scores=[],
        )
        root = _spans_named(exporter, "agent")[0]
        child = _spans_named(exporter, "tool")[0]
        assert root.attributes["test.suite.name"] == "stamp"
        assert root.attributes["test.suite.run.id"] == result.summary.run_id
        assert root.attributes["test.case.id"] == "c1"
        assert root.attributes["test.case.name"] == "first"
        assert child.parent.span_id == root.context.span_id
        assert not [k for k in child.attributes if k.startswith("test.")]

    def test_case_name_omitted_when_absent(self, exporter, events):
        evaluate(name="no-name", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        root = _spans_named(exporter, "agent")[0]
        assert "test.case.name" not in root.attributes
        assert "test.case.name" not in events(CASE_EVENT)[0].log_record.attributes

    def test_events_parented_to_agent_root(self, exporter, events):
        def judge(i, o, e):
            return EvalScore(name="relevance", value=0.7, label="relevant", explanation="ok")

        result = evaluate(
            name="parent",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1", "case_name": "first"}],
            scores=[judge],
        )
        root = _spans_named(exporter, "agent")[0]
        score_rec = events("gen_ai.evaluation.result")[0]
        case_rec = events(CASE_EVENT)[0]
        for rec in (score_rec, case_rec):
            assert rec.log_record.trace_id == root.context.trace_id
            assert rec.log_record.span_id == root.context.span_id
        a = score_rec.log_record.attributes
        for key in GROUPING:
            assert a[key] == root.attributes[key]
        assert a["opensearch.evaluation.level"] == "span"
        assert a["gen_ai.evaluation.score.label"] == "relevant"
        assert score_rec.instrumentation_scope.name == SDK_SCOPE
        case = result.cases[0]
        assert case.trace_id == format(root.context.trace_id, "032x")
        assert case.span_id == format(root.context.span_id, "016x")

    def test_case_event_shape(self, events):
        evaluate(
            name="shape",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
            metadata={"model": "gpt-4", "test.suite.name": "oops"},
        )
        rec = events(CASE_EVENT)[0]
        lr = rec.log_record
        a = lr.attributes
        assert a["test.suite.name"] == "shape"
        assert a["test.case.id"] == "c1"
        assert a["test.case.result.status"] == "pass"
        assert "error.type" not in a
        assert isinstance(a["opensearch.test_case.duration_ms"], float)
        assert a["model"] == "gpt-4"
        # record_io is off by default
        assert not [
            k for k in a if k in ("opensearch.test_case.input", "opensearch.test_case.output")
        ]
        assert lr.timestamp is not None
        assert lr.severity_number.value == 9
        assert rec.instrumentation_scope.name == "opensearch-genai-observability-sdk-py"

    def test_case_event_record_io(self, events):
        evaluate(
            name="io",
            task=_traced_agent(),
            data=[{"input": "question", "expected": "answer"}],
            scores=[],
            record_io=True,
        )
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["opensearch.test_case.input"] == '"question"'
        assert a["opensearch.test_case.expected"] == '"answer"'
        assert a["opensearch.test_case.output"] == '"answer: question"'

    def test_crashed_case_gets_error_event_and_run_continues(self, exporter, events):
        result = evaluate(
            name="crash",
            task=_traced_agent(),
            data=[{"input": "boom", "case_id": "bad"}, {"input": "q2", "case_id": "good"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert [c.case_id for c in result.cases] == ["bad", "good"]
        assert result.cases[0].error == "task broke"
        assert result.cases[0].status == "fail"  # historical value
        assert result.cases[0].error_type == "ValueError"

        by_case = {
            r.log_record.attributes["test.case.id"]: r.log_record for r in events(CASE_EVENT)
        }
        bad = by_case["bad"]
        assert bad.attributes["test.case.result.status"] == "error"
        assert bad.attributes["error.type"] == "ValueError"
        # Still parented to the crashed agent run's root span
        bad_root = next(
            s for s in _spans_named(exporter, "agent") if s.attributes["test.case.id"] == "bad"
        )
        assert bad.span_id == bad_root.context.span_id
        assert by_case["good"].attributes["test.case.result.status"] == "pass"
        # Scorers are skipped for the crashed case only
        scored = [r.log_record.attributes["test.case.id"] for r in events(EVALUATION)]
        assert scored == ["good"]

    def test_scorer_error_marks_case_error(self, events):
        def bad_scorer(i, o, e):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err-evt",
            task=_traced_agent(),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0, bad_scorer],
        )
        assert result.cases[0].status == "fail"
        assert result.cases[0].scorer_errors
        a = events(CASE_EVENT)[0].log_record.attributes
        assert a["test.case.result.status"] == "error"
        assert a["error.type"] == "RuntimeError"
        failed = [
            r.log_record.attributes
            for r in events(EVALUATION)
            if r.log_record.attributes["gen_ai.evaluation.name"] == "bad_scorer"
        ]
        assert failed[0]["error.type"] == "RuntimeError"
        run = events(RUN_EVENT)[0].log_record.attributes
        assert run["test.suite.run.status"] == "failure"
        assert run["opensearch.test_suite_run.error_count"] == 1

    def test_untraced_task_events_without_trace_context(self, exporter, events):
        result = evaluate(
            name="untraced",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert exporter.get_finished_spans() == ()
        case_lr = events(CASE_EVENT)[0].log_record
        score_lr = events(EVALUATION)[0].log_record
        for lr in (case_lr, score_lr):
            assert lr.trace_id == 0 and lr.span_id == 0
            assert lr.attributes["test.case.id"] == "c1"
            assert lr.attributes["test.suite.name"] == "untraced"
        assert "opensearch.evaluation.level" not in score_lr.attributes
        assert result.cases[0].trace_id is None

    def test_untraced_task_ignores_ambient_span(self, events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("outer"):
            evaluate(name="ambient", task=lambda x: x, data=[{"input": "q1"}], scores=[])
        lr = events(CASE_EVENT)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0

    def test_agent_run_is_own_trace_inside_active_span(self, exporter):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("outer") as outer:
            evaluate(name="nested", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        agent = _spans_named(exporter, "agent")[0]
        assert agent.parent is None
        assert agent.context.trace_id != outer.get_span_context().trace_id
        assert agent.attributes["test.suite.name"] == "nested"

    def test_run_event_counts(self, events):
        def scorer(i, o, e):
            if i == "q3":
                raise RuntimeError("nope")
            return 1.0

        result = evaluate(
            name="counts",
            task=_traced_agent(),
            data=[{"input": "q1"}, {"input": "boom"}, {"input": "q3"}, {"input": "q4"}],
            scores=[scorer],
            metadata={"agent_version": "v2"},
        )
        recs = events(RUN_EVENT)
        assert len(recs) == 1
        lr = recs[0].log_record
        a = lr.attributes
        assert lr.trace_id == 0 and lr.span_id == 0
        assert a["test.suite.name"] == "counts"
        assert a["test.suite.run.id"] == result.summary.run_id
        assert a["test.suite.run.status"] == "failure"
        assert a["opensearch.test_suite_run.case_count"] == 4
        assert a["opensearch.test_suite_run.pass_count"] == 2
        assert a["opensearch.test_suite_run.fail_count"] == 0
        assert a["opensearch.test_suite_run.error_count"] == 2
        assert isinstance(a["opensearch.test_suite_run.duration_ms"], float)
        assert a["agent_version"] == "v2"
        assert len(events(CASE_EVENT)) == 4

    def test_run_event_success(self, events):
        evaluate(name="ok", task=lambda x: x, data=[{"input": "q1"}], scores=[])
        a = events(RUN_EVENT)[0].log_record.attributes
        assert a["test.suite.run.status"] == "success"
        assert a["opensearch.test_suite_run.pass_count"] == 1

    def test_spans_outside_evaluate_not_stamped(self, exporter):
        evaluate(name="before", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        exporter.clear()
        _traced_agent()("q2")
        agent = _spans_named(exporter, "agent")[0]
        assert not [k for k in agent.attributes if k.startswith("test.")]

    def test_online_score_unaffected(self, exporter, log_exporter, events):
        from opensearch_genai_observability_sdk_py import score

        evaluate(name="warmup", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        exporter.clear()
        log_exporter.clear()
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            score("relevance", 0.9)
        lr = events(EVALUATION)[-1].log_record
        assert lr.span_id == span.get_span_context().span_id
        assert not [k for k in lr.attributes if k.startswith("test.")]
        assert not [k for k in exporter.get_finished_spans()[0].attributes if k.startswith("test.")]
        assert events(CASE_EVENT) == []


class TestBenchmarkEvents:
    def test_no_spans(self, exporter):
        with Benchmark(name="b-no-spans") as b:
            b.log(input="q1", scores={"a": 1.0})
        assert exporter.get_finished_spans() == ()

    def test_case_and_score_events_without_link(self, events):
        with Benchmark(name="b-evt", metadata={"model": "gpt-4"}, record_io=True) as b:
            b.log(
                input="q1",
                output="a1",
                expected="e1",
                case_id="c1",
                case_name="one",
                scores={"accuracy": 0.9},
                metadata={"split": "dev"},
                duration_ms=12.5,
            )
        case_lr = events(CASE_EVENT)[0].log_record
        score_lr = events(EVALUATION)[0].log_record
        for lr in (case_lr, score_lr):
            assert lr.trace_id == 0 and lr.span_id == 0
            assert lr.attributes["test.suite.run.id"] == b.run_id
            assert lr.attributes["test.case.id"] == "c1"
            assert lr.attributes["test.case.name"] == "one"
        a = case_lr.attributes
        assert a["test.case.result.status"] == "pass"
        assert a["opensearch.test_case.input"] == '"q1"'
        assert a["opensearch.test_case.output"] == '"a1"'
        assert a["opensearch.test_case.expected"] == '"e1"'
        assert a["opensearch.test_case.duration_ms"] == 12.5
        assert a["model"] == "gpt-4"
        assert a["split"] == "dev"
        # Case content never goes on score events
        assert not [k for k in score_lr.attributes if k.startswith("opensearch.test_case.")]

    def test_events_parented_to_linked_span(self, events):
        with Benchmark(name="b-link") as b:
            b.log(input="q1", scores={"a": 1.0}, trace_id=TRACE_ID, span_id=SPAN_ID)
        for lr in (events(CASE_EVENT)[0].log_record, events(EVALUATION)[0].log_record):
            assert format(lr.trace_id, "032x") == TRACE_ID
            assert format(lr.span_id, "016x") == SPAN_ID

    def test_status_and_run_event(self, events):
        b = Benchmark(name="b-status")
        b.log(input="q1", scores={"a": 1.0})
        b.log(input="q2", status="fail")
        b.log(input="q3", error="boom", error_type="TimeoutError")
        summary = b.close()
        statuses = {
            r.log_record.attributes["test.case.id"]: r.log_record.attributes
            for r in events(CASE_EVENT)
        }
        err = statuses[_make_case_id("q3")]
        assert err["test.case.result.status"] == "error"
        assert err["error.type"] == "TimeoutError"
        assert statuses[_make_case_id("q2")]["test.case.result.status"] == "fail"
        run = events(RUN_EVENT)[0].log_record.attributes
        assert run["test.suite.run.status"] == "failure"
        assert run["opensearch.test_suite_run.case_count"] == 3
        assert run["opensearch.test_suite_run.pass_count"] == 1
        assert run["opensearch.test_suite_run.fail_count"] == 1
        assert run["opensearch.test_suite_run.error_count"] == 1
        assert summary.error_count == 1

    def test_invalid_status_raises(self):
        with Benchmark(name="b-bad-status") as b:
            with pytest.raises(ValueError, match="status must be one of"):
                b.log(input="q1", status="skipped")


# ---------------------------------------------------------------------------
# Declared scorer rules and verdicts
# ---------------------------------------------------------------------------

LABEL = "gen_ai.evaluation.score.label"
THRESHOLD = "opensearch.evaluation.score.threshold"


def _grades(events):
    return {r.log_record.attributes["gen_ai.evaluation.name"]: r for r in events(EVALUATION)}


class TestScorerDecorator:
    def test_bare_decorator_keeps_function(self):
        @scorer
        def exact(i, o, e):
            return 1.0

        assert exact.__name__ == "exact"
        assert exact("q", "a", None) == 1.0
        assert exact.scorer_spec.name is None

    def test_threshold_defaults_direction_maximize(self):
        @scorer(threshold=4)
        def q(i, o, e):
            return 4.0

        assert q.scorer_spec.direction == "maximize"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            scorer(direction="up")

    def test_invalid_evaluator_type_raises(self):
        with pytest.raises(ValueError):
            scorer(evaluator_type="robot")

    def test_non_numeric_threshold_raises(self):
        with pytest.raises(TypeError):
            scorer(threshold="4")

    def test_float_result_uses_declared_name_and_rule(self, events):
        @scorer(
            name="answer_quality",
            threshold=4,
            score_min=1,
            score_max=5,
            evaluator="quality_judge",
            evaluator_type="llm_judge",
        )
        def quality(i, o, e):
            return 3.0

        result = evaluate(name="decl", task=lambda x: x, data=[{"input": "q"}], scores=[quality])
        rec = _grades(events)["answer_quality"]
        a = rec.log_record.attributes
        assert a["gen_ai.evaluation.score.value"] == 3.0
        assert a[THRESHOLD] == 4.0
        assert a["opensearch.evaluation.score.direction"] == "maximize"
        assert a["opensearch.evaluation.score.min"] == 1.0
        assert a["opensearch.evaluation.score.max"] == 5.0
        assert a["opensearch.evaluation.evaluator.name"] == "quality_judge"
        assert a["opensearch.evaluation.evaluator.type"] == "llm_judge"
        assert a[LABEL] == "fail"
        assert rec.instrumentation_scope.name == SDK_SCOPE
        assert result.cases[0].scores == {"answer_quality": 3.0}
        assert result.cases[0].verdicts == {"answer_quality": "fail"}

    def test_eval_score_fields_override_declaration(self, events):
        @scorer(name="latency_budget", direction="minimize", threshold=12)
        def latency(i, o, e):
            return EvalScore(name="latency_budget", value=13.0, threshold=15)

        evaluate(name="override", task=lambda x: x, data=[{"input": "q"}], scores=[latency])
        a = _grades(events)["latency_budget"].log_record.attributes
        assert a[THRESHOLD] == 15.0
        assert a["opensearch.evaluation.score.direction"] == "minimize"
        assert a[LABEL] == "pass"

    def test_passed_and_label_propagate(self, events):
        def checks(i, o, e):
            return [
                EvalScore(name="a", value=0.5, passed=True),
                EvalScore(name="b", value=0.5, label="fail"),
                EvalScore(name="c", value=None, label="pass"),
                EvalScore(name="d", value=0.5, label="pass", passed=False),
            ]

        result = evaluate(name="explicit", task=lambda x: x, data=[{"input": "q"}], scores=[checks])
        grades = _grades(events)
        assert grades["a"].log_record.attributes[LABEL] == "pass"
        assert grades["b"].log_record.attributes[LABEL] == "fail"
        assert grades["c"].log_record.attributes[LABEL] == "pass"
        assert "gen_ai.evaluation.score.value" not in grades["c"].log_record.attributes
        assert grades["d"].log_record.attributes[LABEL] == "pass"  # label wins
        assert result.cases[0].scores == {"a": 0.5, "b": 0.5, "d": 0.5}

    def test_bool_result_is_a_verdict(self, events):
        @scorer(name="tool_selection")
        def tools_ok(i, o, e):
            return False

        evaluate(name="bool", task=lambda x: x, data=[{"input": "q"}], scores=[tools_ok])
        a = _grades(events)["tool_selection"].log_record.attributes
        assert a["gen_ai.evaluation.score.value"] == 0.0
        assert a[LABEL] == "fail"


class TestCaseStatusFromVerdicts:
    def test_no_rules_case_passes(self, events):
        result = evaluate(
            name="no-rules", task=lambda x: x, data=[{"input": "q"}], scores=[lambda i, o, e: 0.0]
        )
        assert result.cases[0].status == "pass"
        assert result.cases[0].verdicts is None
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.result.status"] == "pass"
        assert LABEL not in events(EVALUATION)[0].log_record.attributes

    def test_fail_verdict_fails_case(self, events):
        @scorer(name="answer_quality", threshold=4)
        def quality(i, o, e):
            return 5.0 if i == "good" else 2.0

        result = evaluate(
            name="verdicts",
            task=lambda x: x,
            data=[{"input": "good", "case_id": "g"}, {"input": "bad", "case_id": "b"}],
            scores=[quality],
        )
        assert [c.status for c in result.cases] == ["pass", "fail"]
        by_case = {
            r.log_record.attributes["test.case.id"]: r.log_record.attributes
            for r in events(CASE_EVENT)
        }
        assert by_case["g"]["test.case.result.status"] == "pass"
        assert by_case["b"]["test.case.result.status"] == "fail"
        run = events(RUN_EVENT)[0].log_record.attributes
        assert run["opensearch.test_suite_run.pass_count"] == 1
        assert run["opensearch.test_suite_run.fail_count"] == 1
        assert run["test.suite.run.status"] == "success"  # only errors fail the run

    def test_error_beats_fail_verdict(self, events):
        def bad(i, o, e):
            raise RuntimeError("judge down")

        evaluate(
            name="err-first",
            task=lambda x: x,
            data=[{"input": "q"}],
            scores=[lambda i, o, e: EvalScore(name="x", value=0.0, passed=False), bad],
        )
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.result.status"] == "error"


class TestScorerErrorGradeName:
    def test_error_grade_uses_declared_name(self, events):
        @scorer(name="answer_quality", threshold=4, score_min=1, score_max=5)
        def quality(i, o, e):
            raise RuntimeError("judge down")

        evaluate(name="err-name", task=lambda x: x, data=[{"input": "q"}], scores=[quality])
        grades = _grades(events)
        assert list(grades) == ["answer_quality"]
        a = grades["answer_quality"].log_record.attributes
        assert a["error.type"] == "RuntimeError"
        assert a[THRESHOLD] == 4.0
        assert LABEL not in a
        assert "gen_ai.evaluation.score.value" not in a
        # The evaluator identity stays the function unless declared.
        assert a["opensearch.evaluation.evaluator.name"] == "quality"

    def test_error_grade_falls_back_to_function_name(self, events):
        def quality(i, o, e):
            raise RuntimeError("judge down")

        evaluate(name="err-fn", task=lambda x: x, data=[{"input": "q"}], scores=[quality])
        assert list(_grades(events)) == ["quality"]


class TestBenchmarkVerdicts:
    def test_eval_score_list_with_verdicts(self, events):
        with Benchmark(name="bench-verdicts") as b:
            b.log(
                input="q",
                case_id="c1",
                scores=[
                    EvalScore(
                        name="answer_quality", value=3.0, threshold=4, score_min=1, score_max=5
                    ),
                    EvalScore(name="tool_selection", value=1.0, label="pass"),
                ],
            )
        grades = _grades(events)
        quality = grades["answer_quality"].log_record.attributes
        assert quality[LABEL] == "fail"
        assert quality["opensearch.evaluation.score.max"] == 5.0
        assert grades["tool_selection"].log_record.attributes[LABEL] == "pass"
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.result.status"] == "fail"

    def test_dict_scores_unchanged(self, events):
        with Benchmark(name="bench-dict") as b:
            b.log(input="q", scores={"accuracy": 0.0})
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.result.status"] == "pass"
        assert LABEL not in events(EVALUATION)[0].log_record.attributes

    def test_explicit_status_wins(self, events):
        with Benchmark(name="bench-status") as b:
            b.log(input="q", scores=[EvalScore(name="x", value=0.0, passed=False)], status="pass")
        assert events(CASE_EVENT)[0].log_record.attributes["test.case.result.status"] == "pass"
