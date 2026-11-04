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
# Benchmark: score events only (no spans, no case or run events)
# ---------------------------------------------------------------------------

EVALUATION = "gen_ai.evaluation.result"
SDK_SCOPE = "opensearch-genai-observability-sdk-py"
# Attributes that must never be emitted.
REMOVED_KEYS = ("test.case.result.status", "test.suite.run.status")


@pytest.fixture()
def events(log_exporter):
    """Return a callable listing emitted records with a given event name."""

    def _get(name):
        return [r for r in log_exporter.get_finished_logs() if r.log_record.event_name == name]

    return _get


def _assert_only_evaluation_events(log_exporter):
    """Every record is a gen_ai.evaluation.result without removed attributes."""
    for r in log_exporter.get_finished_logs():
        assert r.log_record.event_name == EVALUATION
        a = r.log_record.attributes
        assert not [k for k in a if k in REMOVED_KEYS]
        assert not [
            k for k in a if k.startswith(("opensearch.test_case.", "opensearch.test_suite_run."))
        ]


class TestBenchmarkRun:
    def test_no_run_event(self, log_exporter):
        with Benchmark(name="my-bench") as b:
            pass
        assert b.run_id.startswith("run_")
        assert log_exporter.get_finished_logs() == ()

    def test_run_metadata_on_score_events(self, events):
        with Benchmark(name="meta-bench", metadata={"model": "gpt-4"}) as b:
            b.log(input="q1", scores={"a": 1.0, "b": 0.5})
        recs = events(EVALUATION)
        assert len(recs) == 2
        assert all(r.log_record.attributes["model"] == "gpt-4" for r in recs)

    def test_reserved_metadata_filtered(self, events):
        with Benchmark(name="reserved-bench", metadata={"test.suite.name": "oops", "m": "x"}) as b:
            b.log(input="q1", scores={"a": 1.0})
        a = events(EVALUATION)[0].log_record.attributes
        assert a["test.suite.name"] == "reserved-bench"
        assert a["m"] == "x"

    def test_evaluation_field_metadata_filtered(self, events, caplog):
        """Metadata cannot set a verdict or error the SDK left unset."""
        with Benchmark(
            name="eval-fields",
            metadata={"gen_ai.evaluation.score.label": "pass", "error.type": "X", "m": "x"},
        ) as b:
            b.log(
                input="q1",
                scores=[
                    EvalScore(
                        name="a",
                        value=0.2,
                        metadata={"opensearch.gen_ai.evaluation.score.threshold": 0.1},
                    )
                ],
            )
        a = events(EVALUATION)[0].log_record.attributes
        assert "gen_ai.evaluation.score.label" not in a
        assert "error.type" not in a
        assert "opensearch.gen_ai.evaluation.score.threshold" not in a
        assert a["m"] == "x"
        assert "'error.type'" in caplog.text and "reserved by the SDK" in caplog.text

    def test_custom_test_prefix_allowed(self, events):
        """User metadata with test.* prefix is allowed if not an SDK key."""
        with Benchmark(name="custom-test", metadata={"test.custom_field": "hello"}) as b:
            b.log(input="q1", scores={"a": 1.0})
        assert events(EVALUATION)[0].log_record.attributes["test.custom_field"] == "hello"

    def test_properties(self):
        b = Benchmark(name="prop-test")
        assert b.name == "prop-test"
        b.close()

    def test_record_io_removed(self):
        with pytest.raises(TypeError):
            Benchmark(name="io", record_io=True)


class TestRunId:
    def test_benchmark_run_id_used_as_given(self, events):
        with Benchmark(name="shard", run_id="  ci-123  ") as b:
            b.log(input="q1", scores={"a": 1.0})
        assert b.run_id == "  ci-123  "
        assert events(EVALUATION)[0].log_record.attributes["test.suite.run.id"] == "  ci-123  "

    def test_evaluate_run_id_used_as_given(self, exporter, events):
        result = evaluate(
            name="shard",
            task=_traced_agent(),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0],
            run_id="ci-456",
        )
        assert result.summary.run_id == "ci-456"
        assert events(EVALUATION)[0].log_record.attributes["test.suite.run.id"] == "ci-456"
        assert _spans_named(exporter, "agent")[0].attributes["test.suite.run.id"] == "ci-456"

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_empty_run_id_raises(self, bad):
        with pytest.raises(ValueError, match="run_id"):
            Benchmark(name="b", run_id=bad)
        with pytest.raises(ValueError, match="run_id"):
            evaluate(name="e", task=lambda x: x, data=[{"input": "q"}], scores=[], run_id=bad)

    def test_generated_when_none(self):
        first = Benchmark(name="gen", run_id=None)
        second = Benchmark(name="gen")
        assert first.run_id.startswith("run_")
        assert first.run_id != second.run_id
        first.close()
        second.close()


class TestBenchmarkLog:
    def test_score_event_attributes(self, events):
        with Benchmark(name="attr-test") as b:
            b.log(input="What is Python?", scores={"accuracy": 0.9}, case_name="python_q")
        a = events(EVALUATION)[0].log_record.attributes
        assert a["test.suite.name"] == "attr-test"
        assert a["test.suite.run.id"] == b.run_id
        assert a["test.case.id"] == _make_case_id("What is Python?")
        assert a["test.case.name"] == "python_q"

    def test_explicit_case_id(self, events):
        with Benchmark(name="id-test") as b:
            b.log(case_id="custom_123", scores={"a": 1.0})
        assert events(EVALUATION)[0].log_record.attributes["test.case.id"] == "custom_123"

    def test_error_is_local_error_status_only(self, log_exporter):
        b = Benchmark(name="err-test")
        b.log(input="q1", error="something broke")
        b.close()
        assert b._cases[0].status == "error"
        # No scores: nothing emitted, and the message is never an attribute.
        assert log_exporter.get_finished_logs() == ()

    def test_multiple_cases(self, events, log_exporter):
        with Benchmark(name="multi") as b:
            b.log(input="q1", scores={"a": 1.0})
            b.log(input="q2", scores={"a": 0.5})
            b.log(input="q3", scores={"a": 0.0})
        assert len(events(EVALUATION)) == 3
        _assert_only_evaluation_events(log_exporter)

    def test_case_metadata_over_run_metadata(self, events):
        with Benchmark(name="res-meta", metadata={"model": "gpt-4", "split": "test"}) as b:
            b.log(
                input="q1",
                scores={"a": 1.0},
                metadata={"test.case.id": "oops", "split": "dev"},
            )
        a = events(EVALUATION)[0].log_record.attributes
        assert a["test.case.id"] == _make_case_id("q1")
        assert a["model"] == "gpt-4"
        assert a["split"] == "dev"

    def test_metadata_precedence(self, events):
        with Benchmark(
            name="prec", agent_name="real_agent", metadata={"k": "run", "gen_ai.agent.name": "x"}
        ) as b:
            b.log(
                input="q1",
                metadata={"k": "case", "j": "case"},
                scores=[
                    EvalScore(
                        name="acc",
                        value=1.0,
                        metadata={
                            "j": "score",
                            "gen_ai.agent.name": "y",
                            "gen_ai.evaluation.name": "z",
                        },
                    )
                ],
            )
        a = events(EVALUATION)[0].log_record.attributes
        assert a["k"] == "case"
        assert a["j"] == "score"
        # Agent attribute beats score metadata; score fields beat everything.
        assert a["gen_ai.agent.name"] == "real_agent"
        assert a["gen_ai.evaluation.name"] == "acc"

    def test_duration_ms_param_removed(self):
        with Benchmark(name="no-dur") as b:
            with pytest.raises(TypeError):
                b.log(input="q1", duration_ms=1.0)

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
            b.log(input="q1", scores={"a": 1.0}, trace_id="not-hex!", span_id=SPAN_ID)
        lr = events(EVALUATION)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0

    def test_error_type_makes_error_results(self, events):
        b = Benchmark(name="err-type")
        b.log(
            input="q1",
            scores=[
                EvalScore(name="acc", value=0.0, threshold=0.5, explanation="timed out"),
                EvalScore(name="tone", value=1.0, label="pass"),
            ],
            error_type="TimeoutError",
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
        )
        summary = b.close()
        grades = _grades(events)
        assert set(grades) == {"acc", "tone"}
        for rec in grades.values():
            a = rec.log_record.attributes
            assert a["error.type"] == "TimeoutError"
            assert "gen_ai.evaluation.score.value" not in a
            assert LABEL not in a
            assert format(rec.log_record.span_id, "016x") == SPAN_ID
        assert grades["acc"].log_record.attributes["gen_ai.evaluation.explanation"] == "timed out"
        case = b._cases[0]
        assert case.status == "error"
        assert case.scores == {}
        assert case.verdicts is None
        assert summary.error_count == 1
        assert summary.scores == {}


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
        assert lr.attributes["opensearch.gen_ai.evaluation.level"] == "span"

    def test_no_scores_no_events(self, log_exporter):
        with Benchmark(name="no-score") as b:
            b.log(input="q1")
        assert log_exporter.get_finished_logs() == ()

    def test_io_never_recorded(self, events):
        with Benchmark(name="no-io") as b:
            b.log(input="secret", output="answer", expected="truth", scores={"a": 1.0})
        a = events(EVALUATION)[0].log_record.attributes
        assert "secret" not in str(dict(a))
        assert not [k for k in a if k.startswith(("opensearch.test_case.", "test.case.input"))]


class TestBenchmarkSummary:
    def test_close_returns_summary(self):
        b = Benchmark(name="sum-test")
        b.log(input="q1", scores={"accuracy": 1.0, "relevance": 0.8})
        b.log(input="q2", scores={"accuracy": 0.5, "relevance": 0.9})
        summary = b.close()

        assert isinstance(summary, BenchmarkSummary)
        assert summary.benchmark_name == "sum-test"
        assert summary.total_cases == 2
        assert summary.pass_count == 2
        assert summary.error_count == 0
        assert summary.scores["accuracy"].avg == 0.75
        assert summary.scores["accuracy"].min == 0.5
        assert summary.scores["accuracy"].max == 1.0
        assert summary.scores["accuracy"].count == 2
        assert summary.scores["relevance"].avg == pytest.approx(0.85)

    def test_summary_counts(self):
        b = Benchmark(name="err-sum")
        b.log(input="q1", scores={"a": 1.0})
        b.log(input="q2", status="fail")
        b.log(input="q3", error="failed")
        b.log(input="q4", error_type="TimeoutError")
        summary = b.close()
        assert summary.total_cases == 4
        assert (summary.pass_count, summary.fail_count, summary.error_count) == (1, 1, 2)

    def test_summary_str_format(self):
        b = Benchmark(name="fmt-test")
        b.log(input="q1", scores={"accuracy": 0.9})
        text = str(b.close())
        assert "BENCHMARK: fmt-test" in text
        assert "Pass: 1 | Fail: 0 | Errors: 0" in text
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

    def test_named_scorer_is_an_attribute_and_scope_is_sdk(self, events, log_exporter):
        def judge(i, o, e):
            return 1.0

        evaluate(name="scope", task=lambda x: x, data=[{"input": "q1"}], scores=[judge])
        _assert_only_evaluation_events(log_exporter)
        grade = events(EVALUATION)[0]
        assert grade.log_record.attributes["opensearch.gen_ai.evaluation.evaluator.name"] == "judge"
        assert grade.instrumentation_scope.name == SDK_SCOPE
        assert grade.instrumentation_scope.version == SDK_VERSION
        assert not grade.instrumentation_scope.attributes

    def test_anonymous_scorer_uses_sdk_scope(self, events):
        evaluate(name="anon", task=lambda x: x, data=[{"input": "q1"}], scores=[lambda i, o, e: 1])
        rec = events(EVALUATION)[0]
        assert rec.instrumentation_scope.name == SDK_SCOPE
        assert rec.instrumentation_scope.version == SDK_VERSION

    def test_benchmark_events_use_sdk_scope(self, events):
        with Benchmark(name="b-scope") as b:
            b.log(input="q1", scores={"a": 1.0})
        rec = events(EVALUATION)[0]
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
        assert result.cases[0].status == "error"
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
            scores=[lambda i, o, e: 1.0],
        )
        assert events(EVALUATION)[0].log_record.attributes["test.case.id"] == "my_custom_id"

    def test_evaluate_prints_summary_to_stdout(self, capsys):
        evaluate(name="print-eval", task=lambda x: x, data=[{"input": "q1"}], scores=[])
        assert "BENCHMARK: print-eval" in capsys.readouterr().out

    def test_evaluate_scorer_error_surfaces(self):
        def bad_scorer(input_val, output, expected):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err", task=lambda x: x, data=[{"input": "q1"}], scores=[bad_scorer]
        )
        assert result.cases[0].status == "error"
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
        assert a["opensearch.gen_ai.evaluation.evaluator.name"] == "judge"
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

    def test_evaluate_metadata_on_score_events(self, events):
        evaluate(
            name="res-meta-eval",
            task=lambda x: x,
            data=[{"input": "q1"}, {"input": "q2"}],
            scores=[lambda i, o, e: EvalScore(name="a", value=1.0, metadata={"m": "score"})],
            metadata={"test.suite.name": "oops", "model": "gpt-4", "m": "run"},
        )
        recs = events(EVALUATION)
        assert len(recs) == 2
        for r in recs:
            a = r.log_record.attributes
            assert a["test.suite.name"] == "res-meta-eval"
            assert a["model"] == "gpt-4"
            assert a["m"] == "score"

    def test_record_io_removed(self):
        with pytest.raises(TypeError):
            evaluate(name="io", task=lambda x: x, data=[], scores=[], record_io=True)


# ---------------------------------------------------------------------------
# Offline runs: agent trace + score events
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
        evaluate(
            name="no-name",
            task=_traced_agent(),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0],
        )
        root = _spans_named(exporter, "agent")[0]
        assert "test.case.name" not in root.attributes
        assert "test.case.name" not in events(EVALUATION)[0].log_record.attributes

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
        score_rec = events(EVALUATION)[0]
        assert score_rec.log_record.trace_id == root.context.trace_id
        assert score_rec.log_record.span_id == root.context.span_id
        a = score_rec.log_record.attributes
        for key in GROUPING:
            assert a[key] == root.attributes[key]
        assert a["opensearch.gen_ai.evaluation.level"] == "span"
        assert a["gen_ai.evaluation.score.label"] == "relevant"
        assert score_rec.instrumentation_scope.name == SDK_SCOPE
        case = result.cases[0]
        assert case.trace_id == format(root.context.trace_id, "032x")
        assert case.span_id == format(root.context.span_id, "016x")

    def test_score_event_shape(self, events, log_exporter):
        evaluate(
            name="shape",
            task=_traced_agent(),
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
            metadata={"model": "gpt-4", "test.suite.name": "oops"},
        )
        _assert_only_evaluation_events(log_exporter)
        rec = events(EVALUATION)[0]
        lr = rec.log_record
        a = lr.attributes
        assert a["test.suite.name"] == "shape"
        assert a["test.case.id"] == "c1"
        assert "error.type" not in a
        assert a["model"] == "gpt-4"
        assert not [k for k in a if "input" in k or "output" in k]
        assert lr.timestamp is not None
        assert lr.severity_number.value == 9
        assert rec.instrumentation_scope.name == "opensearch-genai-observability-sdk-py"

    def test_crashed_case_emits_error_result_per_scorer(self, exporter, events):
        @scorer(name="answer_quality", threshold=4, evaluator="judge", evaluator_type="llm_judge")
        def quality(i, o, e):
            return 5.0

        def exact(i, o, e):
            return 1.0

        result = evaluate(
            name="crash",
            task=_traced_agent(),
            data=[{"input": "boom", "case_id": "bad"}, {"input": "q2", "case_id": "good"}],
            scores=[quality, exact, lambda i, o, e: 1.0],
            metadata={"model": "gpt-4"},
        )
        assert [c.case_id for c in result.cases] == ["bad", "good"]
        bad_case = result.cases[0]
        assert bad_case.error == "task broke"
        assert bad_case.status == "error"
        assert bad_case.error_type == "ValueError"
        assert bad_case.scores == {}
        assert result.cases[1].status == "pass"

        bad_root = next(
            s for s in _spans_named(exporter, "agent") if s.attributes["test.case.id"] == "bad"
        )
        bad = [
            r.log_record
            for r in events(EVALUATION)
            if r.log_record.attributes["test.case.id"] == "bad"
        ]
        assert sorted(lr.attributes["gen_ai.evaluation.name"] for lr in bad) == [
            "<lambda>",
            "answer_quality",
            "exact",
        ]
        for lr in bad:
            a = lr.attributes
            assert a["error.type"] == "ValueError"
            assert "gen_ai.evaluation.score.value" not in a
            assert LABEL not in a
            assert "gen_ai.evaluation.explanation" not in a
            assert a["model"] == "gpt-4"
            assert a["test.suite.name"] == "crash"
            # Still parented to the crashed agent run's root span
            assert lr.trace_id == bad_root.context.trace_id
            assert lr.span_id == bad_root.context.span_id
        by_name = {lr.attributes["gen_ai.evaluation.name"]: lr.attributes for lr in bad}
        assert by_name["answer_quality"]["opensearch.gen_ai.evaluation.evaluator.name"] == "judge"
        assert (
            by_name["answer_quality"]["opensearch.gen_ai.evaluation.evaluator.type"] == "llm_judge"
        )
        assert by_name["answer_quality"][THRESHOLD] == 4.0
        assert by_name["exact"]["opensearch.gen_ai.evaluation.evaluator.name"] == "exact"
        assert "opensearch.gen_ai.evaluation.evaluator.name" not in by_name["<lambda>"]
        good = [r for r in events(EVALUATION) if r.log_record.attributes["test.case.id"] == "good"]
        assert len(good) == 3
        assert all("error.type" not in r.log_record.attributes for r in good)

    def test_crashed_untraced_case_has_no_trace_context(self, events):
        def boom(x):
            raise KeyError("x")

        evaluate(name="crash2", task=boom, data=[{"input": "q"}], scores=[lambda i, o, e: 1.0])
        lr = events(EVALUATION)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0
        assert lr.attributes["error.type"] == "KeyError"

    def test_scorer_error_marks_case_error(self, events):
        def bad_scorer(i, o, e):
            raise RuntimeError("scorer broke")

        result = evaluate(
            name="scorer-err-evt",
            task=_traced_agent(),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0, bad_scorer],
        )
        assert result.cases[0].status == "error"
        assert result.cases[0].error_type == "RuntimeError"
        assert result.cases[0].scorer_errors
        failed = [
            r.log_record.attributes
            for r in events(EVALUATION)
            if r.log_record.attributes["gen_ai.evaluation.name"] == "bad_scorer"
        ]
        assert failed[0]["error.type"] == "RuntimeError"
        # The exception message is not exported: it can carry URLs, keys or bodies.
        assert "gen_ai.evaluation.explanation" not in failed[0]
        assert not any("scorer broke" in str(v) for v in failed[0].values())
        assert any("scorer broke" in e for e in result.cases[0].scorer_errors)
        assert result.summary.error_count == 1

    def test_untraced_task_events_without_trace_context(self, exporter, events):
        result = evaluate(
            name="untraced",
            task=lambda x: x,
            data=[{"input": "q1", "case_id": "c1"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert exporter.get_finished_spans() == ()
        score_lr = events(EVALUATION)[0].log_record
        assert score_lr.trace_id == 0 and score_lr.span_id == 0
        assert score_lr.attributes["test.case.id"] == "c1"
        assert score_lr.attributes["test.suite.name"] == "untraced"
        assert "opensearch.gen_ai.evaluation.level" not in score_lr.attributes
        assert result.cases[0].trace_id is None

    def test_untraced_task_ignores_ambient_span(self, events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("outer"):
            evaluate(
                name="ambient", task=lambda x: x, data=[{"input": "q1"}], scores=[lambda i, o, e: 1]
            )
        lr = events(EVALUATION)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0

    def test_agent_run_is_own_trace_inside_active_span(self, exporter):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("outer") as outer:
            evaluate(name="nested", task=_traced_agent(), data=[{"input": "q1"}], scores=[])
        agent = _spans_named(exporter, "agent")[0]
        assert agent.parent is None
        assert agent.context.trace_id != outer.get_span_context().trace_id
        assert agent.attributes["test.suite.name"] == "nested"

    def test_summary_counts(self, log_exporter):
        def scorer_fn(i, o, e):
            if i == "q3":
                raise RuntimeError("nope")
            return 1.0

        result = evaluate(
            name="counts",
            task=_traced_agent(),
            data=[{"input": "q1"}, {"input": "boom"}, {"input": "q3"}, {"input": "q4"}],
            scores=[scorer_fn],
        )
        s = result.summary
        assert (s.total_cases, s.pass_count, s.fail_count, s.error_count) == (4, 2, 0, 2)
        assert [c.status for c in result.cases] == ["pass", "error", "error", "pass"]
        # One evaluation result per case (the scorer), nothing else.
        assert len(log_exporter.get_finished_logs()) == 4
        _assert_only_evaluation_events(log_exporter)

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
        assert len(log_exporter.get_finished_logs()) == 1


class TestBenchmarkEvents:
    def test_no_spans(self, exporter):
        with Benchmark(name="b-no-spans") as b:
            b.log(input="q1", scores={"a": 1.0})
        assert exporter.get_finished_spans() == ()

    def test_score_events_without_link(self, events, log_exporter):
        with Benchmark(name="b-evt", metadata={"model": "gpt-4"}) as b:
            b.log(
                input="q1",
                output="a1",
                expected="e1",
                case_id="c1",
                case_name="one",
                scores={"accuracy": 0.9},
                metadata={"split": "dev"},
            )
        _assert_only_evaluation_events(log_exporter)
        lr = events(EVALUATION)[0].log_record
        assert lr.trace_id == 0 and lr.span_id == 0
        a = lr.attributes
        assert a["test.suite.run.id"] == b.run_id
        assert a["test.case.id"] == "c1"
        assert a["test.case.name"] == "one"
        assert a["model"] == "gpt-4"
        assert a["split"] == "dev"

    def test_events_parented_to_linked_span(self, events):
        with Benchmark(name="b-link") as b:
            b.log(input="q1", scores={"a": 1.0}, trace_id=TRACE_ID, span_id=SPAN_ID)
        lr = events(EVALUATION)[0].log_record
        assert format(lr.trace_id, "032x") == TRACE_ID
        assert format(lr.span_id, "016x") == SPAN_ID

    def test_local_statuses(self, log_exporter):
        b = Benchmark(name="b-status")
        b.log(input="q1", scores={"a": 1.0})
        b.log(input="q2", status="fail")
        b.log(input="q3", error="boom", error_type="TimeoutError")
        summary = b.close()
        assert [c.status for c in b._cases] == ["pass", "fail", "error"]
        assert b._cases[2].error_type == "TimeoutError"
        assert (summary.pass_count, summary.fail_count, summary.error_count) == (1, 1, 1)
        _assert_only_evaluation_events(log_exporter)

    def test_invalid_status_raises(self):
        with Benchmark(name="b-bad-status") as b:
            with pytest.raises(ValueError, match="status must be one of"):
                b.log(input="q1", status="skipped")


# ---------------------------------------------------------------------------
# Declared scorer rules and verdicts
# ---------------------------------------------------------------------------

LABEL = "gen_ai.evaluation.score.label"
THRESHOLD = "opensearch.gen_ai.evaluation.score.threshold"


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
        assert a["opensearch.gen_ai.evaluation.score.direction"] == "maximize"
        assert a["opensearch.gen_ai.evaluation.score.min"] == 1.0
        assert a["opensearch.gen_ai.evaluation.score.max"] == 5.0
        assert a["opensearch.gen_ai.evaluation.evaluator.name"] == "quality_judge"
        assert a["opensearch.gen_ai.evaluation.evaluator.type"] == "llm_judge"
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
        assert a["opensearch.gen_ai.evaluation.score.direction"] == "minimize"
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
        assert (result.summary.pass_count, result.summary.fail_count) == (1, 1)
        assert result.summary.error_count == 0

    def test_error_beats_fail_verdict(self, events):
        def bad(i, o, e):
            raise RuntimeError("judge down")

        result = evaluate(
            name="err-first",
            task=lambda x: x,
            data=[{"input": "q"}],
            scores=[lambda i, o, e: EvalScore(name="x", value=0.0, passed=False), bad],
        )
        assert result.cases[0].status == "error"


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
        assert a["opensearch.gen_ai.evaluation.evaluator.name"] == "quality"

    def test_error_grade_falls_back_to_function_name(self, events):
        def quality(i, o, e):
            raise RuntimeError("judge down")

        evaluate(name="err-fn", task=lambda x: x, data=[{"input": "q"}], scores=[quality])
        assert list(_grades(events)) == ["quality"]


JUDGE_LABEL = "opensearch.gen_ai.evaluation.judge_label"


class TestLabelVerdictRule:
    """The stored label always agrees with TestCaseResult.status."""

    def _log(self, *scores):
        with Benchmark(name="label-rule") as b:
            b.log(input="q", case_id="c", scores=list(scores))
        return b._cases[0]

    def test_judge_label_with_failing_passed(self, events):
        case = self._log(EvalScore(name="rel", value=0.4, label="relevant", passed=False))
        a = _grades(events)["rel"].log_record.attributes
        assert (a[LABEL], a[JUDGE_LABEL]) == ("fail", "relevant")
        assert case.status == "fail" and case.verdicts == {"rel": "fail"}

    def test_judge_label_with_failing_threshold(self, events):
        case = self._log(EvalScore(name="q", value=2.0, label="partially_correct", threshold=4))
        a = _grades(events)["q"].log_record.attributes
        assert (a[LABEL], a[JUDGE_LABEL]) == ("fail", "partially_correct")
        assert case.status == "fail"

    def test_uppercase_verdict_label_stored_lowercase(self, events):
        case = self._log(EvalScore(name="ok", value=1.0, label="PASS"))
        a = _grades(events)["ok"].log_record.attributes
        assert a[LABEL] == "pass"
        assert JUDGE_LABEL not in a
        assert case.status == "pass"

    def test_conflicting_verdicts_warn_and_label_wins(self, events, caplog):
        case = self._log(EvalScore(name="x", value=0.0, label="pass", passed=False, threshold=1))
        a = _grades(events)["x"].log_record.attributes
        assert a[LABEL] == "pass"
        assert case.status == "pass"
        assert "conflicts with passed (fail), threshold (fail)" in caplog.text

    def test_evaluate_status_matches_stored_label(self, events):
        @scorer(name="rel", threshold=0.5)
        def rel(i, o, e):
            return EvalScore(name="rel", value=0.2, label="relevant")

        result = evaluate(name="lr", task=lambda x: x, data=[{"input": "q"}], scores=[rel])
        a = _grades(events)["rel"].log_record.attributes
        assert a[LABEL] == "fail" == result.cases[0].verdicts["rel"]
        assert result.cases[0].status == "fail"


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
        assert quality["opensearch.gen_ai.evaluation.score.max"] == 5.0
        assert grades["tool_selection"].log_record.attributes[LABEL] == "pass"
        assert b._cases[0].status == "fail"

    def test_dict_scores_unchanged(self, events):
        with Benchmark(name="bench-dict") as b:
            b.log(input="q", scores={"accuracy": 0.0})
        assert b._cases[0].status == "pass"
        assert LABEL not in events(EVALUATION)[0].log_record.attributes

    def test_explicit_status_wins(self, events):
        with Benchmark(name="bench-status") as b:
            b.log(input="q", scores=[EvalScore(name="x", value=0.0, passed=False)], status="pass")
        assert b._cases[0].status == "pass"


# ---------------------------------------------------------------------------
# Expected output on evaluation events
# ---------------------------------------------------------------------------

EXPECTED = "opensearch.test.case.expected"
EXPECTED_TRUNCATED = "opensearch.test.case.expected_truncated"


class TestExpectedOutput:
    def test_string_on_every_result_of_the_case(self, events):
        with Benchmark(name="exp-str") as b:
            b.log(input="q", expected="Paris", scores={"a": 1.0, "b": 0.5})
        records = events(EVALUATION)
        assert len(records) == 2
        for r in records:
            assert r.log_record.attributes[EXPECTED] == "Paris"
            assert EXPECTED_TRUNCATED not in r.log_record.attributes

    def test_dict_is_json_encoded(self, events):
        def exact(input_val, output, expected):
            return 1.0

        evaluate(
            name="exp-dict",
            task=lambda x: x,
            data=[{"input": "q", "expected": {"city": "Paris", "n": 1}}],
            scores=[exact],
        )
        a = events(EVALUATION)[0].log_record.attributes
        assert a[EXPECTED] == '{"city":"Paris","n":1}'
        assert EXPECTED_TRUNCATED not in a

    def test_truncated_at_4096(self, events):
        with Benchmark(name="exp-long") as b:
            b.log(input="q", expected="x" * 5000, scores={"a": 1.0})
        a = events(EVALUATION)[0].log_record.attributes
        assert a[EXPECTED] == "x" * 4096
        assert a[EXPECTED_TRUNCATED] is True

    def test_exactly_4096_not_truncated(self, events):
        with Benchmark(name="exp-edge") as b:
            b.log(input="q", expected="y" * 4096, scores={"a": 1.0})
        a = events(EVALUATION)[0].log_record.attributes
        assert a[EXPECTED] == "y" * 4096
        assert EXPECTED_TRUNCATED not in a

    def test_absent_when_no_expected(self, events):
        def exact(input_val, output, expected):
            return 1.0

        with Benchmark(name="exp-none") as b:
            b.log(input="q", scores={"a": 1.0})
        evaluate(name="exp-none-eval", task=lambda x: x, data=[{"input": "q"}], scores=[exact])
        records = events(EVALUATION)
        assert len(records) == 2
        for r in records:
            assert EXPECTED not in r.log_record.attributes
            assert EXPECTED_TRUNCATED not in r.log_record.attributes

    def test_metadata_cannot_set_expected(self, events):
        with Benchmark(name="exp-meta", metadata={EXPECTED: "run"}) as b:
            b.log(
                input="q",
                metadata={EXPECTED_TRUNCATED: True},
                scores=[EvalScore(name="a", value=1.0, metadata={EXPECTED: "score"})],
            )
        a = events(EVALUATION)[0].log_record.attributes
        assert EXPECTED not in a
        assert EXPECTED_TRUNCATED not in a

    def test_on_error_results_and_not_on_root_span(self, exporter, events):
        tracer = trace.get_tracer("test")

        def crashing(x):
            with tracer.start_as_current_span("invoke_agent"):
                raise RuntimeError("boom")

        def exact(input_val, output, expected):
            return 1.0

        evaluate(
            name="exp-crash",
            task=crashing,
            data=[{"input": "q", "expected": "A"}],
            scores=[exact],
        )
        a = events(EVALUATION)[0].log_record.attributes
        assert a["error.type"] == "RuntimeError"
        assert a[EXPECTED] == "A"
        root = [s for s in exporter.get_finished_spans() if s.name == "invoke_agent"][0]
        assert EXPECTED not in (root.attributes or {})


class TestJudgeTraceIdFromEvalScore:
    JUDGE = "opensearch.gen_ai.evaluation.judge.trace_id"

    def test_benchmark_log(self, events):
        with Benchmark(name="judge-log") as b:
            b.log(input="q", scores=[EvalScore(name="a", value=1.0, judge_trace_id="ab" * 16)])
        assert events(EVALUATION)[0].log_record.attributes[self.JUDGE] == "ab" * 16

    def test_evaluate_scorer(self, events):
        def judge(input_val, output, expected):
            return EvalScore(name="q", value=0.9, judge_trace_id="cd" * 16)

        evaluate(name="judge-eval", task=lambda x: x, data=[{"input": "q"}], scores=[judge])
        assert events(EVALUATION)[0].log_record.attributes[self.JUDGE] == "cd" * 16

    def test_absent_by_default(self, events):
        with Benchmark(name="judge-none") as b:
            b.log(input="q", scores={"a": 1.0})
        assert self.JUDGE not in events(EVALUATION)[0].log_record.attributes
