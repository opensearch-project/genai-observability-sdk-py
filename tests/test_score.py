# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.score."""

import pytest
from opentelemetry import trace
from opentelemetry._logs import SeverityNumber

from opensearch_genai_observability_sdk_py._internal import SDK_SCOPE_NAME, SDK_VERSION
from opensearch_genai_observability_sdk_py._internal import parse_hex as _parse_hex
from opensearch_genai_observability_sdk_py.score import EXPLANATION_MAX_CHARS, score

TRACE_ID = "6ebb9835f43af1552f2cebb9f5165e39"
SPAN_ID = "89829115c2128845"


def _one_event(eval_events):
    events = eval_events()
    assert len(events) == 1, events
    return events[0]


def _ids(record):
    lr = record.log_record
    return format(lr.trace_id, "032x"), format(lr.span_id, "016x")


# ---------------------------------------------------------------------------
# gen_ai.evaluation.result log record
# ---------------------------------------------------------------------------


class TestEventEmission:
    def test_event_name_is_top_level_field(self, eval_events):
        score(name="relevance", value=0.9)
        lr = _one_event(eval_events).log_record
        assert lr.event_name == "gen_ai.evaluation.result"
        assert "event.name" not in lr.attributes

    def test_severity_info(self, eval_events):
        score(name="relevance", value=0.9)
        lr = _one_event(eval_events).log_record
        assert lr.severity_number == SeverityNumber.INFO

    def test_timestamp_set(self, eval_events):
        import time

        before = time.time_ns()
        score(name="relevance", value=0.9)
        after = time.time_ns()
        lr = _one_event(eval_events).log_record
        assert lr.timestamp is not None
        assert before <= lr.timestamp <= after

    def test_body_empty(self, eval_events):
        score(name="relevance", value=0.9)
        assert _one_event(eval_events).log_record.body is None

    def test_no_span_created_by_default(self, exporter, eval_events):
        score(name="relevance", value=0.9, trace_id=TRACE_ID, span_id=SPAN_ID)
        assert exporter.get_finished_spans() == ()
        assert len(eval_events()) == 1

    def test_default_scope_is_sdk(self, eval_events):
        score(name="relevance", value=0.9)
        scope = _one_event(eval_events).instrumentation_scope
        assert scope.name == SDK_SCOPE_NAME == "opensearch-genai-observability-sdk-py"
        assert scope.version == SDK_VERSION
        assert scope.version
        # No invented scope attributes: the scope name is the SDK identity.
        assert not scope.attributes

    def test_evaluator_is_an_attribute_not_the_scope(self, eval_events):
        score(name="relevance", value=0.9, evaluator="relevance_judge", evaluator_version="3.1")
        rec = _one_event(eval_events)
        # The scope is the instrumentation library (the SDK), as OTel defines it.
        assert rec.instrumentation_scope.name == SDK_SCOPE_NAME
        assert rec.instrumentation_scope.version == SDK_VERSION
        assert (
            rec.log_record.attributes["opensearch.evaluation.evaluator.name"] == "relevance_judge"
        )
        assert rec.log_record.attributes["opensearch.evaluation.evaluator.version"] == "3.1"

    def test_multiple_scores_are_independent(self, eval_events):
        score(name="a", value=0.1)
        score(name="b", value=0.2)
        score(name="c", value=0.3)
        values = {
            r.log_record.attributes["gen_ai.evaluation.name"]: r.log_record.attributes[
                "gen_ai.evaluation.score.value"
            ]
            for r in eval_events()
        }
        assert values == {"a": 0.1, "b": 0.2, "c": 0.3}


class TestEventParenting:
    def test_explicit_span(self, eval_events):
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID)
        rec = _one_event(eval_events)
        assert _ids(rec) == (TRACE_ID, SPAN_ID)
        assert rec.log_record.attributes["opensearch.evaluation.level"] == "span"

    def test_explicit_ids_accept_0x_and_uppercase(self, eval_events):
        score(name="accuracy", trace_id="0x" + TRACE_ID.upper(), span_id=SPAN_ID.upper())
        assert _ids(_one_event(eval_events)) == (TRACE_ID, SPAN_ID)

    def test_active_span(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat gpt-4o") as span:
            score(name="relevance", value=0.92)
            sc = span.get_span_context()
        rec = _one_event(eval_events)
        assert rec.log_record.trace_id == sc.trace_id
        assert rec.log_record.span_id == sc.span_id
        assert rec.log_record.attributes["opensearch.evaluation.level"] == "span"

    def test_explicit_ids_win_over_active_span(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("other"):
            score(name="relevance", trace_id=TRACE_ID, span_id=SPAN_ID)
        assert _ids(_one_event(eval_events)) == (TRACE_ID, SPAN_ID)

    def test_trace_only_sets_the_record_trace_id_and_no_span_id(self, eval_events):
        score(name="relevance", value=0.92, trace_id=TRACE_ID)
        lr = _one_event(eval_events).log_record
        assert format(lr.trace_id, "032x") == TRACE_ID
        assert lr.span_id == 0
        assert "opensearch.evaluation.trace_id" not in lr.attributes
        assert lr.attributes["opensearch.evaluation.level"] == "trace"

    def test_trace_only_never_derives_root_span_id(self, eval_events, exporter):
        """Regression: no span id from the low 64 bits of the trace id."""
        score(name="relevance", value=0.92, trace_id=TRACE_ID)
        lr = _one_event(eval_events).log_record
        assert lr.span_id != int(TRACE_ID, 16) & 0xFFFFFFFFFFFFFFFF

    def test_trace_only_ignores_active_span(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("unrelated"):
            score(name="relevance", trace_id=TRACE_ID)
        lr = _one_event(eval_events).log_record
        assert format(lr.trace_id, "032x") == TRACE_ID and lr.span_id == 0

    def test_trace_id_normalized(self, eval_events):
        score(name="relevance", trace_id="0x" + TRACE_ID.upper())
        lr = _one_event(eval_events).log_record
        assert format(lr.trace_id, "032x") == TRACE_ID

    def test_no_context_no_level(self, eval_events):
        score(name="test", value=0.5)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0
        assert "opensearch.evaluation.level" not in lr.attributes

    def test_invalid_trace_id_standalone(self, eval_events):
        score(name="test", value=0.5, trace_id="not-hex!", span_id=SPAN_ID)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0
        assert "opensearch.evaluation.trace_id" not in lr.attributes

    def test_invalid_span_id_falls_back_to_trace_level(self, eval_events):
        score(name="test", value=0.5, trace_id=TRACE_ID, span_id="not-hex!")
        lr = _one_event(eval_events).log_record
        assert lr.span_id == 0
        assert format(lr.trace_id, "032x") == TRACE_ID
        assert lr.attributes["opensearch.evaluation.level"] == "trace"

    def test_span_id_without_trace_id_ignored(self, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("unrelated"):
            score(name="test", span_id=SPAN_ID)
        lr = _one_event(eval_events).log_record
        assert lr.trace_id == 0 and lr.span_id == 0


class TestEventAttributes:
    def test_standard_attributes(self, eval_events):
        score(
            name="helpfulness",
            value=0.83,
            label="helpful",
            explanation="Good response",
            response_id="resp_123",
            conversation_id="conv-1",
            agent_name="travel_agent",
        )
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.name"] == "helpfulness"
        assert a["gen_ai.evaluation.score.value"] == 0.83
        assert a["gen_ai.evaluation.score.label"] == "helpful"
        assert a["gen_ai.evaluation.explanation"] == "Good response"
        assert a["gen_ai.response.id"] == "resp_123"
        assert a["gen_ai.conversation.id"] == "conv-1"
        assert a["gen_ai.agent.name"] == "travel_agent"

    def test_extension_attributes(self, eval_events):
        score(
            name="faithfulness",
            value=4,
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            level="tool_call",
            evaluator="faith_judge",
            evaluator_version="1.0",
            evaluator_type="llm_judge",
            direction="maximize",
            score_min=1,
            score_max=5,
            judge_trace_id="abcd" * 8,
        )
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.level"] == "tool_call"
        assert a["opensearch.evaluation.evaluator.name"] == "faith_judge"
        assert a["opensearch.evaluation.evaluator.version"] == "1.0"
        assert a["opensearch.evaluation.evaluator.type"] == "llm_judge"
        assert a["opensearch.evaluation.score.direction"] == "maximize"
        assert a["opensearch.evaluation.score.min"] == 1.0
        assert a["opensearch.evaluation.score.max"] == 5.0
        assert isinstance(a["opensearch.evaluation.score.min"], float)
        assert a["opensearch.evaluation.judge.trace_id"] == "abcd" * 8

    def test_value_is_double(self, eval_events):
        score(name="t", value=1)
        v = _one_event(eval_events).log_record.attributes["gen_ai.evaluation.score.value"]
        assert v == 1.0 and isinstance(v, float)

    def test_zero_value_kept(self, eval_events):
        score(name="toxicity", value=0.0)
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.score.value"] == 0.0

    def test_minimal(self, eval_events):
        score(name="accuracy")
        a = dict(_one_event(eval_events).log_record.attributes)
        assert a == {"gen_ai.evaluation.name": "accuracy"}

    def test_error_type(self, eval_events):
        score(name="relevance", error_type="timeout")
        a = _one_event(eval_events).log_record.attributes
        assert a["error.type"] == "timeout"
        assert "gen_ai.evaluation.score.value" not in a

    def test_session_level_default(self, eval_events):
        score(name="user_satisfaction", value=0.9, conversation_id="session-456")
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.level"] == "session"
        assert a["gen_ai.conversation.id"] == "session-456"

    def test_explicit_level_overrides_default(self, eval_events):
        score(name="t", trace_id=TRACE_ID, span_id=SPAN_ID, level="trace")
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.level"] == "trace"

    def test_unknown_enum_recorded_with_warning(self, eval_events, caplog):
        score(name="t", evaluator_type="robot", direction="up", level="galaxy")
        a = _one_event(eval_events).log_record.attributes
        assert a["opensearch.evaluation.evaluator.type"] == "robot"
        assert "evaluator_type" in caplog.text

    def test_long_explanation_not_truncated_at_500(self, eval_events):
        score(name="t", explanation="x" * 5000)
        a = _one_event(eval_events).log_record.attributes
        assert len(a["gen_ai.evaluation.explanation"]) == 5000

    def test_explanation_safety_cap(self, eval_events):
        score(name="t", explanation="x" * (EXPLANATION_MAX_CHARS * 2))
        expl = _one_event(eval_events).log_record.attributes["gen_ai.evaluation.explanation"]
        assert len(expl) == EXPLANATION_MAX_CHARS == 8192
        assert expl.endswith("...[truncated]")

    def test_attributes_passthrough(self, eval_events):
        score(
            name="t",
            value=0.9,
            attributes={"test.suite.run.id": "run_001", "test.case.id": "case_001"},
        )
        a = _one_event(eval_events).log_record.attributes
        assert a["test.suite.run.id"] == "run_001"
        assert a["test.case.id"] == "case_001"


# ---------------------------------------------------------------------------
# Verdict: assessment and threshold
# ---------------------------------------------------------------------------

ASSESSMENT = "gen_ai.evaluation.score.label"
THRESHOLD = "opensearch.evaluation.score.threshold"
DIRECTION = "opensearch.evaluation.score.direction"


class TestVerdict:
    def test_no_rule_no_verdict(self, eval_events):
        score(name="relevance", value=0.9)
        a = _one_event(eval_events).log_record.attributes
        assert ASSESSMENT not in a
        assert THRESHOLD not in a
        assert DIRECTION not in a

    def test_explicit_assessment(self, eval_events):
        score(name="tool_selection", value=0.5, assessment="pass")
        assert _one_event(eval_events).log_record.attributes[ASSESSMENT] == "pass"

    def test_assessment_normalized(self, eval_events):
        score(name="a", assessment=" FAIL ")
        score(name="b", assessment=True)
        values = [r.log_record.attributes[ASSESSMENT] for r in eval_events()]
        assert values == ["fail", "pass"]

    def test_invalid_assessment_dropped_with_warning(self, eval_events, caplog):
        score(name="a", value=1.0, assessment="maybe")
        assert ASSESSMENT not in _one_event(eval_events).log_record.attributes
        assert "assessment" in caplog.text

    def test_threshold_maximize_default(self, eval_events):
        score(name="q", value=4.0, threshold=4)
        score(name="q", value=3.0, threshold=4)
        a, b = (r.log_record.attributes for r in eval_events())
        assert a[THRESHOLD] == 4.0
        assert isinstance(a[THRESHOLD], float)
        assert a[DIRECTION] == "maximize"
        assert a[ASSESSMENT] == "pass"
        assert b[ASSESSMENT] == "fail"

    def test_threshold_minimize(self, eval_events):
        score(name="latency", value=12.0, threshold=12, direction="minimize")
        score(name="latency", value=12.5, threshold=12, direction="minimize")
        a, b = (r.log_record.attributes for r in eval_events())
        assert a[DIRECTION] == "minimize"
        assert a[ASSESSMENT] == "pass"
        assert b[ASSESSMENT] == "fail"

    def test_explicit_assessment_wins_over_threshold(self, eval_events):
        score(name="q", value=1.0, threshold=4, assessment="pass")
        a = _one_event(eval_events).log_record.attributes
        assert a[ASSESSMENT] == "pass"
        assert a[THRESHOLD] == 4.0

    def test_verdict_is_the_label_and_no_assessment_attribute(self, eval_events):
        score(name="q", value=5.0, threshold=4)
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.score.label"] == "pass"
        assert "opensearch.evaluation.assessment" not in a

    def test_explicit_label_is_kept_over_the_verdict(self, eval_events):
        score(name="q", value=2.0, threshold=4, label="partially_correct")
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.score.label"] == "partially_correct"
        assert a[THRESHOLD] == 4.0

    def test_explicit_assessment_wins_over_a_label(self, eval_events, caplog):
        score(name="q", value=2.0, label="partially_correct", assessment="pass")
        a = _one_event(eval_events).log_record.attributes
        assert a["gen_ai.evaluation.score.label"] == "pass"
        assert "replaces label" in caplog.text

    def test_threshold_without_value_no_verdict(self, eval_events):
        score(name="q", threshold=4, error_type="TimeoutError")
        a = _one_event(eval_events).log_record.attributes
        assert a[THRESHOLD] == 4.0
        assert ASSESSMENT not in a


# ---------------------------------------------------------------------------
# No span carrier
# ---------------------------------------------------------------------------


class TestNoSpanCarrier:
    def test_carrier_parameter_removed(self):
        with pytest.raises(TypeError):
            score(name="t", carrier="span")  # type: ignore[call-arg]

    def test_legacy_env_var_ignored(self, monkeypatch, exporter, eval_events):
        monkeypatch.setenv("OPENSEARCH_GENAI_EVAL_CARRIER", "span")
        score(name="t", value=0.5, trace_id=TRACE_ID, span_id=SPAN_ID)
        assert exporter.get_finished_spans() == ()
        assert len(eval_events()) == 1

    def test_active_span_gets_no_child_or_event(self, exporter, eval_events):
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat"):
            score(name="t", value=0.5)
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["chat"]
        assert spans[0].events == ()
        assert len(eval_events()) == 1


# ---------------------------------------------------------------------------
# _parse_hex helper
# ---------------------------------------------------------------------------


class TestParseHex:
    def test_plain_hex(self):
        assert _parse_hex("ff") == 255

    def test_0x_prefix(self):
        assert _parse_hex("0xff") == 255

    def test_uppercase_0x_prefix(self):
        assert _parse_hex("0XFF") == 255

    def test_invalid_returns_none(self):
        assert _parse_hex("not-hex!") is None

    def test_empty_string_returns_zero(self):
        assert _parse_hex("") == 0
