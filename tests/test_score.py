# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.score."""

from opensearch_genai_observability_sdk_py._internal import parse_hex as _parse_hex
from opensearch_genai_observability_sdk_py.score import score

TRACE_ID = "6ebb9835f43af1552f2cebb9f5165e39"
SPAN_ID = "89829115c2128845"


# ---------------------------------------------------------------------------
# Span name and gen_ai.operation.name
# ---------------------------------------------------------------------------


class TestSpanStructure:
    def test_span_name_includes_metric_name(self, exporter):
        score(name="relevance", value=0.9)
        span = exporter.get_finished_spans()[0]
        assert span.name == "evaluation relevance"

    def test_operation_name_is_evaluation(self, exporter):
        score(name="accuracy", value=0.8)
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.operation.name"] == "evaluation"

    def test_evaluation_name_attribute(self, exporter):
        score(name="factuality", value=0.7)
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.evaluation.name"] == "factuality"

    def test_event_emitted(self, exporter):
        score(name="relevance", value=0.9)
        span = exporter.get_finished_spans()[0]
        assert len(span.events) == 1
        assert span.events[0].name == "gen_ai.evaluation.result"

    def test_event_attributes(self, exporter):
        score(
            name="helpfulness",
            value=0.83,
            label="Very helpful",
            explanation="Good response",
            response_id="resp_123",
        )
        span = exporter.get_finished_spans()[0]
        event = span.events[0]
        assert event.attributes["gen_ai.evaluation.name"] == "helpfulness"
        assert event.attributes["gen_ai.evaluation.score.value"] == 0.83
        assert event.attributes["gen_ai.evaluation.score.label"] == "Very helpful"
        assert event.attributes["gen_ai.evaluation.explanation"] == "Good response"
        assert event.attributes["gen_ai.response.id"] == "resp_123"

    def test_event_minimal(self, exporter):
        score(name="accuracy")
        span = exporter.get_finished_spans()[0]
        event = span.events[0]
        assert event.attributes["gen_ai.evaluation.name"] == "accuracy"
        assert "gen_ai.evaluation.score.value" not in event.attributes


# ---------------------------------------------------------------------------
# Span-level scoring — score span is a child of the evaluated span
# ---------------------------------------------------------------------------


class TestSpanLevelScoring:
    def test_score_span_joins_evaluated_trace(self, exporter):
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID)

        span = exporter.get_finished_spans()[0]
        assert format(span.context.trace_id, "032x") == TRACE_ID

    def test_score_span_parent_is_evaluated_span(self, exporter):
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID)

        span = exporter.get_finished_spans()[0]
        assert format(span.parent.span_id, "016x") == SPAN_ID

    def test_score_value(self, exporter):
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID)
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.95

    def test_no_string_trace_id_attribute(self, exporter):
        """trace_id is encoded in the span context, not duplicated as an attribute."""
        score(name="accuracy", value=0.95, trace_id=TRACE_ID, span_id=SPAN_ID)
        span = exporter.get_finished_spans()[0]
        assert "gen_ai.evaluation.trace_id" not in span.attributes
        assert "gen_ai.evaluation.span_id" not in span.attributes


# ---------------------------------------------------------------------------
# Trace-level scoring — score span attaches to root span of the trace
# ---------------------------------------------------------------------------


class TestTraceLevelScoring:
    def test_score_span_joins_evaluated_trace(self, exporter):
        score(name="relevance", value=0.92, trace_id=TRACE_ID)

        span = exporter.get_finished_spans()[0]
        assert format(span.context.trace_id, "032x") == TRACE_ID

    def test_parent_is_derived_from_trace_id(self, exporter):
        """Root span_id = lower 64 bits of trace_id."""
        score(name="relevance", value=0.92, trace_id=TRACE_ID)

        span = exporter.get_finished_spans()[0]
        expected_span_id = int(TRACE_ID, 16) & 0xFFFFFFFFFFFFFFFF
        assert span.parent.span_id == expected_span_id


# ---------------------------------------------------------------------------
# Standalone score (no trace context)
# ---------------------------------------------------------------------------


class TestStandaloneScore:
    def test_no_trace_id_emits_root_span(self, exporter):
        score(name="test", value=0.5)
        span = exporter.get_finished_spans()[0]
        assert span.parent is None

    def test_invalid_trace_id_emits_standalone(self, exporter):
        score(name="test", value=0.5, trace_id="not-hex!")
        span = exporter.get_finished_spans()[0]
        assert span.parent is None

    def test_invalid_span_id_emits_standalone(self, exporter):
        score(name="test", value=0.5, trace_id=TRACE_ID, span_id="not-hex!")
        span = exporter.get_finished_spans()[0]
        assert span.parent is None


# ---------------------------------------------------------------------------
# Score attributes
# ---------------------------------------------------------------------------


class TestScoreAttributes:
    def test_score_value(self, exporter):
        score(name="test", value=0.95)
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.95

    def test_zero_value(self, exporter):
        score(name="toxicity", value=0.0)
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.evaluation.score.value"] == 0.0

    def test_no_value(self, exporter):
        score(name="test")
        span = exporter.get_finished_spans()[0]
        assert "gen_ai.evaluation.score.value" not in span.attributes

    def test_label(self, exporter):
        score(name="test", label="pass")
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.evaluation.score.label"] == "pass"

    def test_no_label(self, exporter):
        score(name="test", value=0.5)
        span = exporter.get_finished_spans()[0]
        assert "gen_ai.evaluation.score.label" not in span.attributes

    def test_explanation(self, exporter):
        score(name="test", explanation="Correct answer.")
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.evaluation.explanation"] == "Correct answer."

    def test_explanation_truncated_at_500(self, exporter):
        score(name="test", explanation="x" * 1000)
        span = exporter.get_finished_spans()[0]
        assert len(span.attributes["gen_ai.evaluation.explanation"]) == 500

    def test_no_explanation(self, exporter):
        score(name="test", value=0.5)
        span = exporter.get_finished_spans()[0]
        assert "gen_ai.evaluation.explanation" not in span.attributes

    def test_response_id(self, exporter):
        score(name="test", response_id="chatcmpl-abc123")
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.response.id"] == "chatcmpl-abc123"

    def test_no_response_id(self, exporter):
        score(name="test", value=0.5)
        span = exporter.get_finished_spans()[0]
        assert "gen_ai.response.id" not in span.attributes

    def test_attributes_passthrough(self, exporter):
        score(
            name="test",
            value=0.9,
            attributes={
                "test.suite.run.id": "run_001",
                "test.case.id": "case_001",
            },
        )
        span = exporter.get_finished_spans()[0]
        assert span.attributes["test.suite.run.id"] == "run_001"
        assert span.attributes["test.case.id"] == "case_001"

    def test_no_attributes(self, exporter):
        score(name="test", value=0.5)
        span = exporter.get_finished_spans()[0]
        assert "test.suite.run.id" not in span.attributes


# ---------------------------------------------------------------------------
# Multiple scores
# ---------------------------------------------------------------------------


class TestMultipleScores:
    def test_multiple_scores_are_independent(self, exporter):
        score(name="a", value=0.1)
        score(name="b", value=0.2)
        score(name="c", value=0.3)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3
        values = {
            s.attributes["gen_ai.evaluation.name"]: s.attributes["gen_ai.evaluation.score.value"]
            for s in spans
        }
        assert values == {"a": 0.1, "b": 0.2, "c": 0.3}


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
