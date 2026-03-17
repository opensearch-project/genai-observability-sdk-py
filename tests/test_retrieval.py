# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for retrieval.py — message parsing and span doc mapping."""

import json
import unittest.mock

import pytest

from opensearch_genai_observability_sdk_py.retrieval import (
    Message,
    OpenSearchTraceRetriever,
    _extract_messages_from_doc,
    _parse_messages,
    map_span_doc,
)


def _msg(role: str, content: str) -> str:
    """Build a GenAI semconv message JSON string."""
    return json.dumps([{"role": role, "parts": [{"type": "text", "content": content}]}])


def _msgs(*pairs: tuple[str, str]) -> str:
    """Build multi-message JSON string from (role, content) pairs."""
    return json.dumps([{"role": r, "parts": [{"type": "text", "content": c}]} for r, c in pairs])


# ---------------------------------------------------------------------------
# _parse_messages
# ---------------------------------------------------------------------------


class TestParseMessages:
    def test_valid_message(self):
        result = _parse_messages(_msg("user", "hello"))
        assert result == [Message(role="user", content="hello")]

    def test_multiple_messages(self):
        result = _parse_messages(_msgs(("user", "hi"), ("assistant", "hey")))
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"

    def test_multiple_text_parts_joined(self):
        raw = json.dumps(
            [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "content": "a"},
                        {"type": "text", "content": "b"},
                    ],
                }
            ]
        )
        result = _parse_messages(raw)
        assert result[0].content == "a\nb"

    def test_non_text_parts_skipped(self):
        raw = json.dumps(
            [
                {
                    "role": "user",
                    "parts": [
                        {"type": "image", "content": "data"},
                        {"type": "text", "content": "hi"},
                    ],
                }
            ]
        )
        result = _parse_messages(raw)
        assert result[0].content == "hi"

    def test_no_text_parts_skips_message(self):
        raw = json.dumps(
            [
                {
                    "role": "user",
                    "parts": [{"type": "image", "content": "data"}],
                }
            ]
        )
        assert _parse_messages(raw) == []

    def test_empty_string(self):
        assert _parse_messages("") == []

    def test_none(self):
        assert _parse_messages(None) == []

    def test_invalid_json(self):
        assert _parse_messages("not json") == []

    def test_missing_role_defaults_unknown(self):
        raw = json.dumps([{"parts": [{"type": "text", "content": "hi"}]}])
        assert _parse_messages(raw)[0].role == "unknown"


# ---------------------------------------------------------------------------
# _extract_messages_from_doc
# ---------------------------------------------------------------------------


class TestExtractMessagesFromDoc:
    def test_events_format(self):
        doc = {
            "attributes": {},
            "events": [
                {
                    "attributes": {
                        "gen_ai.input.messages": _msg("user", "q"),
                        "gen_ai.output.messages": _msg("assistant", "a"),
                    }
                }
            ],
        }
        inp, out = _extract_messages_from_doc(doc)
        assert inp[0].content == "q"
        assert out[0].content == "a"

    def test_span_attrs_format(self):
        doc = {
            "attributes": {
                "gen_ai.input.messages": _msg("user", "q"),
                "gen_ai.output.messages": _msg("assistant", "a"),
            },
        }
        inp, out = _extract_messages_from_doc(doc)
        assert inp[0].content == "q"
        assert out[0].content == "a"

    def test_events_take_priority_over_attrs(self):
        doc = {
            "attributes": {
                "gen_ai.input.messages": _msg("user", "attr"),
            },
            "events": [
                {
                    "attributes": {
                        "gen_ai.input.messages": _msg("user", "event"),
                    }
                }
            ],
        }
        inp, _ = _extract_messages_from_doc(doc)
        assert inp[0].content == "event"

    def test_falls_back_to_attrs_when_no_events(self):
        doc = {
            "attributes": {
                "gen_ai.input.messages": _msg("user", "attr"),
            },
            "events": [],
        }
        inp, _ = _extract_messages_from_doc(doc)
        assert inp[0].content == "attr"

    def test_empty_doc(self):
        inp, out = _extract_messages_from_doc({"attributes": {}})
        assert inp == []
        assert out == []


# ---------------------------------------------------------------------------
# map_span_doc
# ---------------------------------------------------------------------------


class TestMapSpanDoc:
    def test_basic_mapping(self):
        doc = {
            "traceId": "abc123",
            "spanId": "span1",
            "parentSpanId": "parent1",
            "name": "invoke_agent Weather",
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-01T00:00:01Z",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "Weather",
                "gen_ai.request.model": "claude-sonnet-4",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
            },
        }
        span = map_span_doc(doc)
        assert span.trace_id == "abc123"
        assert span.span_id == "span1"
        assert span.operation_name == "invoke_agent"
        assert span.agent_name == "Weather"
        assert span.model == "claude-sonnet-4"
        assert span.input_tokens == 100
        assert span.output_tokens == 50

    def test_tool_span(self):
        doc = {
            "traceId": "t1",
            "spanId": "s1",
            "parentSpanId": "",
            "name": "execute_tool get_weather",
            "startTime": "",
            "endTime": "",
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.call.arguments": '{"city": "Mumbai"}',
                "gen_ai.tool.call.result": "25C sunny",
            },
        }
        span = map_span_doc(doc)
        assert span.tool_name == "get_weather"
        assert span.tool_call_arguments == '{"city": "Mumbai"}'
        assert span.tool_call_result == "25C sunny"

    def test_missing_fields_default(self):
        span = map_span_doc({})
        assert span.trace_id == ""
        assert span.operation_name == ""
        assert span.input_tokens == 0
        assert span.output_tokens == 0
        assert span.input_messages == []
        assert span.output_messages == []

    def test_raw_preserved(self):
        doc = {"traceId": "t1", "custom_field": "value"}
        span = map_span_doc(doc)
        assert span.raw["custom_field"] == "value"


# ---------------------------------------------------------------------------
# OpenSearchTraceRetriever (mocked OpenSearch client)
# ---------------------------------------------------------------------------


def _span_doc(
    trace_id="t1",
    span_id="s1",
    parent_span_id="",
    name="invoke_agent test",
    operation_name="invoke_agent",
    agent_name="TestAgent",
    input_content="hello",
    output_content="world",
):
    """Build a minimal OpenSearch span doc for testing."""
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": name,
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "attributes": {
            "gen_ai.operation.name": operation_name,
            "gen_ai.agent.name": agent_name,
            "gen_ai.input.messages": _msg("user", input_content),
            "gen_ai.output.messages": _msg("assistant", output_content),
        },
    }


class TestRetrieverConstructor:
    def test_missing_opensearchpy_raises(self):
        import unittest.mock

        with unittest.mock.patch.dict("sys.modules", {"opensearchpy": None}):
            with pytest.raises(ImportError, match="opensearch-py"):
                OpenSearchTraceRetriever(host="https://localhost:9200")

    def test_default_params(self):
        with unittest.mock.patch("opensearchpy.OpenSearch"):
            retriever = OpenSearchTraceRetriever()
            assert retriever._index == "otel-v1-apm-span-*"


class TestGetSession:
    @pytest.fixture
    def retriever(self):
        with unittest.mock.patch("opensearchpy.OpenSearch") as mock_cls:
            r = OpenSearchTraceRetriever(host="https://localhost:9200")
            r._mock_client = mock_cls.return_value
            return r

    def _mock_search(self, retriever, docs):
        retriever._mock_client.search.return_value = {
            "hits": {"hits": [{"_source": d} for d in docs]}
        }

    def test_happy_path_conversation_id(self, retriever):
        docs = [_span_doc(trace_id="t1", span_id="s1")]
        self._mock_search(retriever, docs)

        session = retriever.get_traces("conv_123")
        assert session.session_id == "conv_123"
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        assert session.traces[0].spans[0].agent_name == "TestAgent"

    def test_session_not_found_returns_empty(self, retriever):
        self._mock_search(retriever, [])
        session = retriever.get_traces("nonexistent")
        assert session.session_id == "nonexistent"
        assert session.traces == []

    def test_multiple_traces_grouped(self, retriever):
        docs = [
            _span_doc(trace_id="t1", span_id="s1"),
            _span_doc(trace_id="t1", span_id="s2"),
            _span_doc(trace_id="t2", span_id="s3"),
        ]
        self._mock_search(retriever, docs)

        session = retriever.get_traces("conv_123")
        assert len(session.traces) == 2
        trace_ids = {t.trace_id for t in session.traces}
        assert trace_ids == {"t1", "t2"}
        # t1 has 2 spans, t2 has 1
        spans_by_trace = {t.trace_id: len(t.spans) for t in session.traces}
        assert spans_by_trace["t1"] == 2
        assert spans_by_trace["t2"] == 1

    def test_fallback_to_trace_id(self, retriever):
        """When conversation_id query returns nothing, falls back to traceId."""
        retriever._mock_client.search.side_effect = [
            {"hits": {"hits": []}},  # conversation_id miss
            {"hits": {"hits": [{"_source": _span_doc(trace_id="t1")}]}},  # traceId hit
        ]
        session = retriever.get_traces("t1")
        assert len(session.traces) == 1

    def test_opensearch_error_propagates(self, retriever):
        retriever._mock_client.search.side_effect = Exception("connection refused")
        with pytest.raises(Exception, match="connection refused"):
            retriever.get_traces("conv_123")

    def test_messages_parsed_from_spans(self, retriever):
        docs = [_span_doc(input_content="question", output_content="answer")]
        self._mock_search(retriever, docs)

        session = retriever.get_traces("conv_123")
        span = session.traces[0].spans[0]
        assert span.input_messages[0].content == "question"
        assert span.output_messages[0].content == "answer"


# ---------------------------------------------------------------------------
# list_traces
# ---------------------------------------------------------------------------


class TestListRootSpans:
    @pytest.fixture
    def retriever(self):
        with unittest.mock.patch("opensearchpy.OpenSearch") as mock_cls:
            r = OpenSearchTraceRetriever(host="https://localhost:9200")
            r._mock_client = mock_cls.return_value
            return r

    @staticmethod
    def _mock_search(retriever, docs):
        retriever._mock_client.search.return_value = {
            "hits": {"hits": [{"_source": d} for d in docs]}
        }

    def test_returns_root_spans(self, retriever):
        self._mock_search(retriever, [_span_doc(trace_id="t1"), _span_doc(trace_id="t2")])
        results = retriever.list_root_spans()
        assert len(results) == 2
        assert results[0].trace_id == "t1"

    def test_filters_by_services(self, retriever):
        self._mock_search(retriever, [_span_doc(trace_id="t1")])
        retriever.list_root_spans(services=["weather-agent"])
        call_body = retriever._mock_client.search.call_args[1]["body"]
        terms = [c for c in call_body["query"]["bool"]["must"] if "terms" in c]
        assert terms[0]["terms"]["serviceName"] == ["weather-agent"]

    def test_no_service_filter(self, retriever):
        self._mock_search(retriever, [])
        retriever.list_root_spans()
        call_body = retriever._mock_client.search.call_args[1]["body"]
        terms = [c for c in call_body["query"]["bool"]["must"] if "terms" in c]
        assert terms == []

    def test_empty_results(self, retriever):
        self._mock_search(retriever, [])
        assert retriever.list_root_spans() == []

    def test_since_datetime(self, retriever):
        from datetime import datetime, timezone

        self._mock_search(retriever, [])
        ts = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
        retriever.list_root_spans(since=ts)
        call_body = retriever._mock_client.search.call_args[1]["body"]
        range_clause = [c for c in call_body["query"]["bool"]["must"] if "range" in c]
        assert range_clause[0]["range"]["startTime"]["gte"] == "2026-03-16T12:00:00.000Z"


# ---------------------------------------------------------------------------
# get_traces truncation
# ---------------------------------------------------------------------------


class TestGetTraceTruncation:
    @pytest.fixture
    def retriever(self):
        with unittest.mock.patch("opensearchpy.OpenSearch") as mock_cls:
            r = OpenSearchTraceRetriever(host="https://localhost:9200")
            r._mock_client = mock_cls.return_value
            return r

    def test_truncated_when_at_limit(self, retriever):
        docs = [_span_doc(trace_id="t1") for _ in range(5)]
        retriever._mock_client.search.side_effect = [
            {"hits": {"hits": [{"_source": d} for d in docs]}},
        ]
        session = retriever.get_traces("t1", max_spans=5)
        assert session.truncated is True

    def test_not_truncated_when_under_limit(self, retriever):
        docs = [_span_doc(trace_id="t1") for _ in range(3)]
        retriever._mock_client.search.side_effect = [
            {"hits": {"hits": [{"_source": d} for d in docs]}},
        ]
        session = retriever.get_traces("t1", max_spans=5)
        assert session.truncated is False


# ---------------------------------------------------------------------------
# find_evaluated_trace_ids
# ---------------------------------------------------------------------------


class TestFindEvaluatedTraceIds:
    @pytest.fixture
    def retriever(self):
        with unittest.mock.patch("opensearchpy.OpenSearch") as mock_cls:
            r = OpenSearchTraceRetriever(host="https://localhost:9200")
            r._mock_client = mock_cls.return_value
            return r

    def test_returns_evaluated_ids(self, retriever):
        retriever._mock_client.search.return_value = {
            "aggregations": {"evaluated": {"buckets": [{"key": "t1"}, {"key": "t3"}]}}
        }
        result = retriever.find_evaluated_trace_ids(["t1", "t2", "t3"])
        assert result == {"t1", "t3"}

    def test_empty_input(self, retriever):
        assert retriever.find_evaluated_trace_ids([]) == set()

    def test_none_evaluated(self, retriever):
        retriever._mock_client.search.return_value = {
            "aggregations": {"evaluated": {"buckets": []}}
        }
        result = retriever.find_evaluated_trace_ids(["t1", "t2"])
        assert result == set()
