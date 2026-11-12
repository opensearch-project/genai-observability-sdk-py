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

    def test_agent_name_and_version(self):
        doc = {"attributes": {"gen_ai.agent.name": "Weather", "gen_ai.agent.version": "v2"}}
        span = map_span_doc(doc)
        assert span.agent_name == "Weather"
        assert span.agent_version == "v2"

    def test_agent_version_missing_is_empty(self):
        assert map_span_doc({"attributes": {"gen_ai.agent.name": "Weather"}}).agent_version == ""

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
    agent_version="v1",
    marked=False,
):
    """Build a minimal OpenSearch span doc for testing."""
    attributes = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.agent.name": agent_name,
        "gen_ai.agent.version": agent_version,
        "gen_ai.input.messages": _msg("user", input_content),
        "gen_ai.output.messages": _msg("assistant", output_content),
    }
    if marked:
        attributes["derived.gen_ai.is_root"] = True
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": name,
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "attributes": attributes,
    }


def _hits(docs):
    return {"hits": {"hits": [{"_source": d} for d in docs]}}


def _trace_buckets(*trace_ids, other=0):
    """Response of the conversation lookup (trace ids, earliest first)."""
    return {
        "aggregations": {
            "by_trace": {
                "sum_other_doc_count": other,
                "buckets": [{"key": t, "doc_count": 1} for t in trace_ids],
            }
        }
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

    def _bodies(self, retriever):
        return [c.kwargs["body"] for c in retriever._mock_client.search.call_args_list]

    def test_happy_path_conversation_id(self, retriever):
        docs = [_span_doc(trace_id="t1", span_id="s1")]
        retriever._mock_client.search.side_effect = [_trace_buckets("t1"), _hits(docs)]

        session = retriever.get_traces("conv_123")
        assert session.session_id == "conv_123"
        assert len(session.traces) == 1
        assert len(session.traces[0].spans) == 1
        assert session.traces[0].spans[0].agent_name == "TestAgent"
        assert session.traces[0].spans[0].agent_version == "v1"

    def test_conversation_returns_full_traces(self, retriever):
        """The conversation's trace ids are looked up, then every span of those traces."""
        docs = [
            _span_doc(trace_id="t1", span_id="root1"),
            _span_doc(
                trace_id="t1", span_id="chat1", parent_span_id="root1", operation_name="chat"
            ),
            _span_doc(
                trace_id="t1",
                span_id="tool1",
                parent_span_id="root1",
                operation_name="execute_tool",
            ),
            _span_doc(trace_id="t2", span_id="root2"),
            _span_doc(
                trace_id="t2", span_id="chat2", parent_span_id="root2", operation_name="chat"
            ),
        ]
        retriever._mock_client.search.side_effect = [_trace_buckets("t1", "t2"), _hits(docs)]

        session = retriever.get_traces("conv_123")
        assert [t.trace_id for t in session.traces] == ["t1", "t2"]
        assert [len(t.spans) for t in session.traces] == [3, 2]
        assert session.truncated is False

        lookup, fetch = self._bodies(retriever)
        assert lookup["size"] == 0
        assert lookup["query"] == {"term": {"attributes.gen_ai.conversation.id": "conv_123"}}
        by_trace = lookup["aggs"]["by_trace"]
        assert by_trace["terms"]["field"] == "traceId"
        assert by_trace["terms"]["order"] == {"first_start": "asc"}
        assert by_trace["aggs"]["first_start"] == {"min": {"field": "startTime"}}
        assert fetch["query"] == {"terms": {"traceId": ["t1", "t2"]}}
        assert fetch["sort"] == [{"startTime": "asc"}]
        assert fetch["size"] == 10000

    def test_conversation_trace_lookup_is_bounded(self, retriever):
        retriever._mock_client.search.side_effect = [
            _trace_buckets("t1", "t2", other=3),
            _hits([_span_doc(trace_id="t1"), _span_doc(trace_id="t2", span_id="s2")]),
        ]
        session = retriever.get_traces("conv_123", max_traces=2)
        lookup = self._bodies(retriever)[0]
        assert lookup["aggs"]["by_trace"]["terms"]["size"] == 2
        assert session.truncated is True

    def test_conversation_shape_matches_trace_id(self, retriever):
        """By conversation and by trace id give the same TraceRecord for a trace."""
        docs = [
            _span_doc(trace_id="t1", span_id="root1"),
            _span_doc(
                trace_id="t1", span_id="chat1", parent_span_id="root1", operation_name="chat"
            ),
        ]
        retriever._mock_client.search.side_effect = [
            _trace_buckets("t1"),
            _hits(docs),
            {"aggregations": {"by_trace": {"buckets": []}}},  # "t1" is not a conversation id
            _hits(docs),
        ]
        by_conversation = retriever.get_traces("conv_123")
        by_trace = retriever.get_traces("t1")
        assert by_conversation.traces == by_trace.traces

    def test_session_not_found_returns_empty(self, retriever):
        retriever._mock_client.search.side_effect = [_trace_buckets(), _hits([])]
        session = retriever.get_traces("nonexistent")
        assert session.session_id == "nonexistent"
        assert session.traces == []

    def test_multiple_traces_grouped(self, retriever):
        docs = [
            _span_doc(trace_id="t1", span_id="s1"),
            _span_doc(trace_id="t1", span_id="s2"),
            _span_doc(trace_id="t2", span_id="s3"),
        ]
        retriever._mock_client.search.side_effect = [_trace_buckets("t1", "t2"), _hits(docs)]

        session = retriever.get_traces("conv_123")
        assert len(session.traces) == 2
        spans_by_trace = {t.trace_id: len(t.spans) for t in session.traces}
        assert spans_by_trace == {"t1": 2, "t2": 1}

    def test_fallback_to_trace_id(self, retriever):
        """When no span carries the conversation id, falls back to traceId."""
        retriever._mock_client.search.side_effect = [
            {"aggregations": {}},  # conversation_id miss
            _hits([_span_doc(trace_id="t1")]),  # traceId hit
        ]
        session = retriever.get_traces("t1")
        assert len(session.traces) == 1
        assert self._bodies(retriever)[1]["query"] == {"term": {"traceId": "t1"}}

    def test_opensearch_error_propagates(self, retriever):
        retriever._mock_client.search.side_effect = Exception("connection refused")
        with pytest.raises(Exception, match="connection refused"):
            retriever.get_traces("conv_123")

    def test_messages_parsed_from_spans(self, retriever):
        docs = [_span_doc(input_content="question", output_content="answer")]
        retriever._mock_client.search.side_effect = [_trace_buckets("t1"), _hits(docs)]

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

    def test_ai_root_condition(self, retriever):
        """Agent Traces rule: (no parent AND GenAI span) OR marked derived.gen_ai.is_root."""
        self._mock_search(retriever, [])
        retriever.list_root_spans()
        must = retriever._mock_client.search.call_args[1]["body"]["query"]["bool"]["must"]
        ai_root = next(c["bool"] for c in must if "bool" in c)
        assert ai_root["minimum_should_match"] == 1
        assert ai_root["should"] == [
            {
                "bool": {
                    "filter": [
                        {"term": {"parentSpanId": ""}},
                        {"exists": {"field": "attributes.gen_ai.operation.name"}},
                    ]
                }
            },
            {"term": {"attributes.derived.gen_ai.is_root": True}},
        ]
        # No bare parentSpanId filter: an HTTP server root must not be required.
        assert {"term": {"parentSpanId": ""}} not in must

    def test_returns_marked_agent_span_behind_http_root(self, retriever):
        """An HTTP-rooted trace is listed by its marked agent span, with agent and messages."""
        agent = _span_doc(
            trace_id="t1",
            span_id="agent",
            parent_span_id="http",
            name="invoke_agent travel_planner",
            agent_name="travel_planner",
            marked=True,
        )
        self._mock_search(retriever, [agent])
        (root,) = retriever.list_root_spans()
        assert root.span_id == "agent"
        assert root.agent_name == "travel_planner"
        assert root.agent_version == "v1"
        assert root.input_messages[0].content == "hello"

    def test_drops_marked_span_shadowed_by_genai_root(self, retriever):
        """A trace with a GenAI root is listed once, by that root."""
        root = _span_doc(trace_id="t1", span_id="root")
        nested = _span_doc(trace_id="t1", span_id="nested", parent_span_id="root", marked=True)
        other = _span_doc(trace_id="t2", span_id="agent2", parent_span_id="http2", marked=True)
        self._mock_search(retriever, [nested, root, other])
        assert [r.span_id for r in retriever.list_root_spans()] == ["root", "agent2"]

    def test_parallel_marked_agents_each_listed(self, retriever):
        a = _span_doc(trace_id="t1", span_id="a", parent_span_id="http", marked=True)
        b = _span_doc(trace_id="t1", span_id="b", parent_span_id="http", marked=True)
        self._mock_search(retriever, [a, b])
        assert [r.span_id for r in retriever.list_root_spans()] == ["a", "b"]

    def test_max_results_after_dropping_shadowed(self, retriever):
        docs = [
            _span_doc(trace_id="t1", span_id="n1", parent_span_id="r1", marked=True),
            _span_doc(trace_id="t1", span_id="r1"),
            _span_doc(trace_id="t2", span_id="r2"),
            _span_doc(trace_id="t3", span_id="r3"),
        ]
        self._mock_search(retriever, docs)
        results = retriever.list_root_spans(max_results=2)
        assert [r.span_id for r in results] == ["r1", "r2"]
        assert retriever._mock_client.search.call_args[1]["body"]["size"] == 4


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
        retriever._mock_client.search.side_effect = [_trace_buckets("t1"), _hits(docs)]
        session = retriever.get_traces("t1", max_spans=5)
        assert session.truncated is True

    def test_not_truncated_when_under_limit(self, retriever):
        docs = [_span_doc(trace_id="t1") for _ in range(3)]
        retriever._mock_client.search.side_effect = [_trace_buckets("t1"), _hits(docs)]
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

    def test_empty_input(self, retriever):
        assert retriever.find_evaluated_trace_ids([]) == set()
        retriever._mock_client.search.assert_not_called()

    def test_event_results_from_evaluation_index(self, retriever):
        retriever._mock_client.search.return_value = {
            "aggregations": {
                "by_trace": {"buckets": [{"key": "t1"}, {"key": "t2"}]},
            }
        }
        assert retriever.find_evaluated_trace_ids(["t1", "t2", "t3"]) == {"t1", "t2"}
        # One query, against the evaluation index only (no span fallback)
        assert retriever._mock_client.search.call_count == 1
        call = retriever._mock_client.search.call_args
        assert call.kwargs["index"] == "otel-v1-genai-evaluation*"
        assert call.kwargs["ignore_unavailable"] is True

    def test_only_grade_events_count(self, retriever):
        """Test case / run events in the same index are not evaluation results."""
        retriever._mock_client.search.return_value = {"aggregations": {}}
        retriever.find_evaluated_trace_ids(["t1"])
        query = retriever._mock_client.search.call_args.kwargs["body"]["query"]["bool"]
        assert {"term": {"eventName": "gen_ai.evaluation.result"}} in query["filter"]

    def test_missing_index_is_not_an_error(self, retriever):
        """ignore_unavailable / allow_no_indices make a missing index return no buckets."""
        retriever._mock_client.search.return_value = {"hits": {"hits": []}}
        assert retriever.find_evaluated_trace_ids(["t1"]) == set()
        call = retriever._mock_client.search.call_args
        assert call.kwargs["allow_no_indices"] is True
        assert call.kwargs["ignore_unavailable"] is True

    @pytest.mark.parametrize(
        "error", [ConnectionError("connection refused"), PermissionError("403"), ValueError("x")]
    )
    def test_other_errors_raise(self, retriever, error):
        """Auth / connection / mapping errors must not look like "nothing evaluated"."""
        retriever._mock_client.search.side_effect = error
        with pytest.raises(type(error)):
            retriever.find_evaluated_trace_ids(["t1", "t3"])

    def test_no_filter_counts_any_evaluator(self, retriever):
        retriever._mock_client.search.return_value = {"aggregations": {}}
        retriever.find_evaluated_trace_ids(["t1"])
        filters = retriever._mock_client.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
        assert filters == [
            {"term": {"eventName": "gen_ai.evaluation.result"}},
            {"terms": {"traceId": ["t1"]}},
        ]

    def test_evaluator_filter(self, retriever):
        retriever._mock_client.search.return_value = {
            "aggregations": {"by_trace": {"buckets": [{"key": "t1"}]}}
        }
        found = retriever.find_evaluated_trace_ids(["t1", "t2"], evaluator="answer_quality_judge")
        assert found == {"t1"}
        filters = retriever._mock_client.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
        assert {
            "term": {
                "attributes.opensearch.gen_ai.evaluation.evaluator.name": "answer_quality_judge"
            }
        } in filters
        assert not any("attributes.gen_ai.evaluation.name" in f.get("term", {}) for f in filters)

    def test_evaluation_name_filter(self, retriever):
        retriever._mock_client.search.return_value = {"aggregations": {}}
        retriever.find_evaluated_trace_ids(["t1"], evaluation_name="answer_quality")
        filters = retriever._mock_client.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
        assert {"term": {"attributes.gen_ai.evaluation.name": "answer_quality"}} in filters
        assert not any(
            "attributes.opensearch.gen_ai.evaluation.evaluator.name" in f.get("term", {})
            for f in filters
        )

    def test_evaluator_and_evaluation_name_filters(self, retriever):
        retriever._mock_client.search.return_value = {"aggregations": {}}
        retriever.find_evaluated_trace_ids(["t1"], evaluator="judge", evaluation_name="quality")
        filters = retriever._mock_client.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
        assert {"term": {"attributes.opensearch.gen_ai.evaluation.evaluator.name": "judge"}} in (
            filters
        )
        assert {"term": {"attributes.gen_ai.evaluation.name": "quality"}} in filters

    def test_none_evaluated(self, retriever):
        retriever._mock_client.search.return_value = {"aggregations": {}}
        assert retriever.find_evaluated_trace_ids(["t1", "t2"]) == set()
