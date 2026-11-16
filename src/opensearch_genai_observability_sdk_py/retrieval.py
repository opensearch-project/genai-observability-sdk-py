# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Retrieve GenAI trace data from OpenSearch.

Framework-agnostic: works with any agent library that emits OTel GenAI
semantic convention spans indexed by Data Prepper into ``otel-v1-apm-span-*``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from opensearch_genai_observability_sdk_py.enrich import AGENT_ROOT_ATTRIBUTE
from opensearch_genai_observability_sdk_py.score import (
    EVAL_EVALUATOR_NAME_KEY,
    EVALUATION_EVENT_NAME,
)

if TYPE_CHECKING:
    from opensearchpy import RequestsAWSV4SignerAuth

logger = logging.getLogger(__name__)

# list_root_spans() lists a trace by its agent root: a span marked with
# derived.gen_ai.is_root (by Data Prepper, an SDK or the user), for example the
# agent span below an HTTP server span, or a trace root (no parent) that is an
# agent operation. A judge's own parentless chat span is not an agent root.
_AI_ROOT_FIELD = f"attributes.{AGENT_ROOT_ATTRIBUTE}"
_OPERATION_FIELD = "attributes.gen_ai.operation.name"
_AGENT_OPERATIONS = ("invoke_agent", "create_agent")
_CONVERSATION_FIELD = "attributes.gen_ai.conversation.id"
_TRACE_ID_FIELD = "traceId"

_ROOT_RULE_HINT = (
    "list_root_spans() found no agent roots. A root is a span marked "
    "derived.gen_ai.is_root, or a span without a parent whose gen_ai.operation.name "
    "is invoke_agent or create_agent. Call mark_agent_root() in the agent's span, "
    "or trace the agent with @observe(op=Op.INVOKE_AGENT)."
)
_root_hint_logged = False

# Upper bound on the traces get_traces() collects for one conversation.
DEFAULT_MAX_CONVERSATION_TRACES = 1000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single user or assistant message extracted from a span."""

    role: str
    content: str


@dataclass
class SpanRecord:
    """Normalised view of one OpenSearch span document."""

    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    start_time: str
    end_time: str
    operation_name: str  # invoke_agent | execute_tool | chat
    agent_name: str = ""
    model: str = ""
    input_messages: list[Message] = field(default_factory=list)
    output_messages: list[Message] = field(default_factory=list)
    tool_name: str = ""
    tool_call_arguments: str = ""
    tool_call_result: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # Last, so positional construction of the earlier fields keeps working.
    agent_version: str = ""


@dataclass
class TraceRecord:
    """All spans sharing a single traceId."""

    trace_id: str
    spans: list[SpanRecord] = field(default_factory=list)


@dataclass
class SessionRecord:
    """All traces for a session (conversation)."""

    session_id: str
    traces: list[TraceRecord] = field(default_factory=list)
    truncated: bool = False


# ---------------------------------------------------------------------------
# Mapper: raw OpenSearch doc → SpanRecord
# ---------------------------------------------------------------------------


def _parse_messages(raw: str | list | None) -> list[Message]:
    """Parse a JSON-encoded messages string into Message objects.

    Handles the GenAI semconv ``[{"role": ..., "parts": [{"content": ...}]}]``
    format used by both span attributes and event attributes.
    """
    if not raw:
        return []
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    messages: list[Message] = []
    for item in items if isinstance(items, list) else [items]:
        if not isinstance(item, dict):
            continue  # not a GenAI message, for example a JSON-encoded plain string
        role = item.get("role", "unknown")
        parts = item.get("parts", [])
        text_parts = [
            p.get("content", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"
        ]
        if text_parts:
            messages.append(Message(role=role, content="\n".join(text_parts)))
    return messages


def _extract_messages_from_doc(doc: dict[str, Any]) -> tuple[list[Message], list[Message]]:
    """Extract input/output messages from a span doc, trying events first then attributes."""
    attrs = doc.get("attributes", {})
    events = doc.get("events", [])

    input_msgs: list[Message] = []
    output_msgs: list[Message] = []

    # Format 1 & 2: events (latest GenAI conventions)
    for evt in events:
        evt_attrs = evt.get("attributes", {})
        if not input_msgs and "gen_ai.input.messages" in evt_attrs:
            input_msgs = _parse_messages(evt_attrs["gen_ai.input.messages"])
        if not output_msgs and "gen_ai.output.messages" in evt_attrs:
            output_msgs = _parse_messages(evt_attrs["gen_ai.output.messages"])

    # Format 3: span attributes
    if not input_msgs:
        input_msgs = _parse_messages(attrs.get("gen_ai.input.messages"))
    if not output_msgs:
        output_msgs = _parse_messages(attrs.get("gen_ai.output.messages"))

    return input_msgs, output_msgs


def map_span_doc(doc: dict[str, Any]) -> SpanRecord:
    """Convert a raw OpenSearch span document to a SpanRecord."""
    attrs = doc.get("attributes", {})
    input_msgs, output_msgs = _extract_messages_from_doc(doc)

    return SpanRecord(
        trace_id=doc.get("traceId", ""),
        span_id=doc.get("spanId", ""),
        parent_span_id=doc.get("parentSpanId", ""),
        name=doc.get("name", ""),
        start_time=doc.get("startTime", ""),
        end_time=doc.get("endTime", ""),
        operation_name=attrs.get("gen_ai.operation.name", ""),
        agent_name=attrs.get("gen_ai.agent.name", ""),
        agent_version=str(attrs.get("gen_ai.agent.version", "") or ""),
        model=attrs.get("gen_ai.request.model", "") or attrs.get("gen_ai.response.model", ""),
        input_messages=input_msgs,
        output_messages=output_msgs,
        tool_name=attrs.get("gen_ai.tool.name", ""),
        tool_call_arguments=attrs.get("gen_ai.tool.call.arguments", ""),
        tool_call_result=attrs.get("gen_ai.tool.call.result", ""),
        input_tokens=attrs.get("gen_ai.usage.input_tokens", 0) or 0,
        output_tokens=attrs.get("gen_ai.usage.output_tokens", 0) or 0,
        raw=doc,
    )


def _agent_root_query() -> dict[str, Any]:
    """Marked span, or a parentless span of an agent operation."""
    return {
        "bool": {
            "should": [
                {"term": {_AI_ROOT_FIELD: True}},
                {
                    "bool": {
                        "filter": [
                            {"term": {"parentSpanId": ""}},
                            {"terms": {_OPERATION_FIELD: list(_AGENT_OPERATIONS)}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def _log_root_hint_once() -> None:
    global _root_hint_logged
    if not _root_hint_logged:
        _root_hint_logged = True
        logger.info(_ROOT_RULE_HINT)


# ---------------------------------------------------------------------------
# OpenSearch client wrapper
# ---------------------------------------------------------------------------


class OpenSearchTraceRetriever:
    """Query OpenSearch for GenAI trace spans and return SessionRecords.

    Args:
        host: OpenSearch endpoint (e.g. ``https://localhost:9200``).
        index: Index pattern for span data.
        evaluation_index: Index pattern for evaluation results
            (``gen_ai.evaluation.result`` events routed by Data Prepper).
        auth: Authentication — either a ``(username, password)`` tuple for
            basic auth, or a ``RequestsAWSV4SignerAuth`` instance for SigV4.
        verify_certs: Whether to verify TLS certificates.
    """

    def __init__(
        self,
        host: str = "https://localhost:9200",
        index: str = "otel-v1-apm-span-*",
        evaluation_index: str = "otel-v1-genai-evaluation*",
        auth: tuple[str, str] | RequestsAWSV4SignerAuth | None = None,
        verify_certs: bool = True,
    ):
        try:
            from opensearchpy import OpenSearch
        except ImportError as e:
            raise ImportError(
                "opensearch-py is required for retrieval. "
                "Install with: pip install opensearch-genai-observability-sdk-py[opensearch]"
            ) from e

        kwargs: dict[str, Any] = {
            "hosts": [host],
            "use_ssl": host.startswith("https"),
            "verify_certs": verify_certs,
        }
        if auth:
            kwargs["http_auth"] = auth
            # SigV4 auth requires the requests-based connection class.
            if not isinstance(auth, tuple):
                kwargs["connection_class"] = self._get_requests_connection()

        self._client = OpenSearch(**kwargs)
        self._index = index
        self._evaluation_index = evaluation_index

    @staticmethod
    def _get_requests_connection() -> Any:
        from opensearchpy import RequestsHttpConnection

        return RequestsHttpConnection

    def get_traces(
        self,
        identifier: str,
        max_spans: int = 10000,
        *,
        max_traces: int = DEFAULT_MAX_CONVERSATION_TRACES,
    ) -> SessionRecord:
        """Retrieve all spans of a conversation's traces, or of one trace.

        Looks up the traces whose spans carry ``gen_ai.conversation.id`` =
        *identifier* (earliest first), then fetches every span of those
        traces, so each ``TraceRecord`` holds the whole trace (model calls,
        tools, and the agent span) exactly as ``get_traces(trace_id)`` does.
        If no span has that conversation id, *identifier* is treated as a
        ``traceId``.

        Args:
            identifier: A conversation/session ID or a trace ID.
            max_spans: Maximum spans to fetch, over all traces.
            max_traces: Maximum traces to collect for a conversation (the
                earliest ones).

        ``SessionRecord.truncated`` is ``True`` when either limit was reached.
        """
        trace_ids, more_traces = self._conversation_trace_ids(identifier, max_traces)
        if trace_ids:
            docs = self._query_by_trace_ids(trace_ids, max_spans)
        else:
            docs = self._query_by_trace_id(identifier, max_spans)
        if not docs:
            return SessionRecord(session_id=identifier)

        session = self._build_session(identifier, docs)
        session.truncated = more_traces or len(docs) >= max_spans
        return session

    def list_root_spans(
        self,
        *,
        services: list[str] | None = None,
        since: datetime | None = None,
        max_results: int = 50,
    ) -> list[SpanRecord]:
        """List the agent roots of recent traces, newest first, one per trace.

        A span is an agent root when it is marked with
        ``derived.gen_ai.is_root`` (by Data Prepper, ``mark_agent_root()`` or
        the user), or when it has no parent and its ``gen_ai.operation.name``
        is ``invoke_agent`` or ``create_agent``. Other parentless GenAI spans,
        such as a traced LLM judge's own ``chat`` span, are not listed. For
        an agent behind an HTTP handler this returns the marked agent span
        (with its messages and agent name), not the HTTP server span.

        Each trace is listed once: by its parentless agent root when it has
        one, else by its earliest marked span (for example when the agents
        of several services in one trace are marked). Traces are picked by
        their newest matching span since *since*; the returned root may have
        started earlier.

        When nothing matches, one INFO line per process names this rule.
        Indices where the marker is not mapped yet (no marked span so far)
        are fine: the marker condition matches nothing there.

        Args:
            services: Only return traces from these service names.
            since: Only return traces with an agent root started after this
                time. Defaults to 15 minutes ago.
            max_results: Maximum number of traces (root spans) to return.
        """
        if max_results <= 0:
            return []
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(minutes=15)
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        service_filter = [{"terms": {"serviceName": services}}] if services else []

        # 1. Up to max_results distinct traces with an agent root since `since`.
        traces_body = {
            "size": min(max_results, 10000),
            "_source": [_TRACE_ID_FIELD],
            "sort": [{"startTime": "desc"}],
            "collapse": {"field": _TRACE_ID_FIELD},
            "query": {
                "bool": {
                    "filter": [
                        _agent_root_query(),
                        {"range": {"startTime": {"gte": since_iso}}},
                        *service_filter,
                    ]
                }
            },
        }
        found = (d.get(_TRACE_ID_FIELD) for d in self._search(traces_body))
        trace_ids = [t for t in dict.fromkeys(found) if t]
        if not trace_ids:
            _log_root_hint_once()
            return []

        # 2. The root of each of those traces, at any time: the parentless
        # agent root (scores 1) first, else the earliest marked span.
        roots_body = {
            "size": len(trace_ids),
            "sort": [{"_score": "desc"}, {"startTime": "asc"}],
            "collapse": {"field": _TRACE_ID_FIELD},
            "query": {
                "bool": {
                    "filter": [
                        _agent_root_query(),
                        {"terms": {_TRACE_ID_FIELD: trace_ids}},
                        *service_filter,
                    ],
                    "should": [{"constant_score": {"filter": {"term": {"parentSpanId": ""}}}}],
                }
            },
        }
        by_trace = {d.get(_TRACE_ID_FIELD): d for d in self._search(roots_body)}
        return [map_span_doc(by_trace[t]) for t in trace_ids if t in by_trace]

    def find_evaluated_trace_ids(
        self,
        trace_ids: list[str],
        *,
        evaluator: str | None = None,
        evaluation_name: str | None = None,
    ) -> set[str]:
        """Return the subset of trace_ids that already have an evaluation result.

        Looks for ``gen_ai.evaluation.result`` events in the evaluation index,
        matched on ``traceId``. Test case and run events in the same index are
        not evaluation results and are ignored. A missing evaluation index means
        nothing is evaluated yet and returns an empty set.

        Without a filter, a result from any evaluator counts. Pass *evaluator*
        and/or *evaluation_name* so a job skips only the traces it scored
        itself, not those another evaluator scored.

        Args:
            trace_ids: The traces to check.
            evaluator: Only count results of this evaluator
                (``opensearch.gen_ai.evaluation.evaluator.name``, the
                ``evaluator=`` of ``score()``).
            evaluation_name: Only count results with this name
                (``gen_ai.evaluation.name``, the ``name`` of ``score()``).

        Raises:
            Exception: Any other search error (connection, auth, mapping) from
                the OpenSearch client. It is not turned into an empty set, so
                a failed lookup never looks like "nothing evaluated" and
                callers do not re-run judges and write duplicate results.
        """
        if not trace_ids:
            return set()
        filters: list[dict] = [
            {"term": {"eventName": EVALUATION_EVENT_NAME}},
            {"terms": {"traceId": trace_ids}},
        ]
        if evaluator:
            filters.append({"term": {f"attributes.{EVAL_EVALUATOR_NAME_KEY}": evaluator}})
        if evaluation_name:
            filters.append({"term": {"attributes.gen_ai.evaluation.name": evaluation_name}})
        body = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "by_trace": {"terms": {"field": "traceId", "size": len(trace_ids)}},
            },
        }
        # A missing index (or pattern with no match) returns no hits instead of an error.
        resp = self._client.search(
            index=self._evaluation_index,
            body=body,
            ignore_unavailable=True,
            allow_no_indices=True,
        )
        aggs = resp.get("aggregations", {})
        wanted = set(trace_ids)
        found = {b["key"] for b in aggs.get("by_trace", {}).get("buckets", [])}
        return found & wanted

    # --- queries ---

    def _conversation_trace_ids(
        self, conversation_id: str, max_traces: int
    ) -> tuple[list[str], bool]:
        """Trace ids with a span carrying the conversation id, earliest first.

        Returns the ids and whether more traces exist than *max_traces*.
        """
        body = {
            "size": 0,
            "query": {"term": {_CONVERSATION_FIELD: conversation_id}},
            "aggs": {
                "by_trace": {
                    "terms": {
                        "field": "traceId",
                        "size": max_traces,
                        "order": {"first_start": "asc"},
                    },
                    "aggs": {"first_start": {"min": {"field": "startTime"}}},
                },
            },
        }
        resp = self._client.search(index=self._index, body=body)
        agg = resp.get("aggregations", {}).get("by_trace", {})
        ids = [b["key"] for b in agg.get("buckets", [])]
        return ids, bool(agg.get("sum_other_doc_count", 0))

    def _query_by_trace_ids(self, trace_ids: list[str], size: int) -> list[dict]:
        body = {
            "size": size,
            "query": {"terms": {"traceId": trace_ids}},
            "sort": [{"startTime": "asc"}],
        }
        return self._search(body)

    def _query_by_trace_id(self, trace_id: str, size: int) -> list[dict]:
        body = {
            "size": size,
            "query": {"term": {"traceId": trace_id}},
            "sort": [{"startTime": "asc"}],
        }
        return self._search(body)

    def _search(self, body: dict) -> list[dict]:
        resp = self._client.search(index=self._index, body=body)
        return [hit["_source"] for hit in resp.get("hits", {}).get("hits", [])]

    # --- assembly ---

    @staticmethod
    def _build_session(session_id: str, docs: list[dict]) -> SessionRecord:
        traces_by_id: dict[str, TraceRecord] = {}
        for doc in docs:
            span = map_span_doc(doc)
            tid = span.trace_id
            if tid not in traces_by_id:
                traces_by_id[tid] = TraceRecord(trace_id=tid)
            traces_by_id[tid].spans.append(span)

        return SessionRecord(
            session_id=session_id,
            traces=list(traces_by_id.values()),
        )
