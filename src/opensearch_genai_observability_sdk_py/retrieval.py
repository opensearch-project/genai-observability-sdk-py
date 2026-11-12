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

# Agent Traces lists a trace by its AI root: a trace root that is a GenAI span
# (has gen_ai.operation.name), or a span marked with derived.gen_ai.is_root
# (by Data Prepper, an SDK or the user), for example the agent span below an
# HTTP server span.
_AI_ROOT_FIELD = f"attributes.{AGENT_ROOT_ATTRIBUTE}"
_OPERATION_FIELD = "attributes.gen_ai.operation.name"
_CONVERSATION_FIELD = "attributes.gen_ai.conversation.id"

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
    agent_version: str = ""
    model: str = ""
    input_messages: list[Message] = field(default_factory=list)
    output_messages: list[Message] = field(default_factory=list)
    tool_name: str = ""
    tool_call_arguments: str = ""
    tool_call_result: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


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


def _drop_shadowed_ai_roots(docs: list[dict]) -> list[dict]:
    """Drop marked spans whose trace is also listed through its GenAI root.

    Only spans without a parent pass the GenAI-root condition, so a trace
    with such a hit keeps that one; marked spans deeper in the same trace
    (an SDK marking a nested span) are dropped, as in Agent Traces.
    """
    root_trace_ids = {d.get("traceId") for d in docs if not d.get("parentSpanId")}
    return [d for d in docs if not d.get("parentSpanId") or d.get("traceId") not in root_trace_ids]


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
        """List the AI roots of recent traces, newest first.

        Uses the Agent Traces rule: a span is an AI root when it is a trace
        root that is a GenAI span (no parent, has ``gen_ai.operation.name``)
        or it is marked with ``derived.gen_ai.is_root``. For an agent behind
        an HTTP handler this returns the marked agent span (with its messages
        and agent name), not the HTTP server span. A marked span in a trace
        that also has a GenAI root is left out, so a trace is listed once,
        unless several agents in it are marked (one record each).

        Indices where the marker is not mapped yet (no marked span so far)
        are fine: the marker condition matches nothing there.

        Args:
            services: Only return traces from these service names.
            since: Only return traces started after this time.
                Defaults to 15 minutes ago.
            max_results: Maximum number of root spans to return.
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(minutes=15)
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        ai_root = {
            "bool": {
                "should": [
                    {
                        "bool": {
                            "filter": [
                                {"term": {"parentSpanId": ""}},
                                {"exists": {"field": _OPERATION_FIELD}},
                            ]
                        }
                    },
                    {"term": {_AI_ROOT_FIELD: True}},
                ],
                "minimum_should_match": 1,
            }
        }
        must: list[dict] = [
            ai_root,
            {"range": {"startTime": {"gte": since_iso}}},
        ]
        if services:
            must.append({"terms": {"serviceName": services}})

        body = {
            # A shadowed marked span takes a slot in the page; fetch extra so
            # dropping it still leaves max_results roots when there are more.
            "size": min(max_results * 2, 10000),
            "sort": [{"startTime": "desc"}],
            "query": {"bool": {"must": must}},
        }
        docs = _drop_shadowed_ai_roots(self._search(body))
        return [map_span_doc(doc) for doc in docs[:max_results]]

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
