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

if TYPE_CHECKING:
    from opensearchpy import RequestsAWSV4SignerAuth

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# OpenSearch client wrapper
# ---------------------------------------------------------------------------


class OpenSearchTraceRetriever:
    """Query OpenSearch for GenAI trace spans and return SessionRecords.

    Args:
        host: OpenSearch endpoint (e.g. ``https://localhost:9200``).
        index: Index pattern for span data.
        auth: Authentication — either a ``(username, password)`` tuple for
            basic auth, or a ``RequestsAWSV4SignerAuth`` instance for SigV4.
        verify_certs: Whether to verify TLS certificates.
    """

    def __init__(
        self,
        host: str = "https://localhost:9200",
        index: str = "otel-v1-apm-span-*",
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

    @staticmethod
    def _get_requests_connection() -> Any:
        from opensearchpy import RequestsHttpConnection

        return RequestsHttpConnection

    def get_traces(self, identifier: str, max_spans: int = 10000) -> SessionRecord:
        """Retrieve all spans for a session or trace, grouped by traceId.

        Queries by ``gen_ai.conversation.id`` first. If no results, falls back
        to treating *identifier* as a ``traceId``.

        Args:
            identifier: A conversation/session ID or a trace ID.
            max_spans: Maximum spans to fetch. If the result is truncated,
                ``SessionRecord.truncated`` will be ``True``.
        """
        docs = self._query_by_conversation_id(identifier, max_spans)
        if not docs:
            docs = self._query_by_trace_id(identifier, max_spans)
        if not docs:
            return SessionRecord(session_id=identifier)

        session = self._build_session(identifier, docs)
        session.truncated = len(docs) >= max_spans
        return session

    def list_root_spans(
        self,
        *,
        services: list[str] | None = None,
        since: datetime | None = None,
        max_results: int = 50,
    ) -> list[SpanRecord]:
        """List recent root spans, optionally filtered by service name.

        Returns one ``SpanRecord`` per trace (the root span with no parent).
        Useful for discovering traces to evaluate.

        Args:
            services: Only return traces from these service names.
            since: Only return traces started after this time.
                Defaults to 15 minutes ago.
            max_results: Maximum number of root spans to return.
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(minutes=15)
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        must: list[dict] = [
            {"term": {"parentSpanId": ""}},
            {"range": {"startTime": {"gte": since_iso}}},
        ]
        if services:
            must.append({"terms": {"serviceName": services}})

        body = {
            "size": max_results,
            "sort": [{"startTime": "desc"}],
            "query": {"bool": {"must": must}},
        }
        return [map_span_doc(doc) for doc in self._search(body)]

    def find_evaluated_trace_ids(self, trace_ids: list[str]) -> set[str]:
        """Return the subset of trace_ids that already have an evaluation span."""
        if not trace_ids:
            return set()
        body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"traceId": trace_ids}},
                        {"term": {"attributes.gen_ai.operation.name": "evaluation"}},
                    ],
                }
            },
            "aggs": {"evaluated": {"terms": {"field": "traceId", "size": len(trace_ids)}}},
        }
        resp = self._client.search(index=self._index, body=body)
        return {
            b["key"] for b in resp.get("aggregations", {}).get("evaluated", {}).get("buckets", [])
        }

    # --- queries ---

    def _query_by_conversation_id(self, conversation_id: str, size: int) -> list[dict]:
        body = {
            "size": size,
            "query": {"term": {"attributes.gen_ai.conversation.id": conversation_id}},
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
