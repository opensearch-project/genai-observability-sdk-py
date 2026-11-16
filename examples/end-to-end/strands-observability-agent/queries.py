# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Read-only reference queries against the user's OpenSearch cluster.

The agent's tools call these, and the offline scorers call them again at
eval time to compute the ground truth live (nothing is mocked).
"""

from __future__ import annotations

from typing import Any

from config import LOG_INDEX, SPAN_INDEX, opensearch_client

# OTLP status code of an errored span (Data Prepper stores status.code as an int).
STATUS_ERROR = 2

_client: Any = None


def client() -> Any:
    global _client
    if _client is None:
        _client = opensearch_client()
    return _client


def _since(minutes: int) -> dict[str, Any]:
    return {"range": {"startTime": {"gte": f"now-{int(minutes)}m"}}}


def list_indices() -> list[dict[str, Any]]:
    """Non-system indices with their document counts, largest first."""
    rows = client().cat.indices(format="json", h="index,docs.count,status,health")
    out = [
        {
            "index": r["index"],
            "docs": int(r.get("docs.count") or 0),
            "status": r.get("status"),
            "health": r.get("health"),
        }
        for r in rows
        if not r["index"].startswith(".")
    ]
    return sorted(out, key=lambda r: (-r["docs"], r["index"]))


def services(minutes: int) -> list[str]:
    """Service names that sent spans in the window."""
    body = {
        "size": 0,
        "query": _since(minutes),
        "aggs": {"svc": {"terms": {"field": "serviceName", "size": 100}}},
    }
    resp = client().search(index=SPAN_INDEX, body=body, ignore_unavailable=True)
    return [b["key"] for b in resp.get("aggregations", {}).get("svc", {}).get("buckets", [])]


def error_counts_by_service(minutes: int) -> dict[str, int]:
    """Errored spans (status.code = 2) per service in the last ``minutes``."""
    body = {
        "size": 0,
        "query": {"bool": {"filter": [_since(minutes), {"term": {"status.code": STATUS_ERROR}}]}},
        "aggs": {"svc": {"terms": {"field": "serviceName", "size": 100}}},
    }
    resp = client().search(index=SPAN_INDEX, body=body, ignore_unavailable=True)
    buckets = resp.get("aggregations", {}).get("svc", {}).get("buckets", [])
    return {b["key"]: int(b["doc_count"]) for b in buckets}


def span_count(minutes: int) -> int:
    resp = client().count(index=SPAN_INDEX, body={"query": _since(minutes)})
    return int(resp.get("count", 0))


def top_slow_operations(minutes: int, limit: int = 5) -> list[dict[str, Any]]:
    """Operations (span names) by average duration, slowest first."""
    body = {
        "size": 0,
        "query": _since(minutes),
        "aggs": {
            "op": {
                "terms": {
                    "field": "name",
                    "size": int(limit),
                    "order": {"avg_ns": "desc"},
                },
                "aggs": {
                    "avg_ns": {"avg": {"field": "durationInNanos"}},
                    "max_ns": {"max": {"field": "durationInNanos"}},
                    "svc": {"terms": {"field": "serviceName", "size": 3}},
                },
            }
        },
    }
    resp = client().search(index=SPAN_INDEX, body=body, ignore_unavailable=True)
    out = []
    for b in resp.get("aggregations", {}).get("op", {}).get("buckets", []):
        out.append(
            {
                "operation": b["key"],
                "count": int(b["doc_count"]),
                "avg_ms": round((b["avg_ns"]["value"] or 0) / 1e6, 1),
                "max_ms": round((b["max_ns"]["value"] or 0) / 1e6, 1),
                "services": [s["key"] for s in b["svc"]["buckets"]],
            }
        )
    return out


def get_trace(trace_id: str, max_spans: int = 200) -> list[dict[str, Any]]:
    """The spans of one trace, in start order."""
    body = {
        "size": int(max_spans),
        "query": {"term": {"traceId": trace_id}},
        "sort": [{"startTime": "asc"}],
        "_source": [
            "spanId",
            "parentSpanId",
            "name",
            "serviceName",
            "startTime",
            "durationInNanos",
            "status.code",
        ],
    }
    resp = client().search(index=SPAN_INDEX, body=body, ignore_unavailable=True)
    spans = []
    for hit in resp.get("hits", {}).get("hits", []):
        s = hit["_source"]
        spans.append(
            {
                "span_id": s.get("spanId"),
                "parent_span_id": s.get("parentSpanId") or None,
                "name": s.get("name"),
                "service": s.get("serviceName"),
                "start": s.get("startTime"),
                "duration_ms": round((s.get("durationInNanos") or 0) / 1e6, 1),
                "error": (s.get("status") or {}).get("code") == STATUS_ERROR,
            }
        )
    return spans


def latest_root_trace(exclude_service: str | None = None) -> dict[str, Any] | None:
    """The most recent root span (any service except ``exclude_service``)."""
    must_not = [{"term": {"serviceName": exclude_service}}] if exclude_service else []
    body = {
        "size": 1,
        "query": {"bool": {"filter": [{"term": {"parentSpanId": ""}}], "must_not": must_not}},
        "sort": [{"startTime": "desc"}],
        "_source": ["traceId", "name", "serviceName"],
    }
    resp = client().search(index=SPAN_INDEX, body=body, ignore_unavailable=True)
    hits = resp.get("hits", {}).get("hits", [])
    return hits[0]["_source"] if hits else None


def search_logs(query: str, minutes: int, limit: int = 20) -> dict[str, Any]:
    """Log records matching ``query`` (Lucene syntax) in the last ``minutes``."""
    body = {
        "size": int(limit),
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [{"range": {"time": {"gte": f"now-{int(minutes)}m"}}}],
                "must": [{"query_string": {"query": query or "*", "default_field": "body"}}],
            }
        },
        "sort": [{"time": "desc"}],
    }
    resp = client().search(
        index=LOG_INDEX, body=body, ignore_unavailable=True, allow_no_indices=True
    )
    hits = resp.get("hits", {})
    total = hits.get("total", {})
    rows = []
    for hit in hits.get("hits", []):
        s = hit["_source"]
        rows.append(
            {
                "time": s.get("time"),
                "severity": s.get("severityText"),
                "service": s.get("serviceName")
                or (s.get("resource") or {}).get("attributes", {}).get("service.name"),
                "body": str(s.get("body"))[:500],
            }
        )
    return {"total": int(total.get("value", 0) if isinstance(total, dict) else total), "logs": rows}


def index_exists(index: str) -> bool:
    return bool(client().indices.exists(index=index))
