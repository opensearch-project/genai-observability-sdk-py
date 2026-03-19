# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Shared components for the demo scripts.

- OTLPJsonExporter: sends spans as JSON to agent-health (port 4001)
- Bedrock agent: real LLM calls with @observe tracing
- Knowledge base and test cases
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import boto3
import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from opensearch_genai_observability_sdk_py import Op, enrich, observe

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_HEALTH_URL = os.environ.get("AGENT_HEALTH_URL", "http://localhost:4001")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0")


# ---------------------------------------------------------------------------
# OTLP JSON Exporter — agent-health accepts JSON, not protobuf
# ---------------------------------------------------------------------------


class OTLPJsonExporter(SpanExporter):
    """Sends spans as OTLP JSON so agent-health can parse them."""

    def __init__(self, endpoint: str | None = None) -> None:
        self._endpoint = endpoint or f"{AGENT_HEALTH_URL}/v1/traces"

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        resource_spans = self._build_resource_spans(spans)
        payload = {"resourceSpans": resource_spans}
        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            return SpanExportResult.SUCCESS if resp.status_code == 200 else SpanExportResult.FAILURE
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def _build_resource_spans(self, spans: Sequence[ReadableSpan]) -> list[dict]:
        by_resource: dict[str, list[ReadableSpan]] = {}
        for span in spans:
            key = str(id(span.resource))
            by_resource.setdefault(key, []).append(span)

        result = []
        for _, group in by_resource.items():
            resource_attrs = []
            if group[0].resource:
                for k, v in group[0].resource.attributes.items():
                    resource_attrs.append({"key": k, "value": self._attr_value(v)})

            scope_spans = [self._convert_span(s) for s in group]
            result.append({
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [{"spans": scope_spans}],
            })
        return result

    def _convert_span(self, span: ReadableSpan) -> dict:
        ctx = span.get_span_context()
        attrs = [{"key": k, "value": self._attr_value(v)} for k, v in (span.attributes or {}).items()]

        events = []
        for event in span.events or []:
            event_attrs = [{"key": k, "value": self._attr_value(v)} for k, v in (event.attributes or {}).items()]
            events.append({
                "name": event.name,
                "timeUnixNano": str(event.timestamp or 0),
                "attributes": event_attrs,
            })

        links = []
        for link in span.links or []:
            link_attrs = [{"key": k, "value": self._attr_value(v)} for k, v in (link.attributes or {}).items()]
            links.append({
                "traceId": format(link.context.trace_id, '032x'),
                "spanId": format(link.context.span_id, '016x'),
                "attributes": link_attrs,
            })

        parent_id = ""
        if span.parent and span.parent.span_id:
            parent_id = format(span.parent.span_id, '016x')

        d = {
            "traceId": format(ctx.trace_id, '032x'),
            "spanId": format(ctx.span_id, '016x'),
            "parentSpanId": parent_id,
            "name": span.name,
            "startTimeUnixNano": str(span.start_time or 0),
            "endTimeUnixNano": str(span.end_time or 0),
            "attributes": attrs,
            "events": events,
            "status": {"code": 2 if span.status and span.status.status_code.value == 2 else 0},
        }
        if links:
            d["links"] = links
        return d

    def _attr_value(self, v: Any) -> dict:
        if isinstance(v, str):
            return {"stringValue": v}
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        if isinstance(v, (list, tuple)):
            return {"arrayValue": {"values": [self._attr_value(x) for x in v]}}
        return {"stringValue": str(v)}


# ---------------------------------------------------------------------------
# Knowledge base — simple in-memory docs about OpenSearch
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "opensearch-overview",
        "content": (
            "OpenSearch is an open-source search and analytics suite forked from "
            "Elasticsearch 7.10 and Kibana 7.10 under the Apache 2.0 license. "
            "It provides distributed search, log analytics, observability, and "
            "security analytics. Key differences from Elasticsearch include Apache 2.0 "
            "licensing, built-in security plugin, community-driven governance under the "
            "Linux Foundation, and native ML capabilities via ML Commons."
        ),
    },
    {
        "id": "vector-search",
        "content": (
            "Vector search in OpenSearch uses the k-NN plugin. To set it up: "
            "1) Enable the k-NN plugin (enabled by default). "
            "2) Create an index with a knn_vector field, specifying dimension and "
            "engine (nmslib, faiss, or lucene). "
            "3) Index documents with vector embeddings. "
            "4) Search using knn query with a target vector and k parameter. "
            "Supported distance functions include L2 (euclidean), cosine similarity, "
            "and inner product."
        ),
    },
    {
        "id": "observability",
        "content": (
            "OpenSearch provides comprehensive observability features: "
            "log analytics with log ingestion and exploration, "
            "trace analytics supporting OpenTelemetry, Jaeger, and Zipkin formats, "
            "metrics monitoring with Prometheus integration, "
            "alerting with configurable monitors and notifications, "
            "anomaly detection using Random Cut Forest algorithm, "
            "and OpenSearch Dashboards for visualization."
        ),
    },
    {
        "id": "security",
        "content": (
            "OpenSearch Security plugin provides: TLS/SSL encryption for data in transit, "
            "multiple authentication backends (internal users, LDAP, Active Directory, "
            "SAML, OpenID Connect), fine-grained role-based access control (RBAC), "
            "field-level and document-level security for multi-tenancy, "
            "audit logging for compliance, and cross-cluster search permissions."
        ),
    },
    {
        "id": "ml-commons",
        "content": (
            "ML Commons is an OpenSearch plugin for running ML models within the cluster. "
            "It supports: local model deployment (uploading models like sentence transformers), "
            "external model connectors (OpenAI, Bedrock, SageMaker, Cohere), "
            "built-in algorithms (anomaly detection, clustering, regression), "
            "and ML inference pipelines for real-time enrichment. "
            "ML Commons enables neural search by generating embeddings at index and query time."
        ),
    },
]


# ---------------------------------------------------------------------------
# Test cases — questions about OpenSearch
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "input": "What is OpenSearch and how is it different from Elasticsearch?",
        "expected": (
            "OpenSearch is an open-source search and analytics suite forked from "
            "Elasticsearch 7.10 under Apache 2.0. Key differences: open-source licensing, "
            "built-in security, community governance, native ML capabilities."
        ),
        "case_id": "opensearch_vs_es",
        "case_name": "OpenSearch vs Elasticsearch",
    },
    {
        "input": "How do I set up vector search in OpenSearch?",
        "expected": (
            "Enable k-NN plugin, create index with knn_vector field specifying dimension "
            "and engine (nmslib/faiss/lucene), index vectors, search using knn query."
        ),
        "case_id": "vector_search_setup",
        "case_name": "Vector search setup",
    },
    {
        "input": "What observability features does OpenSearch provide?",
        "expected": (
            "Log analytics, trace analytics (OTel/Jaeger/Zipkin), metrics monitoring, "
            "alerting, anomaly detection, and dashboards for visualization."
        ),
        "case_id": "observability_features",
        "case_name": "Observability features",
    },
    {
        "input": "How does OpenSearch handle authentication and security?",
        "expected": (
            "Security plugin provides TLS encryption, authentication (LDAP, SAML, OpenID), "
            "role-based access control, field/document-level security, audit logging."
        ),
        "case_id": "security_features",
        "case_name": "Security features",
    },
    {
        "input": "What is the ML Commons framework in OpenSearch?",
        "expected": (
            "ML Commons plugin for running ML models in OpenSearch. Supports local models, "
            "external connectors (OpenAI, Bedrock, SageMaker), built-in algorithms, "
            "and ML inference pipelines for neural search."
        ),
        "case_id": "ml_commons",
        "case_name": "ML Commons framework",
    },
]


# ---------------------------------------------------------------------------
# Retrieval — simple keyword-based document lookup
# ---------------------------------------------------------------------------


@observe(op=Op.RETRIEVAL)
def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Retrieve relevant documents from the knowledge base."""
    enrich(tool_name="opensearch_vector_search")

    query_lower = query.lower()
    scored = []
    for doc in KNOWLEDGE_BASE:
        words = doc["content"].lower().split()
        query_words = query_lower.split()
        overlap = sum(1 for w in query_words if w in words)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# Bedrock LLM call
# ---------------------------------------------------------------------------


@observe(op=Op.CHAT)
def call_bedrock(prompt: str, system: str = "", model: str | None = None) -> str:
    """Call AWS Bedrock Claude and return the response text."""
    model_id = model or BEDROCK_MODEL
    enrich(model=model_id, system="aws.bedrock")

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
    }
    if system:
        body["system"] = system

    response = client.invoke_model(modelId=model_id, body=json.dumps(body))
    result = json.loads(response["body"].read())

    answer = result["content"][0]["text"]
    input_tokens = result.get("usage", {}).get("input_tokens", 0)
    output_tokens = result.get("usage", {}).get("output_tokens", 0)
    enrich(input_tokens=input_tokens, output_tokens=output_tokens)

    return answer


# ---------------------------------------------------------------------------
# Agent — RAG pipeline (retrieve → generate)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "basic": "Answer the question based on the provided context. Be brief.",
    "detailed": (
        "You are an OpenSearch expert. Answer the question based on the provided "
        "context. Be thorough, accurate, and cite specific features. "
        "Structure your answer clearly."
    ),
}


def make_agent(prompt_version: str = "basic", model: str | None = None):
    """Create an agent function with the given prompt version."""
    system_prompt = SYSTEM_PROMPTS.get(prompt_version, SYSTEM_PROMPTS["basic"])

    @observe(name="opensearch-qa-agent", op=Op.INVOKE_AGENT)
    def agent(question: str) -> str:
        docs = retrieve(question)
        context = "\n\n".join(f"[{d['id']}]: {d['content']}" for d in docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return call_bedrock(prompt, system=system_prompt, model=model)

    return agent
