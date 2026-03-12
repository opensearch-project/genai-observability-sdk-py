# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Scoring with opensearch-genai-observability-sdk-py.

Shows how to submit evaluation scores as OTEL spans. Scores flow through
the same exporter pipeline as traces — same SigV4 auth, same Data Prepper
endpoint. Supports three scoring levels: span, trace, and session.
"""

from opensearch_genai_observability_sdk_py import register, score

# --- Setup ---
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")


# --- Span-level score (score a specific LLM call or tool execution) ---
score(
    name="relevance",
    value=0.95,
    trace_id="abc123def456",
    span_id="789abc",
    source="llm-judge",
    explanation="Answer directly addresses the question with correct facts",
)


# --- Trace-level score (score an entire workflow) ---
score(
    name="quality",
    value=0.88,
    trace_id="abc123def456",
    label="good",
    source="human",
    explanation="Reviewed by QA team, response is accurate and well-formatted",
)


# --- Session-level score (score across multiple traces in a conversation) ---
score(
    name="user_satisfaction",
    value=0.92,
    conversation_id="session-456",
    label="satisfied",
    source="human",
)


# --- Score with metadata ---
score(
    name="latency_check",
    value=1.0,
    trace_id="abc123def456",
    source="heuristic",
    metadata={"threshold_ms": 500, "actual_ms": 120},
)


# Each score() call creates an OTEL span like:
#
#   Span: gen_ai.evaluation.result
#   Attributes:
#     gen_ai.evaluation.name = "relevance"
#     gen_ai.evaluation.score.value = 0.95
#     gen_ai.evaluation.trace_id = "abc123def456"
#     gen_ai.evaluation.span_id = "789abc"
#     gen_ai.evaluation.source = "llm-judge"
#     gen_ai.evaluation.explanation = "Answer directly addresses..."
