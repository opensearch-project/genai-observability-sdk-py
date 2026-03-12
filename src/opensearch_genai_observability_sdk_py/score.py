# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Score submission for OpenSearch AI observability.

Sends evaluation scores as OTel spans using gen_ai.evaluation.*
semantic convention attributes.

Two scoring levels:

- **Span-level:** ``trace_id`` + ``span_id`` — score a specific span.
  The score span is created as a child of that span (same trace).
- **Trace-level:** ``trace_id`` only — score the entire trace.
  The score span is attached to the root span of that trace.

In both cases the score span joins the evaluated trace, so it appears
alongside the execution spans in the trace waterfall.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-observability-sdk-py-scores"


def score(
    name: str,
    value: float | None = None,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
    source: str = "sdk",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Submit an evaluation score as an OTel span.

    Creates a ``gen_ai.evaluation.*`` span that is attached to the
    evaluated trace, so scores appear in the same trace waterfall as the
    spans they evaluate.

    Two scoring levels:

    - **Span-level** (``trace_id`` + ``span_id``): the score span is a
      child of the specified span.
    - **Trace-level** (``trace_id`` only): the score span is attached to
      the root span of the trace.

    If neither ``trace_id`` nor ``span_id`` is given, a standalone score
    span is emitted with no trace context.

    Args:
        name: Evaluation metric name (e.g., ``"relevance"``, ``"factuality"``).
        value: Numeric score value.
        trace_id: Hex trace ID of the trace being scored.
        span_id: Hex span ID of the specific span being scored.
            When omitted, the score attaches to the root span of the trace.
        label: Human-readable label (e.g., ``"pass"``, ``"relevant"``).
        explanation: Evaluator justification or rationale (truncated to 500 chars).
        response_id: Completion ID for correlation with a specific LLM response.
        source: Who created the score — ``"sdk"``, ``"human"``,
            ``"llm-judge"``, ``"heuristic"``.
        metadata: Optional arbitrary key-value metadata.

    Examples:
        # Span-level: score a specific span
        score(
            name="accuracy",
            value=0.95,
            trace_id="6ebb9835f43af1552f2cebb9f5165e39",
            span_id="89829115c2128845",
            explanation="Weather data is correct",
            source="heuristic",
        )

        # Trace-level: score the whole trace (attaches to root span)
        score(
            name="relevance",
            value=0.92,
            trace_id="6ebb9835f43af1552f2cebb9f5165e39",
            explanation="Response addresses the query",
            source="llm-judge",
        )
    """
    tracer = trace.get_tracer(_TRACER_NAME)

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "evaluation",
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.source": source,
    }

    if value is not None:
        attrs["gen_ai.evaluation.score.value"] = value
    if label:
        attrs["gen_ai.evaluation.score.label"] = label
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = explanation[:500]
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if metadata:
        for k, v in metadata.items():
            attrs[f"gen_ai.evaluation.metadata.{k}"] = str(v)

    ctx = _build_parent_context(trace_id, span_id)

    with tracer.start_as_current_span(
        f"evaluation {name}",
        context=ctx,
        attributes=attrs,
    ):
        logger.debug("Score emitted: %s=%s (trace=%s)", name, value, trace_id)


def _build_parent_context(
    trace_id: str | None,
    span_id: str | None,
) -> Any:
    """Build an OTel context that parents the score span to the evaluated span.

    - Both provided → attach to that specific span.
    - Only trace_id → attach to the root span (derived from trace_id).
    - Neither → return None (standalone span, no parent context).
    """
    if not trace_id:
        return None

    trace_id_int = _parse_hex(trace_id)
    if trace_id_int is None:
        logger.warning("score(): invalid trace_id '%s', emitting standalone span", trace_id)
        return None

    # For trace-level scoring (no span_id), derive the root span_id from
    # the lower 64 bits of the trace_id — a standard convention for root spans.
    if span_id:
        span_id_int = _parse_hex(span_id)
        if span_id_int is None:
            logger.warning("score(): invalid span_id '%s', emitting standalone span", span_id)
            return None
    else:
        span_id_int = trace_id_int & 0xFFFFFFFFFFFFFFFF  # lower 64 bits

    parent_span_context = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(parent_span_context))


def _parse_hex(value: str) -> int | None:
    """Parse a hex string (with or without 0x prefix) to int. Returns None on failure."""
    try:
        return int(value.lstrip("0x").lstrip("0X") or "0", 16)
    except ValueError:
        return None
