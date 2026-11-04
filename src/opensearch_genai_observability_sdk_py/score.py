# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Evaluation results for OpenSearch AI observability.

``score()`` records one evaluation result as an OTel **log-based event**
named ``gen_ai.evaluation.result``, following the `OTel GenAI semantic
conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/>`_.
The event is a LogRecord exported over OTLP logs (``/v1/logs``) and carries
the trace context of the **evaluated** span, so it joins the trace without
creating a new span. Data Prepper routes these events to the
``otel-v1-genai-evaluation`` index.

Scoring levels:

- **Span-level:** ``trace_id`` + ``span_id``, or call ``score()`` while the
  evaluated span is active. The event carries that span's ids.
- **Trace-level:** ``trace_id`` only. The event's own trace id is that trace
  and it has no span id (none is derived).
- **Session-level:** ``conversation_id`` (``gen_ai.conversation.id``) with
  ``level="session"``.

The instrumentation scope of every event is the SDK
(``opensearch-genai-observability-sdk-py`` and the SDK version), the
instrumentation library as OTel defines it. The evaluator identity is in the
``opensearch.gen_ai.evaluation.evaluator.*`` attributes. Data Prepper and OpenSearch
Dashboards derive the result source from the scope name.

Verdicts: readers show pass/fail only for a declared rule, otherwise the
result is neutral ("Scored"). Declare it with ``label="pass"`` / ``"fail"``
(``gen_ai.evaluation.score.label``, OTel's own examples) or with
``threshold`` (+ ``direction``): with no label, the SDK writes the verdict
the threshold gives as the label (see ``threshold_verdict``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.trace import INVALID_SPAN_ID, NonRecordingSpan, SpanContext, TraceFlags

from opensearch_genai_observability_sdk_py._internal import (
    AGENT_NAME_KEY,
    AGENT_VERSION_KEY,
    get_event_logger,
    parse_hex,
    resolve_agent,
)

logger = logging.getLogger(__name__)

EVALUATION_EVENT_NAME = "gen_ai.evaluation.result"

# Safety cap for ``gen_ai.evaluation.explanation``. Judge rationales are kept
# whole up to this many characters; longer values are cut and end with
# ``_TRUNCATION_MARKER``.
EXPLANATION_MAX_CHARS = 8192
_TRUNCATION_MARKER = "...[truncated]"

# OpenSearch attributes for evaluation fields OTel does not define yet. They
# mirror the OTel ``gen_ai.evaluation.*`` names under ``opensearch.``.
EVAL_LEVEL_KEY = "opensearch.gen_ai.evaluation.level"
EVAL_EVALUATOR_NAME_KEY = "opensearch.gen_ai.evaluation.evaluator.name"
EVAL_EVALUATOR_VERSION_KEY = "opensearch.gen_ai.evaluation.evaluator.version"
EVAL_EVALUATOR_TYPE_KEY = "opensearch.gen_ai.evaluation.evaluator.type"
EVAL_SCORE_DIRECTION_KEY = "opensearch.gen_ai.evaluation.score.direction"
EVAL_SCORE_MIN_KEY = "opensearch.gen_ai.evaluation.score.min"
EVAL_SCORE_MAX_KEY = "opensearch.gen_ai.evaluation.score.max"
EVAL_SCORE_THRESHOLD_KEY = "opensearch.gen_ai.evaluation.score.threshold"
EVAL_JUDGE_TRACE_ID_KEY = "opensearch.gen_ai.evaluation.judge.trace_id"

_VALID_LEVELS = ("span", "trace", "session", "tool_call")
_VALID_EVALUATOR_TYPES = ("llm_judge", "code", "human", "external")
_VALID_DIRECTIONS = ("maximize", "minimize")
VALID_VERDICTS = ("pass", "fail")


def _no_span_context() -> Context:
    """A context that explicitly holds no valid span.

    An empty ``Context()`` is falsy, and the Logs SDK then falls back to the
    current context (the active span), so set ``INVALID_SPAN`` explicitly.
    """
    return trace.set_span_in_context(trace.INVALID_SPAN, Context())


def score(
    name: str,
    value: float | None = None,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
    attributes: dict[str, Any] | None = None,
    level: str | None = None,
    evaluator: str | None = None,
    evaluator_version: str | None = None,
    evaluator_type: str | None = None,
    direction: str | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    conversation_id: str | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
    error_type: str | None = None,
    judge_trace_id: str | None = None,
    threshold: float | None = None,
) -> None:
    """Record an evaluation result as a ``gen_ai.evaluation.result`` event.

    Emits one OTel LogRecord through the Logs API with ``event_name`` set to
    ``gen_ai.evaluation.result``, ``timestamp`` set to now, severity INFO and
    all data in attributes. The record is parented to the evaluated span:

    - ``trace_id`` + ``span_id`` given: that span (works for spans that have
      already ended, for example offline or async evaluation).
    - nothing given: the currently active span, if any (inline evaluation).
    - ``trace_id`` only: the record's own trace id is that trace, with no
      span id, and the level defaults to ``trace``.

    The instrumentation scope is always the SDK
    (``opensearch-genai-observability-sdk-py`` with the SDK version); the
    evaluator is named in ``opensearch.gen_ai.evaluation.evaluator.*``. Data Prepper
    and OpenSearch Dashboards derive the result source from the scope.

    ``register()`` configures the OTLP logs pipeline. With your own OTel
    setup, configure a global ``LoggerProvider`` or the events are dropped.

    Attributes written (only when the value is given):

    - Standard: ``gen_ai.evaluation.name``, ``gen_ai.evaluation.score.value``,
      ``gen_ai.evaluation.score.label``, ``gen_ai.evaluation.explanation``,
      ``gen_ai.response.id``, ``error.type``, ``gen_ai.conversation.id``,
      ``gen_ai.agent.name``.
    - Extensions (until the upstream proposals merge):
      ``opensearch.gen_ai.evaluation.level``, ``.evaluator.name``,
      ``.evaluator.version``, ``.evaluator.type``, ``.score.direction``,
      ``.score.min``, ``.score.max``, ``.score.threshold``,
      ``.judge.trace_id``.

    Verdict: ``gen_ai.evaluation.score.label`` = ``pass`` / ``fail``. Pass it
    as ``label``. Otherwise, when both ``value`` and ``threshold`` are given
    and no ``label`` is, the SDK writes it: ``pass`` if ``value >= threshold``
    (``direction="maximize"``, the default with a threshold) or
    ``value <= threshold`` (``direction="minimize"``), else ``fail``. With
    neither, no verdict is recorded and readers show the result as neutral.

    Args:
        name: Evaluation metric name (e.g., ``"relevance"``, ``"factuality"``).
        value: Numeric score value.
        trace_id: Hex trace ID of the evaluated trace.
        span_id: Hex span ID of the evaluated span. Without it the result is
            trace-level (no span id is ever derived from the trace id).
        label: Low-cardinality label (e.g., ``"relevant"``). ``"pass"`` /
            ``"fail"`` is the verdict.
        explanation: Evaluator rationale. Kept whole up to
            ``EXPLANATION_MAX_CHARS`` (8192) characters.
        response_id: Completion ID of the evaluated LLM response.
        attributes: Extra attributes added as-is (e.g., ``test.*`` grouping
            keys from semantic-conventions-genai#79).
        level: ``"span"``, ``"trace"``, ``"session"`` or ``"tool_call"``.
            Defaults to ``"span"`` with a span context, ``"trace"`` with only
            ``trace_id``, ``"session"`` with only ``conversation_id``.
        evaluator: Evaluator name; also used as the instrumentation scope.
        evaluator_version: Evaluator version (instrumentation scope version).
        evaluator_type: ``"llm_judge"``, ``"code"``, ``"human"`` or ``"external"``.
        direction: ``"maximize"`` or ``"minimize"``: which way is better.
        score_min: Lower bound of the score scale.
        score_max: Upper bound of the score scale.
        conversation_id: ``gen_ai.conversation.id`` of the evaluated session.
        agent_name: ``gen_ai.agent.name`` of the evaluated agent. When omitted:
            the evaluated span's own ``gen_ai.agent.name`` (inline scoring of
            the active span), else the ``register(agent_name=)`` default,
            else left out.
        agent_version: ``gen_ai.agent.version`` of the evaluated agent, resolved the
            same way.
        error_type: ``error.type`` when the evaluator itself failed.
        judge_trace_id: Trace id of the judge's own run, if traced.
        threshold: Pass threshold (``opensearch.gen_ai.evaluation.score.threshold``).
            ``direction`` defaults to ``"maximize"`` and is recorded.

    Examples:
        # Inline: score the active span
        with tracer.start_as_current_span("chat gpt-4o"):
            ...
            score("relevance", 0.92, label="relevant", evaluator="relevance_judge")

        # Offline: score a stored span
        score(
            name="accuracy",
            value=0.95,
            trace_id="6ebb9835f43af1552f2cebb9f5165e39",
            span_id="89829115c2128845",
            explanation="Weather data is correct",
            evaluator="accuracy_judge",
            evaluator_type="llm_judge",
            score_min=0.0,
            score_max=1.0,
            direction="maximize",
        )

        # Declared rule: 1-5 judge, pass at 4 or more (the SDK writes the label)
        score("answer_quality", 4.0, threshold=4, score_min=1, score_max=5)

        # Explicit verdict from a code check
        score("tool_selection", 1.0, label="pass", evaluator="tool_check")

        # Trace-level with benchmark grouping
        score(
            name="helpfulness",
            value=0.83,
            trace_id="6ebb9835f43af1552f2cebb9f5165e39",
            label="helpful",
            attributes={"test.suite.run.id": "run_001", "test.case.id": "case_001"},
        )
    """
    ctx, has_span, target_trace_id = _resolve_target(trace_id, span_id)
    # Inline scoring reads the agent from the active span itself; a stored
    # span (trace_id + span_id) is not looked up.
    active_span = trace.get_current_span() if ctx is None and has_span else None
    agent = resolve_agent(agent_name, agent_version, span=active_span)

    resolved_level = level
    if resolved_level is None:
        if has_span:
            resolved_level = "span"
        elif target_trace_id:
            resolved_level = "trace"
        elif conversation_id:
            resolved_level = "session"

    event_attrs = build_evaluation_attributes(
        name=name,
        value=value,
        label=label,
        explanation=explanation,
        response_id=response_id,
        level=resolved_level,
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        evaluator_type=evaluator_type,
        direction=direction,
        score_min=score_min,
        score_max=score_max,
        conversation_id=conversation_id,
        agent_name=agent.get(AGENT_NAME_KEY),
        agent_version=agent.get(AGENT_VERSION_KEY),
        error_type=error_type,
        judge_trace_id=judge_trace_id,
        target_trace_id=target_trace_id,
        threshold=threshold,
        extra=attributes,
    )
    emit_evaluation_event(
        event_attrs,
        context=ctx,
        evaluator=evaluator,
        evaluator_version=evaluator_version,
    )
    logger.debug("Score emitted: %s=%s (trace=%s)", name, value, trace_id)


# ---------------------------------------------------------------------------
# Shared helpers (also used by benchmark.py)
# ---------------------------------------------------------------------------


def build_evaluation_attributes(
    *,
    name: str,
    value: float | None = None,
    label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
    level: str | None = None,
    evaluator: str | None = None,
    evaluator_version: str | None = None,
    evaluator_type: str | None = None,
    direction: str | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    conversation_id: str | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
    error_type: str | None = None,
    judge_trace_id: str | None = None,
    target_trace_id: str | None = None,
    threshold: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``gen_ai.evaluation.result`` attribute map (only set keys)."""
    _check_enum("level", level, _VALID_LEVELS)
    _check_enum("evaluator_type", evaluator_type, _VALID_EVALUATOR_TYPES)
    _check_enum("direction", direction, _VALID_DIRECTIONS)
    if threshold is not None and direction is None:
        direction = "maximize"
    verdict = threshold_verdict(value, threshold=threshold, direction=direction)

    attrs: dict[str, Any] = {"gen_ai.evaluation.name": name}
    if value is not None:
        attrs["gen_ai.evaluation.score.value"] = float(value)
    # The verdict is the label (OTel examples "pass" / "fail"). A label the caller set
    # wins; without one, the verdict the threshold gives is written as the label.
    if label:
        attrs["gen_ai.evaluation.score.label"] = label
    elif verdict:
        attrs["gen_ai.evaluation.score.label"] = verdict
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = _truncate(explanation, EXPLANATION_MAX_CHARS)
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if error_type:
        attrs["error.type"] = error_type
    if conversation_id:
        attrs["gen_ai.conversation.id"] = conversation_id
    if agent_name:
        attrs[AGENT_NAME_KEY] = agent_name
    if agent_version:
        attrs[AGENT_VERSION_KEY] = agent_version
    if level:
        attrs[EVAL_LEVEL_KEY] = level
    if evaluator:
        attrs[EVAL_EVALUATOR_NAME_KEY] = evaluator
    if evaluator_version:
        attrs[EVAL_EVALUATOR_VERSION_KEY] = evaluator_version
    if evaluator_type:
        attrs[EVAL_EVALUATOR_TYPE_KEY] = evaluator_type
    if direction:
        attrs[EVAL_SCORE_DIRECTION_KEY] = direction
    if score_min is not None:
        attrs[EVAL_SCORE_MIN_KEY] = float(score_min)
    if score_max is not None:
        attrs[EVAL_SCORE_MAX_KEY] = float(score_max)
    if threshold is not None:
        attrs[EVAL_SCORE_THRESHOLD_KEY] = float(threshold)
    if judge_trace_id:
        attrs[EVAL_JUDGE_TRACE_ID_KEY] = judge_trace_id
    if extra:
        attrs.update(extra)
    return attrs


def label_verdict(label: str | None) -> str | None:
    """``"pass"`` / ``"fail"`` when the label is a verdict (any case), else ``None``."""
    if not label:
        return None
    normalized = label.strip().lower()
    return normalized if normalized in VALID_VERDICTS else None


def threshold_verdict(
    value: float | None,
    *,
    threshold: float | None = None,
    direction: str | None = None,
) -> str | None:
    """The verdict a pass rule gives: ``"pass"``, ``"fail"`` or ``None``.

    With both ``value`` and ``threshold``: ``pass`` when ``value >= threshold``
    (``direction`` ``"maximize"``, the default) or ``value <= threshold``
    (``"minimize"``), else ``fail``. Else ``None``: no declared rule.
    """
    if value is None or threshold is None:
        return None
    if direction == "minimize":
        return "pass" if float(value) <= float(threshold) else "fail"
    return "pass" if float(value) >= float(threshold) else "fail"


def emit_evaluation_event(
    attributes: dict[str, Any],
    *,
    context: Context | None = None,
    evaluator: str | None = None,
    evaluator_version: str | None = None,
) -> None:
    """Emit one ``gen_ai.evaluation.result`` LogRecord via the global LoggerProvider.

    ``context`` carries the evaluated span (``None`` means the current
    context, i.e. the active span). The scope is always the SDK.
    """
    otel_logger = get_event_logger(evaluator, evaluator_version)
    otel_logger.emit(
        event_name=EVALUATION_EVENT_NAME,
        timestamp=time.time_ns(),
        severity_number=SeverityNumber.INFO,
        attributes=attributes,
        context=context,
    )


def _trace_only_context(trace_id_int: int) -> Context:
    """A Context whose span carries ``trace_id`` and no span id (invalid span id).

    The OTel log record takes the trace id from it and leaves the span id empty,
    so a trace-level result sits on its trace without inventing a span.
    """
    sc = SpanContext(
        trace_id=trace_id_int,
        span_id=INVALID_SPAN_ID,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(sc))


def span_context_for(trace_id: str, span_id: str) -> Context | None:
    """Return a Context whose current span is the (remote) span ``trace_id/span_id``.

    Returns ``None`` if either id fails to parse.
    """
    trace_id_int = parse_hex(trace_id)
    span_id_int = parse_hex(span_id)
    if not trace_id_int or not span_id_int:
        return None
    sc = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(sc))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_target(
    trace_id: str | None,
    span_id: str | None,
) -> tuple[Context | None, bool, str | None]:
    """Work out which span/trace is evaluated.

    Returns ``(context, has_span, target_trace_id)``:

    - ``context`` is passed to ``Logger.emit``.
      ``None`` means "use the current context" (the active span).
      ``_no_span_context()`` means "no span context at all".
    - ``has_span`` is ``True`` when the context carries a valid span.
    - ``target_trace_id`` is the normalized trace id for trace-level results
      (no span id known), else ``None``.
    """
    if not trace_id:
        if span_id:
            logger.warning("score(): span_id given without trace_id; ignoring both ids")
            return _no_span_context(), False, None
        active = trace.get_current_span().get_span_context()
        return None, active.is_valid, None

    trace_id_int = parse_hex(trace_id)
    if not trace_id_int:
        logger.warning("score(): invalid trace_id '%s', emitting without trace context", trace_id)
        return _no_span_context(), False, None
    normalized_trace_id = format(trace_id_int, "032x")

    if span_id:
        ctx = span_context_for(trace_id, span_id)
        if ctx is not None:
            return ctx, True, None
        logger.warning(
            "score(): invalid span_id '%s', recording a trace-level result instead", span_id
        )

    # Trace-level: never invent a span id. The record's own trace id is the
    # target trace and its span id stays empty.
    return _trace_only_context(trace_id_int), False, normalized_trace_id


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= len(_TRUNCATION_MARKER):
        return value[:limit]
    return value[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _check_enum(arg: str, value: str | None, allowed: tuple[str, ...]) -> None:
    if value is not None and value not in allowed:
        logger.warning("score(): %s=%r is not one of %s; recording as-is", arg, value, allowed)
