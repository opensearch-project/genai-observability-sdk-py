# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Experiment tracking for GenAI evaluation results.

Upload evaluation results from any harness (RAGAS, DeepEval, pytest, custom)
as OTel spans with ``test.*`` attributes and ``gen_ai.evaluation.result``
events. Results flow via OTLP to OpenSearch for analysis and comparison.

Two modes of use:

**Direct upload** — you already have results::

    with Experiment(name="rag-v2", metadata={"model": "gpt-4.1"}) as exp:
        exp.log(input="What is Python?", output="A language",
                scores={"accuracy": 1.0})

**Full eval runner** — SDK runs your task and scorers::

    result = evaluate(
        name="rag-v2",
        task=my_agent,
        data=[{"input": "What is Python?", "expected": "programming language"}],
        scores=[accuracy_scorer],
    )

OTel conventions:

- ``test.suite.name``, ``test.suite.run.id`` — experiment identity
  (https://github.com/open-telemetry/semantic-conventions/issues/3398)
- ``test.case.id``, ``test.case.name``, ``test.case.result.status`` — case identity
- ``gen_ai.evaluation.result`` event — score values
  (https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, SpanKind, TraceFlags

logger = logging.getLogger(__name__)

_TRACER_NAME = "opensearch-genai-observability-sdk-py-experiments"


# ---------------------------------------------------------------------------
# Score dataclass — return type for scorer functions used with evaluate()
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Score:
    """Result from a scorer function.

    Returned by scorer callables passed to ``evaluate()``.
    """

    name: str
    value: float
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScoreSummary:
    name: str
    avg: float
    min: float
    max: float
    count: int


@dataclasses.dataclass
class ExperimentSummary:
    experiment_name: str
    run_id: str
    total_cases: int
    error_count: int
    duration_ms: float
    scores: dict[str, ScoreSummary]

    def __str__(self) -> str:
        width = 60
        lines = [
            "=" * width,
            f"  EXPERIMENT: {self.experiment_name}",
            f"  Run: {self.run_id} | Cases: {self.total_cases}"
            f" | Errors: {self.error_count}"
            f" | Duration: {self.duration_ms / 1000:.1f}s",
            "",
        ]
        for s in self.scores.values():
            lines.append(
                f"  {s.name:<20s} avg={s.avg:.2f}  min={s.min:.2f}"
                f"  max={s.max:.2f}  (n={s.count})"
            )
        lines.append("=" * width)
        return "\n".join(lines)


@dataclasses.dataclass
class CaseResult:
    case_id: str
    case_name: str | None
    input: Any
    output: Any
    expected: Any
    scores: dict[str, float]
    error: str | None
    status: str


@dataclasses.dataclass
class EvaluateResult:
    summary: ExperimentSummary
    cases: list[CaseResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_id(input_val: Any) -> str:
    """Derive a stable case ID from the input value."""
    content = json.dumps(input_val, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _make_run_id() -> str:
    """Generate a unique run ID with timestamp prefix."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short_uuid}"


def _build_span_link(
    trace_id_hex: str, span_id_hex: str | None
) -> Link | None:
    """Build an OTel span link from hex trace/span IDs."""
    try:
        trace_id_int = int(trace_id_hex, 16)
    except (ValueError, TypeError):
        logger.warning("Invalid trace_id '%s', skipping span link", trace_id_hex)
        return None

    if span_id_hex:
        try:
            span_id_int = int(span_id_hex, 16)
        except (ValueError, TypeError):
            logger.warning("Invalid span_id '%s', skipping span link", span_id_hex)
            return None
    else:
        span_id_int = trace_id_int & 0xFFFFFFFFFFFFFFFF

    ctx = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return Link(ctx)


def _add_score_events(
    span: trace.Span, scores: dict[str, float]
) -> None:
    """Emit gen_ai.evaluation.result events on a span."""
    for name, value in scores.items():
        event_attrs: dict[str, Any] = {
            "gen_ai.evaluation.name": name,
            "gen_ai.evaluation.score.value": value,
        }
        span.add_event("gen_ai.evaluation.result", attributes=event_attrs)


def _compute_summary(
    experiment_name: str,
    run_id: str,
    cases: list[_CaseRecord],
    start_time: float,
) -> ExperimentSummary:
    """Compute aggregate score statistics from case records."""
    score_values: dict[str, list[float]] = defaultdict(list)
    error_count = 0

    for case in cases:
        if case.error:
            error_count += 1
        for name, value in case.scores.items():
            score_values[name].append(value)

    score_summaries: dict[str, ScoreSummary] = {}
    for name, values in score_values.items():
        score_summaries[name] = ScoreSummary(
            name=name,
            avg=sum(values) / len(values),
            min=min(values),
            max=max(values),
            count=len(values),
        )

    return ExperimentSummary(
        experiment_name=experiment_name,
        run_id=run_id,
        total_cases=len(cases),
        error_count=error_count,
        duration_ms=(time.monotonic() - start_time) * 1000,
        scores=score_summaries,
    )


@dataclasses.dataclass
class _CaseRecord:
    """Internal record for tracking cases before summary computation."""

    case_id: str
    case_name: str | None
    scores: dict[str, float]
    error: str | None


# ---------------------------------------------------------------------------
# Experiment class — direct upload (Mode B)
# ---------------------------------------------------------------------------


class Experiment:
    """Upload evaluation results as OTel experiment spans.

    Creates a root ``test_suite_run`` span and child ``test_case`` spans
    for each ``log()`` call. Results are exported via the configured OTel
    pipeline to OpenSearch.

    Use as a context manager::

        with Experiment(name="rag-v2") as exp:
            exp.log(input="q1", output="a1", scores={"accuracy": 1.0})

    Or manually::

        exp = Experiment(name="rag-v2")
        exp.log(input="q1", output="a1", scores={"accuracy": 1.0})
        summary = exp.close()

    Args:
        name: Experiment name (``test.suite.name``). Stable across runs.
        metadata: Arbitrary metadata attached to the root span.
        record_io: If ``True``, record input/output/expected as span
            attributes. Default ``False`` (PII-safe).
    """

    def __init__(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        record_io: bool = False,
    ) -> None:
        self._name = name
        self._metadata = metadata or {}
        self._record_io = record_io
        self._run_id = _make_run_id()
        self._start_time = time.monotonic()
        self._cases: list[_CaseRecord] = []
        self._closed = False

        tracer = trace.get_tracer(_TRACER_NAME)

        root_attrs: dict[str, Any] = {
            "test.suite.name": name,
            "test.suite.run.id": self._run_id,
            "gen_ai.operation.name": "evaluation",
        }
        for key, val in self._metadata.items():
            root_attrs[key] = val

        self._root_span = tracer.start_span(
            f"test_suite_run {name}",
            kind=SpanKind.INTERNAL,
            attributes=root_attrs,
        )
        self._root_context = trace.set_span_in_context(self._root_span)

        logger.info(
            "Experiment started: name=%s run_id=%s", name, self._run_id
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def run_id(self) -> str:
        return self._run_id

    def log(
        self,
        *,
        input: Any = None,
        output: Any = None,
        expected: Any = None,
        scores: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        case_id: str | None = None,
        case_name: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        """Log a single test case result.

        Creates a ``test_case`` span as a child of the experiment root,
        with ``test.case.*`` attributes and ``gen_ai.evaluation.result``
        events for each score.

        Args:
            input: Test case input data.
            output: Actual output from the agent/system.
            expected: Expected/ground-truth output.
            scores: Pre-computed scores (name → float).
            metadata: Arbitrary case-level metadata.
            error: Error message. Sets status to ``"fail"``.
            case_id: Explicit ``test.case.id``. If omitted, derived
                from a hash of ``input``.
            case_name: Human-readable ``test.case.name``.
            trace_id: Hex trace ID to link to an existing agent trace.
            span_id: Hex span ID to link to a specific agent span.
        """
        if self._closed:
            raise RuntimeError("Experiment is already closed")

        scores = scores or {}
        resolved_case_id = case_id or _make_case_id(input)

        # Determine pass/fail
        if error:
            status = "fail"
        else:
            status = "pass"

        # Build span attributes
        case_attrs: dict[str, Any] = {
            "test.suite.name": self._name,
            "test.suite.run.id": self._run_id,
            "test.case.id": resolved_case_id,
            "test.case.result.status": status,
            "gen_ai.operation.name": "evaluation",
        }
        if case_name:
            case_attrs["test.case.name"] = case_name
        if error:
            case_attrs["test.case.error"] = error
        if metadata:
            for key, val in metadata.items():
                case_attrs[key] = val

        # Record IO if opted in
        if self._record_io:
            if input is not None:
                serialized = json.dumps(input, default=str)
                case_attrs["test.case.input"] = serialized[:10_000]
            if output is not None:
                serialized = json.dumps(output, default=str)
                case_attrs["test.case.output"] = serialized[:10_000]
            if expected is not None:
                serialized = json.dumps(expected, default=str)
                case_attrs["test.case.expected"] = serialized[:10_000]

        # Build span link if trace_id provided
        links: list[Link] = []
        if trace_id:
            link = _build_span_link(trace_id, span_id)
            if link:
                links.append(link)

        # Create the case span as a child of the root
        tracer = trace.get_tracer(_TRACER_NAME)
        case_span = tracer.start_span(
            "test_case",
            context=self._root_context,
            kind=SpanKind.INTERNAL,
            attributes=case_attrs,
            links=links or None,
        )

        # Add score events
        _add_score_events(case_span, scores)

        # Set error status if applicable
        if error:
            case_span.set_status(trace.StatusCode.ERROR, error)

        case_span.end()

        # Track for summary
        self._cases.append(
            _CaseRecord(
                case_id=resolved_case_id,
                case_name=case_name,
                scores=scores,
                error=error,
            )
        )

    def close(self) -> ExperimentSummary:
        """Finalize the experiment and return a summary.

        Ends the root span, computes aggregate statistics, and prints
        a formatted summary to the console.

        Returns:
            ``ExperimentSummary`` with aggregate score statistics.
        """
        if self._closed:
            raise RuntimeError("Experiment is already closed")

        self._closed = True

        summary = _compute_summary(
            self._name, self._run_id, self._cases, self._start_time
        )

        # Set final status on root span
        has_errors = summary.error_count > 0
        run_status = "failure" if has_errors else "success"
        self._root_span.set_attribute("test.suite.run.status", run_status)
        if has_errors:
            self._root_span.set_status(
                trace.StatusCode.ERROR,
                f"{summary.error_count} case(s) failed",
            )

        self._root_span.end()

        print(str(summary))
        logger.info("Experiment completed: %s", self._run_id)

        return summary

    def __enter__(self) -> Experiment:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._closed:
            self.close()


# ---------------------------------------------------------------------------
# evaluate() — full eval runner (Mode A)
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    *,
    task: Callable[[Any], Any],
    data: list[dict[str, Any]] | Iterable[dict[str, Any]],
    scores: list[Callable[..., Score | list[Score] | float]],
    metadata: dict[str, Any] | None = None,
    max_concurrency: int = 1,
    record_io: bool = False,
) -> EvaluateResult:
    """Run a task against a dataset, score outputs, and record results.

    Creates experiment spans where agent execution spans are children
    of each case span, giving full trace waterfall per case.

    Args:
        name: Experiment name (``test.suite.name``).
        task: Callable that takes the input and returns output. Should
            be decorated with ``@observe()`` for full tracing.
        data: Iterable of dicts with ``"input"`` and optionally
            ``"expected"`` and ``"case_id"`` keys.
        scores: List of scorer callables. Each receives
            ``(input, output, expected)`` and returns a ``Score``,
            list of ``Score``, or a plain float.
        metadata: Attached to the root experiment span.
        max_concurrency: Reserved for future async support. Currently
            runs sequentially.
        record_io: Record input/output/expected as span attributes.

    Returns:
        ``EvaluateResult`` with summary and per-case results.
    """
    run_id = _make_run_id()
    start_time = time.monotonic()
    tracer = trace.get_tracer(_TRACER_NAME)

    root_attrs: dict[str, Any] = {
        "test.suite.name": name,
        "test.suite.run.id": run_id,
        "gen_ai.operation.name": "evaluation",
    }
    if metadata:
        for key, val in metadata.items():
            root_attrs[key] = val

    root_span = tracer.start_span(
        f"test_suite_run {name}",
        kind=SpanKind.INTERNAL,
        attributes=root_attrs,
    )
    root_context = trace.set_span_in_context(root_span)

    case_records: list[_CaseRecord] = []
    case_results: list[CaseResult] = []

    for item in data:
        input_val = item.get("input")
        expected_val = item.get("expected")
        case_id = item.get("case_id") or _make_case_id(input_val)
        case_name_val = item.get("case_name")

        # Case span attributes
        case_attrs: dict[str, Any] = {
            "test.suite.name": name,
            "test.suite.run.id": run_id,
            "test.case.id": case_id,
            "gen_ai.operation.name": "evaluation",
        }
        if case_name_val:
            case_attrs["test.case.name"] = case_name_val

        if record_io:
            if input_val is not None:
                case_attrs["test.case.input"] = json.dumps(
                    input_val, default=str
                )[:10_000]
            if expected_val is not None:
                case_attrs["test.case.expected"] = json.dumps(
                    expected_val, default=str
                )[:10_000]

        case_span = tracer.start_span(
            "invoke_agent",
            context=root_context,
            kind=SpanKind.INTERNAL,
            attributes=case_attrs,
        )
        case_context = trace.set_span_in_context(case_span)

        # Run the task inside the case span context
        output_val = None
        error_msg = None
        try:
            token = trace.context_api.attach(case_context)
            try:
                output_val = task(input_val)
            finally:
                trace.context_api.detach(token)
        except Exception as exc:
            error_msg = str(exc)
            case_span.set_status(trace.StatusCode.ERROR, error_msg)
            case_span.record_exception(exc)

        if record_io and output_val is not None:
            case_span.set_attribute(
                "test.case.output",
                json.dumps(output_val, default=str)[:10_000],
            )

        # Run scorers
        case_scores: dict[str, float] = {}
        if error_msg is None:
            for scorer_fn in scores:
                try:
                    result = scorer_fn(input_val, output_val, expected_val)
                    if isinstance(result, Score):
                        case_scores[result.name] = result.value
                    elif isinstance(result, list):
                        for s in result:
                            if isinstance(s, Score):
                                case_scores[s.name] = s.value
                    elif isinstance(result, (int, float)):
                        case_scores[scorer_fn.__name__] = float(result)
                except Exception as exc:
                    logger.warning(
                        "Scorer %s failed for case %s: %s",
                        getattr(scorer_fn, "__name__", scorer_fn),
                        case_id,
                        exc,
                    )

        # Add score events
        _add_score_events(case_span, case_scores)

        # Determine status
        status = "fail" if error_msg else "pass"
        case_span.set_attribute("test.case.result.status", status)
        case_span.end()

        case_records.append(
            _CaseRecord(
                case_id=case_id,
                case_name=case_name_val,
                scores=case_scores,
                error=error_msg,
            )
        )
        case_results.append(
            CaseResult(
                case_id=case_id,
                case_name=case_name_val,
                input=input_val,
                output=output_val,
                expected=expected_val,
                scores=case_scores,
                error=error_msg,
                status=status,
            )
        )

    # Finalize root span
    summary = _compute_summary(name, run_id, case_records, start_time)
    has_errors = summary.error_count > 0
    root_span.set_attribute(
        "test.suite.run.status", "failure" if has_errors else "success"
    )
    if has_errors:
        root_span.set_status(
            trace.StatusCode.ERROR,
            f"{summary.error_count} case(s) failed",
        )
    root_span.end()

    print(str(summary))
    logger.info("evaluate() completed: %s", run_id)

    return EvaluateResult(summary=summary, cases=case_results)
