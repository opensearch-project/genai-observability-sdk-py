# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Benchmark tracking for GenAI evaluation results.

Run offline evaluations, or upload results from any harness (RAGAS, DeepEval,
pytest, custom), as OTel **log events**. Results flow via OTLP logs to
OpenSearch, where Data Prepper routes them to ``otel-v1-genai-evaluation-*``.

Two modes of use:

**Direct upload**: you already have results::

    with Benchmark(name="rag-v2", metadata={"model": "gpt-4.1"}) as b:
        b.log(input="What is Python?", output="A language",
              scores={"accuracy": 1.0})

**Full eval runner**: the SDK runs your task and scorers::

    result = evaluate(
        name="rag-v2",
        task=my_agent,
        data=[{"input": "What is Python?", "expected": "programming language"}],
        scores=[accuracy_scorer],
    )

Telemetry shape::

    trace: the agent run of each case, exactly like production. No harness span.
      root span (first span the task starts)
          test.suite.name, test.suite.run.id, test.case.id, test.case.name
      ... normal GenAI spans

    LogRecord gen_ai.evaluation.result       one per score, parented to the
                                             agent run's root span, test.* copied
    LogRecord opensearch.test_case.result    one per case, parented to the
                                             agent run's root span:
        test.suite.name, test.suite.run.id, test.case.id, test.case.name,
        test.case.result.status (pass | fail | error), error.type,
        opensearch.test_case.input / .expected / .output (record_io),
        opensearch.test_case.duration_ms
    LogRecord opensearch.test_suite_run.result   one per run, no trace context:
        test.suite.name, test.suite.run.id, test.suite.run.status,
        opensearch.test_suite_run.case_count / .pass_count / .fail_count /
        .error_count / .duration_ms

If the task starts no span, the case event and the score events carry no
trace context (the ``test.*`` attributes are still set).

- ``test.*`` follows the proposed test conventions
  (semantic-conventions-genai#79, semantic-conventions#1262).
- ``gen_ai.evaluation.result`` follows the GenAI events spec
  (https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/).
- Test case content lives under ``opensearch.test_case.*``; ``test.*`` is
  owned by the OTel CICD SIG and has no names for it.

Scope: score events use the scorer (evaluator) name when it is a named
function, else the SDK scope; case and run events always use the SDK scope
``opensearch-genai-observability-sdk-py`` with the SDK version. Every scope
carries ``opensearch.genai.sdk.name`` / ``opensearch.genai.sdk.version``
scope attributes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import threading
import time
import uuid
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext

from opensearch_genai_observability_sdk_py._internal import (
    _RESERVED_KEYS,
    AGENT_NAME_KEY,
    AGENT_VERSION_KEY,
    get_event_logger,
    resolve_agent,
    validate_metadata_keys,
)
from opensearch_genai_observability_sdk_py.score import (
    _VALID_DIRECTIONS,
    _VALID_EVALUATOR_TYPES,
    _no_span_context,
    build_evaluation_attributes,
    emit_evaluation_event,
    normalize_assessment,
    resolve_assessment,
    span_context_for,
)

logger = logging.getLogger(__name__)

TEST_CASE_EVENT_NAME = "opensearch.test_case.result"
TEST_SUITE_RUN_EVENT_NAME = "opensearch.test_suite_run.result"

_IO_TRUNCATION_LIMIT = 10_000
_TRUNCATION_MARKER = "...[truncated]"

_CASE_STATUSES = ("pass", "fail", "error")


# ---------------------------------------------------------------------------
# EvalScore dataclass: return type for scorer functions used with evaluate()
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalScore:
    """Result from a scorer function.

    Returned by scorer callables passed to ``evaluate()`` (or passed to
    ``Benchmark.log(scores=[...])``).

    Verdict fields (all optional; unset fields fall back to the scorer's
    ``@scorer(...)`` declaration):

    - ``assessment`` (``"pass"`` / ``"fail"``) or ``passed`` (bool): explicit
      verdict, recorded as ``opensearch.evaluation.assessment``.
      ``assessment`` wins when both are set.
    - ``threshold`` + ``direction``: rule the SDK applies to ``value`` when no
      explicit verdict is given.
    - ``score_min`` / ``score_max``: scale bounds.
    """

    name: str
    value: float | None
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None
    assessment: str | None = None
    passed: bool | None = None
    threshold: float | None = None
    direction: str | None = None
    score_min: float | None = None
    score_max: float | None = None


@dataclasses.dataclass
class _FailedScore:
    """Internal stand-in for a scorer that raised (no score value)."""

    name: str
    value: float | None = None
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None
    assessment: str | None = None
    passed: bool | None = None
    threshold: float | None = None
    direction: str | None = None
    score_min: float | None = None
    score_max: float | None = None


# ---------------------------------------------------------------------------
# @scorer: declare a scorer's evaluation name and pass rule up front
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScorerSpec:
    """What a scorer declares about the evaluation it produces.

    Attached to the function by ``@scorer(...)`` as ``fn.scorer_spec``.
    """

    name: str | None = None
    direction: str | None = None
    threshold: float | None = None
    score_min: float | None = None
    score_max: float | None = None
    evaluator: str | None = None
    evaluator_type: str | None = None


def scorer(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    direction: str | None = None,
    threshold: float | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    evaluator: str | None = None,
    evaluator_type: str | None = None,
) -> Any:
    """Declare a scorer's evaluation name and pass rule for ``evaluate()``.

    Usable bare (``@scorer``) or with arguments::

        @scorer(name="answer_quality", threshold=4, score_min=1, score_max=5,
                evaluator_type="llm_judge")
        def quality(input, output, expected):
            return judge(...)  # float or EvalScore

        @scorer(name="latency_budget", direction="minimize", threshold=12)
        def latency(input, output, expected):
            return seconds

    - ``name``: ``gen_ai.evaluation.name`` of float results and of the error
      grade emitted when the scorer raises (default: the function name).
    - ``threshold`` / ``direction`` (default ``"maximize"`` with a threshold)
      / ``score_min`` / ``score_max``: defaults for every score the scorer
      returns; fields set on a returned ``EvalScore`` take precedence.
    - ``evaluator``: evaluator identity (instrumentation scope and
      ``opensearch.evaluation.evaluator.name``; default: the function name).
    - ``evaluator_type``: ``llm_judge``, ``code``, ``human`` or ``external``.

    Raises:
        ValueError: ``direction`` or ``evaluator_type`` is not a known value.
        TypeError: ``threshold`` / ``score_min`` / ``score_max`` is not a number.
    """
    if direction is not None and direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}; got {direction!r}")
    if evaluator_type is not None and evaluator_type not in _VALID_EVALUATOR_TYPES:
        raise ValueError(
            f"evaluator_type must be one of {_VALID_EVALUATOR_TYPES}; got {evaluator_type!r}"
        )
    for arg, val in (("threshold", threshold), ("score_min", score_min), ("score_max", score_max)):
        if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float))):
            raise TypeError(f"{arg} must be a number; got {val!r}")
    spec = ScorerSpec(
        name=name,
        direction=direction or ("maximize" if threshold is not None else None),
        threshold=threshold,
        score_min=score_min,
        score_max=score_max,
        evaluator=evaluator,
        evaluator_type=evaluator_type,
    )

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper.scorer_spec = spec  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate


def _verdict(s: EvalScore | _FailedScore) -> str | None:
    """Verdict of a resolved score (explicit assessment / passed, else threshold)."""
    explicit = normalize_assessment(s.assessment) if s.assessment is not None else None
    if explicit is None and s.passed is not None:
        explicit = normalize_assessment(bool(s.passed))
    return resolve_assessment(
        s.value, assessment=explicit, threshold=s.threshold, direction=s.direction
    )


def _apply_spec(s: EvalScore, spec: ScorerSpec | None) -> EvalScore:
    """Fill the score's unset rule fields from the scorer declaration."""
    if spec is None:
        return s
    return dataclasses.replace(
        s,
        threshold=s.threshold if s.threshold is not None else spec.threshold,
        direction=s.direction or spec.direction,
        score_min=s.score_min if s.score_min is not None else spec.score_min,
        score_max=s.score_max if s.score_max is not None else spec.score_max,
    )


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
class BenchmarkSummary:
    benchmark_name: str
    run_id: str
    total_cases: int
    error_count: int
    duration_ms: float
    scores: dict[str, ScoreSummary]

    def __str__(self) -> str:
        width = 60
        lines = [
            "=" * width,
            f"  BENCHMARK: {self.benchmark_name}",
            f"  Run: {self.run_id} | Cases: {self.total_cases}"
            f" | Errors: {self.error_count}"
            f" | Duration: {self.duration_ms / 1000:.1f}s",
            "",
        ]
        for s in self.scores.values():
            lines.append(
                f"  {s.name:<20s} avg={s.avg:.2f}  min={s.min:.2f}  max={s.max:.2f}  (n={s.count})"
            )
        lines.append("=" * width)
        return "\n".join(lines)


@dataclasses.dataclass
class TestCaseResult:
    """Per-case result.

    ``status`` keeps its historical values (``"pass"`` / ``"fail"``; a task or
    scorer error is ``"fail"``). The ``opensearch.test_case.result`` event
    reports ``pass`` / ``fail`` / ``error`` and ``error_type`` holds the
    exception class name when the task or a scorer raised. ``trace_id`` /
    ``span_id`` (hex) identify the span the case and score events are
    parented to (the agent run's root span), or are ``None`` when the task
    started no span. ``assessments`` maps evaluation name to the declared
    verdict (``pass`` / ``fail``) for scores that have one; a ``fail``
    verdict makes ``status`` ``"fail"``.
    """

    case_id: str
    case_name: str | None
    input: Any
    output: Any
    expected: Any
    scores: dict[str, float]
    error: str | None
    status: str
    scorer_errors: list[str] | None = None
    error_type: str | None = None
    duration_ms: float | None = None
    trace_id: str | None = None
    span_id: str | None = None
    assessments: dict[str, str] | None = None


@dataclasses.dataclass
class BenchmarkResult:
    summary: BenchmarkSummary
    cases: list[TestCaseResult]


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


def _truncate_io(value: str) -> str:
    """Truncate a serialized IO string, appending a marker if truncated."""
    if len(value) <= _IO_TRUNCATION_LIMIT:
        return value
    return value[: _IO_TRUNCATION_LIMIT - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _io_attr(value: Any) -> str:
    return _truncate_io(json.dumps(value, default=str))


_GROUPING_KEYS = ("test.suite.name", "test.suite.run.id", "test.case.id", "test.case.name")
_AGENT_KEYS = (AGENT_NAME_KEY, AGENT_VERSION_KEY)


def _run_agent(agents: list[dict[str, str]], declared: dict[str, str]) -> dict[str, str]:
    """The run's agent: the declared one, else the one every case agreed on."""
    out = dict(declared)
    for key in _AGENT_KEYS:
        if key in out:
            continue
        values = {a[key] for a in agents if key in a}
        if len(values) == 1 and all(key in a for a in agents):
            out[key] = values.pop()
    return out


def _grouping(suite: str, run_id: str, case_id: str, case_name: str | None) -> dict[str, Any]:
    """The ``test.*`` ids stamped on the agent root span and copied onto events."""
    attrs: dict[str, Any] = {
        "test.suite.name": suite,
        "test.suite.run.id": run_id,
        "test.case.id": case_id,
    }
    if case_name:
        attrs["test.case.name"] = case_name
    return attrs


def _span_ids(ctx: Context) -> tuple[str | None, str | None]:
    sc = trace.get_current_span(ctx).get_span_context()
    if not sc.is_valid:
        return None, None
    return format(sc.trace_id, "032x"), format(sc.span_id, "016x")


# ---------------------------------------------------------------------------
# Agent run scope: stamp test.* on the root span(s) a case's task starts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _CaseScope:
    """State for the test case whose task is currently running."""

    attributes: dict[str, Any]
    root: SpanContext | None = None
    # The root span itself: its final attributes name the agent that ran.
    root_span: Any = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


_current_case: ContextVar[_CaseScope | None] = ContextVar(
    "opensearch_genai_current_test_case", default=None
)


class _CaseRootStamper(SpanProcessor):
    """Stamps ``test.*`` ids on root spans started while a test case runs.

    A root span here is a span without a local parent (no parent, or a remote
    one). The first such span is captured as the agent run's root: the case
    and score events are parented to it. Spans outside ``evaluate()`` are not
    touched.
    """

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        scope = _current_case.get()
        if scope is None:
            return
        parent = span.parent
        if parent is not None and parent.is_valid and not parent.is_remote:
            return
        existing = span.attributes or {}
        for key, val in scope.attributes.items():
            # The agent's own identity on its span wins over the declared one.
            if key in _AGENT_KEYS and key in existing:
                continue
            span.set_attribute(key, val)
        with scope.lock:
            if scope.root is None:
                scope.root = span.get_span_context()
                scope.root_span = span

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_stamped_providers: weakref.WeakSet[Any] = weakref.WeakSet()
_stamper_lock = threading.Lock()


def _ensure_root_stamper() -> None:
    """Install ``_CaseRootStamper`` once on the global SDK TracerProvider."""
    provider = trace.get_tracer_provider()
    add = getattr(provider, "add_span_processor", None)
    if add is None:
        logger.debug(
            "evaluate(): global TracerProvider %r has no add_span_processor; "
            "agent root spans will not be stamped with test.* attributes",
            provider,
        )
        return
    with _stamper_lock:
        if provider in _stamped_providers:
            return
        add(_CaseRootStamper())
        _stamped_providers.add(provider)


@contextlib.contextmanager
def _agent_run_scope(grouping: dict[str, Any]) -> Iterator[_CaseScope]:
    """Run a case's task as its own trace and capture that trace's root span.

    The active span is cleared (so the agent run starts a new trace even if
    ``evaluate()`` is called inside a span); other context values such as
    baggage are kept.
    """
    scope = _CaseScope(attributes=dict(grouping))
    var_token = _current_case.set(scope)
    ctx_token = trace.context_api.attach(trace.set_span_in_context(trace.INVALID_SPAN))
    try:
        yield scope
    finally:
        trace.context_api.detach(ctx_token)
        _current_case.reset(var_token)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _emit_sdk_event(event_name: str, attributes: dict[str, Any], context: Context) -> None:
    """Emit one LogRecord event with the SDK instrumentation scope."""
    otel_logger = get_event_logger()
    otel_logger.emit(
        event_name=event_name,
        timestamp=time.time_ns(),
        severity_number=SeverityNumber.INFO,
        attributes=attributes,
        context=context,
    )


def _emit_case_event(
    grouping: dict[str, Any],
    *,
    status: str,
    context: Context,
    error_type: str | None = None,
    io: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit the ``opensearch.test_case.result`` event for one case."""
    attrs: dict[str, Any] = dict(metadata or {})
    attrs.update(grouping)
    attrs["test.case.result.status"] = status
    if error_type:
        attrs["error.type"] = error_type
    if io:
        attrs.update(io)
    if duration_ms is not None:
        attrs["opensearch.test_case.duration_ms"] = float(duration_ms)
    _emit_sdk_event(TEST_CASE_EVENT_NAME, attrs, context)


def _emit_run_event(
    suite: str,
    run_id: str,
    statuses: list[str],
    *,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
    agent: dict[str, str] | None = None,
) -> str:
    """Emit the ``opensearch.test_suite_run.result`` event; return the run status."""
    error_count = statuses.count("error")
    run_status = "failure" if error_count else "success"
    attrs: dict[str, Any] = dict(metadata or {})
    attrs.update(
        {
            "test.suite.name": suite,
            "test.suite.run.id": run_id,
            "test.suite.run.status": run_status,
            "opensearch.test_suite_run.case_count": len(statuses),
            "opensearch.test_suite_run.pass_count": statuses.count("pass"),
            "opensearch.test_suite_run.fail_count": statuses.count("fail"),
            "opensearch.test_suite_run.error_count": error_count,
            "opensearch.test_suite_run.duration_ms": float(duration_ms),
        }
    )
    if agent:
        attrs.update(agent)
    _emit_sdk_event(TEST_SUITE_RUN_EVENT_NAME, attrs, _no_span_context())
    return run_status


def _emit_scores(
    grouping: dict[str, Any],
    scores: Iterable[EvalScore | _FailedScore],
    *,
    context: Context,
    evaluator: str | None = None,
    evaluator_type: str | None = None,
    error_type: str | None = None,
) -> None:
    """Emit one ``gen_ai.evaluation.result`` event per score.

    Each event is parented to ``context`` (the evaluated span, or no span)
    and carries the ``test.*`` grouping attributes and the verdict.
    """
    has_span = trace.get_current_span(context).get_span_context().is_valid
    for s in scores:
        extra = dict(grouping)
        if s.metadata:
            extra.update(_safe_metadata(s.metadata, context=f"EvalScore({s.name!r})"))
        attrs = build_evaluation_attributes(
            name=s.name,
            value=s.value,
            label=s.label,
            explanation=s.explanation,
            level="span" if has_span else None,
            evaluator=evaluator,
            evaluator_type=evaluator_type,
            direction=s.direction,
            score_min=s.score_min,
            score_max=s.score_max,
            threshold=s.threshold,
            assessment=_verdict(s),
            error_type=error_type,
            extra=extra,
        )
        emit_evaluation_event(attrs, context=context, evaluator=evaluator)


def _assessments(scores: Iterable[EvalScore | _FailedScore]) -> dict[str, str]:
    """Evaluation name -> verdict, for the scores that declare one."""
    out: dict[str, str] = {}
    for s in scores:
        verdict = _verdict(s)
        if verdict is not None:
            out[s.name] = verdict
    return out


def _safe_metadata(metadata: dict[str, Any], *, context: str = "") -> dict[str, Any]:
    """Filter out SDK-reserved keys from user metadata and warn."""
    validate_metadata_keys(metadata, context=context)
    return {k: v for k, v in metadata.items() if k not in _RESERVED_KEYS}


def _compute_summary(
    benchmark_name: str,
    run_id: str,
    cases: list[TestCaseResult],
    start_time: float,
) -> BenchmarkSummary:
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

    return BenchmarkSummary(
        benchmark_name=benchmark_name,
        run_id=run_id,
        total_cases=len(cases),
        error_count=error_count,
        duration_ms=(time.monotonic() - start_time) * 1000,
        scores=score_summaries,
    )


# ---------------------------------------------------------------------------
# Benchmark class: direct upload (Mode B)
# ---------------------------------------------------------------------------


class Benchmark:
    """Upload evaluation results as OTel log events.

    Each ``log()`` call emits one ``opensearch.test_case.result`` event and
    one ``gen_ai.evaluation.result`` event per score; ``close()`` emits one
    ``opensearch.test_suite_run.result`` event. When ``trace_id`` /
    ``span_id`` of the agent run are given, the case and score events are
    parented to that span; otherwise they carry no trace context. No spans
    are created.

    Use as a context manager::

        with Benchmark(name="rag-v2") as b:
            b.log(input="q1", output="a1", scores={"accuracy": 1.0})

    Or manually::

        b = Benchmark(name="rag-v2")
        b.log(input="q1", output="a1", scores={"accuracy": 1.0})
        summary = b.close()

    Args:
        name: Benchmark name (``test.suite.name``). Stable across runs.
        metadata: Arbitrary run metadata, added to the case and run events.
            Keys reserved
            by the SDK (``test.suite.name`` etc.) are filtered out with a
            warning.
        record_io: If ``True``, record input/output/expected as
            ``opensearch.test_case.input`` / ``.output`` / ``.expected`` on
            the case event. Default ``False`` (PII-safe).
        agent_name: ``gen_ai.agent.name`` of the agent under test, stamped on
            the evaluation, case and run events. Defaults to
            ``register(agent_name=)``.
        agent_version: ``agent.version`` of the agent under test, stamped the
            same way. Defaults to ``register(agent_version=)``.
    """

    def __init__(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        record_io: bool = False,
        agent_name: str | None = None,
        agent_version: str | None = None,
    ) -> None:
        self._name = name
        # Declared agent, else the register() default (never invented).
        self._agent = resolve_agent(agent_name, agent_version)
        self._metadata = _safe_metadata(metadata or {}, context=f"Benchmark({name!r})")
        self._record_io = record_io
        self._run_id = _make_run_id()
        self._start_time = time.monotonic()
        self._cases: list[TestCaseResult] = []
        self._statuses: list[str] = []
        self._closed = False

        logger.info("Benchmark started: name=%s run_id=%s", name, self._run_id)

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
        scores: dict[str, float] | list[EvalScore] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        case_id: str | None = None,
        case_name: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        status: str | None = None,
        error_type: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log a single test case result.

        Emits one ``opensearch.test_case.result`` event and one
        ``gen_ai.evaluation.result`` event per score, with the ``test.*``
        ids set.

        Args:
            input: Test case input data.
            output: Actual output from the agent/system.
            expected: Expected/ground-truth output.
            scores: Pre-computed scores: ``{name: value}``, or a list of
                ``EvalScore`` to declare verdicts (``assessment`` / ``passed``
                or ``threshold`` + ``direction``) and scale bounds.
            metadata: Arbitrary case-level metadata. Keys reserved by the
                SDK are filtered out with a warning.
            error: Error message. The case event status becomes ``"error"``
                (``TestCaseResult.status`` stays ``"fail"`` as before).
            case_id: Explicit ``test.case.id``. If omitted, derived
                from a hash of ``input``.
            case_name: Human-readable ``test.case.name``.
            trace_id: Hex trace ID of the agent run that produced ``output``.
            span_id: Hex span ID of the agent run's root span. Required when
                ``trace_id`` is provided. The case and score events are
                parented to it.
            status: Explicit case status, ``"pass"``, ``"fail"`` or
                ``"error"``. Defaults to ``"error"`` when ``error`` or
                ``error_type`` is given, else ``"fail"`` when any score has a
                ``fail`` verdict, else ``"pass"``.
            error_type: ``error.type`` for an errored case (exception class
                name or other low-cardinality error identifier).
            duration_ms: Case duration, recorded as
                ``opensearch.test_case.duration_ms``.
        """
        if self._closed:
            raise RuntimeError("Benchmark is already closed")
        if bool(trace_id) != bool(span_id):
            raise ValueError("Both trace_id and span_id must be provided together, or neither.")
        if status is not None and status not in _CASE_STATUSES:
            raise ValueError(f"status must be one of {', '.join(_CASE_STATUSES)}; got {status!r}")

        if isinstance(scores, dict):
            eval_scores = [EvalScore(name=n, value=v) for n, v in scores.items()]
        else:
            eval_scores = [s for s in (scores or []) if isinstance(s, EvalScore)]
        score_values = {s.name: s.value for s in eval_scores if s.value is not None}
        assessments = _assessments(eval_scores)
        resolved_case_id = case_id or _make_case_id(input)

        verdict_failed = "fail" in assessments.values()
        event_status = status or (
            "error" if error or error_type else ("fail" if verdict_failed else "pass")
        )
        result_status = status or ("fail" if error or verdict_failed else "pass")

        grouping = {
            **_grouping(self._name, self._run_id, resolved_case_id, case_name),
            **self._agent,
        }

        safe_case_meta = (
            _safe_metadata(metadata, context=f"Benchmark.log(case_id={resolved_case_id!r})")
            if metadata
            else {}
        )
        case_meta = {**self._metadata, **safe_case_meta}

        io: dict[str, Any] = {}
        if self._record_io:
            if input is not None:
                io["opensearch.test_case.input"] = _io_attr(input)
            if output is not None:
                io["opensearch.test_case.output"] = _io_attr(output)
            if expected is not None:
                io["opensearch.test_case.expected"] = _io_attr(expected)

        event_ctx: Context = _no_span_context()
        if trace_id and span_id:
            evaluated_ctx = span_context_for(trace_id, span_id)
            if evaluated_ctx is not None:
                event_ctx = evaluated_ctx
            else:
                logger.warning(
                    "Benchmark.log(): invalid trace_id/span_id %r/%r; "
                    "emitting events without trace context",
                    trace_id,
                    span_id,
                )

        _emit_scores(grouping, eval_scores, context=event_ctx)
        _emit_case_event(
            grouping,
            status=event_status,
            context=event_ctx,
            error_type=error_type,
            io=io,
            duration_ms=duration_ms,
            metadata=case_meta,
        )

        ev_trace_id, ev_span_id = _span_ids(event_ctx)
        self._statuses.append(event_status)
        self._cases.append(
            TestCaseResult(
                case_id=resolved_case_id,
                case_name=case_name,
                input=input,
                output=output,
                expected=expected,
                scores=score_values,
                error=error,
                status=result_status,
                error_type=error_type,
                duration_ms=duration_ms,
                trace_id=ev_trace_id,
                span_id=ev_span_id,
                assessments=assessments or None,
            )
        )

    def close(self) -> BenchmarkSummary:
        """Finalize the benchmark and return a summary.

        Emits the ``opensearch.test_suite_run.result`` event, computes aggregate
        statistics, and prints a formatted summary.

        Returns:
            ``BenchmarkSummary`` with aggregate score statistics.
        """
        if self._closed:
            raise RuntimeError("Benchmark is already closed")

        self._closed = True

        summary = _compute_summary(self._name, self._run_id, self._cases, self._start_time)

        _emit_run_event(
            self._name,
            self._run_id,
            self._statuses,
            duration_ms=summary.duration_ms,
            metadata=self._metadata,
            agent=self._agent,
        )

        print(str(summary))
        logger.info("Benchmark completed: %s", self._run_id)

        return summary

    def __enter__(self) -> Benchmark:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._closed:
            self.close()


# ---------------------------------------------------------------------------
# evaluate(): full eval runner (Mode A)
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    *,
    task: Callable[[Any], Any],
    data: list[dict[str, Any]] | Iterable[dict[str, Any]],
    scores: list[Callable[..., EvalScore | list[EvalScore] | float]],
    metadata: dict[str, Any] | None = None,
    record_io: bool = False,
    agent_name: str | None = None,
    agent_version: str | None = None,
) -> BenchmarkResult:
    """Run a task against a dataset, score outputs, and record results.

    Each case's task call is its own trace, exactly like production: the
    SDK adds no span around it. The root span(s) the task starts get
    ``test.suite.name``, ``test.suite.run.id``, ``test.case.id`` and
    ``test.case.name`` (stamped by a span processor installed on the global
    SDK ``TracerProvider``). Per case the SDK then emits one
    ``gen_ai.evaluation.result`` event per score and one
    ``opensearch.test_case.result`` event, both parented to that root span
    (or without trace context when the task started no span). At the end it
    emits one ``opensearch.test_suite_run.result`` event.

    A task that raises does not stop the run: its case event has status
    ``error`` and ``error.type`` = the exception class, and its scorers are
    skipped. A scorer that raises emits an evaluation event with
    ``error.type``, named after the scorer's declared ``@scorer(name=...)``
    (else the function name), and marks the case ``error``.

    Case status (``test.case.result.status``): ``error`` if the task or a
    scorer raised; else ``fail`` if any score has a ``fail`` verdict
    (explicit ``assessment`` / ``passed``, or ``threshold`` + ``direction``
    from the score or its ``@scorer`` declaration); else ``pass``. Scores
    without a declared rule never fail a case.

    Args:
        name: Benchmark name (``test.suite.name``).
        task: Callable that takes the input and returns output. Should
            be decorated with ``@observe()`` (or otherwise instrumented) so
            each case produces a trace.
        data: Iterable of dicts with ``"input"`` and optionally
            ``"expected"``, ``"case_id"`` and ``"case_name"`` keys. Consumed
            once (single-pass); generators are materialized into a list.
        scores: List of scorer callables. Each receives
            ``(input, output, expected)`` and returns an ``EvalScore``,
            list of ``EvalScore``, a plain float, or a bool (a verdict:
            value 1.0/0.0 with ``pass``/``fail``). Decorate with
            ``@scorer(...)`` to declare the evaluation name and pass rule.
            Scorers run outside the agent trace, so a traced LLM judge gets
            its own trace.
        metadata: Run metadata added to the case and run events. Keys
            reserved by the SDK are filtered out with a warning.
        record_io: Record input/output/expected as
            ``opensearch.test_case.input`` / ``.output`` / ``.expected`` on
            the case event.
        agent_name: ``gen_ai.agent.name`` of the agent under test. Stamped on
            the agent root spans (unless the span names its own agent) and on
            every evaluation, case and run event. When omitted, each case uses
            the agent its root span names, else ``register(agent_name=)``.
        agent_version: ``agent.version`` of the agent under test, resolved the
            same way. Lets runs be compared version to version.

    Returns:
        ``BenchmarkResult`` with summary and per-case results.
    """
    run_id = _make_run_id()
    start_time = time.monotonic()

    safe_meta = _safe_metadata(metadata or {}, context=f"evaluate({name!r})")
    _ensure_root_stamper()

    case_results: list[TestCaseResult] = []
    statuses: list[str] = []
    declared_agent = resolve_agent(agent_name, agent_version)
    case_agents: list[dict[str, str]] = []

    # Materialize iterable upfront so generators work
    data_list = list(data)

    for item in data_list:
        case_start = time.monotonic()
        input_val = item.get("input")
        expected_val = item.get("expected")
        case_id = item.get("case_id") or _make_case_id(input_val)
        case_name_val = item.get("case_name")

        grouping = _grouping(name, run_id, case_id, case_name_val)
        # Root spans get the declared agent (where they name none themselves).
        span_stamp = {**grouping, **declared_agent}

        io: dict[str, Any] = {}
        if record_io:
            if input_val is not None:
                io["opensearch.test_case.input"] = _io_attr(input_val)
            if expected_val is not None:
                io["opensearch.test_case.expected"] = _io_attr(expected_val)

        # Run the task as its own trace and capture its root span.
        output_val = None
        error_msg: str | None = None
        error_type: str | None = None
        scope: _CaseScope | None = None
        try:
            with _agent_run_scope(span_stamp) as scope:
                output_val = task(input_val)
        except Exception as exc:
            error_msg = str(exc)
            error_type = type(exc).__qualname__
            logger.warning("Task failed for case %s: %s", case_id, exc)

        if record_io and output_val is not None:
            io["opensearch.test_case.output"] = _io_attr(output_val)

        # The agent that ran: declared, else named on its root span, else the
        # register() default.
        case_agent = resolve_agent(
            agent_name, agent_version, span=scope.root_span if scope is not None else None
        )
        case_agents.append(case_agent)
        grouping = {**grouping, **case_agent}

        # Case and score events are parented to the agent run's root span.
        event_ctx: Context
        if scope is not None and scope.root is not None:
            event_ctx = trace.set_span_in_context(NonRecordingSpan(scope.root))
        else:
            event_ctx = _no_span_context()

        # Run scorers; each scorer's results become gen_ai.evaluation.result
        # events (evaluator = scorer function name).
        case_scores: dict[str, float] = {}
        case_assessments: dict[str, str] = {}
        scorer_errors: list[str] = []
        if error_msg is None:
            for scorer_fn in scores:
                spec: ScorerSpec | None = getattr(scorer_fn, "scorer_spec", None)
                scorer_name = getattr(scorer_fn, "__name__", str(scorer_fn))
                # Evaluation name of float results and of the error grade.
                eval_name = (spec.name if spec else None) or scorer_name
                # Named scorers identify the evaluator; lambdas etc. do not.
                evaluator = (spec.evaluator if spec else None) or (
                    scorer_name if scorer_name.isidentifier() else None
                )
                evaluator_type = spec.evaluator_type if spec else None
                try:
                    result = scorer_fn(input_val, output_val, expected_val)
                except Exception as exc:
                    msg = f"Scorer {scorer_name} failed: {exc}"
                    scorer_errors.append(msg)
                    error_type = error_type or type(exc).__qualname__
                    logger.warning(
                        "Scorer %s failed for case %s: %s",
                        scorer_name,
                        case_id,
                        exc,
                    )
                    # Record the evaluator failure (error.type, no score value),
                    # under the evaluation name the scorer would have produced.
                    failed = _FailedScore(name=eval_name, explanation=str(exc))
                    if spec is not None:
                        failed.direction = spec.direction
                        failed.threshold = spec.threshold
                        failed.score_min = spec.score_min
                        failed.score_max = spec.score_max
                    _emit_scores(
                        grouping,
                        [failed],
                        context=event_ctx,
                        evaluator=evaluator,
                        evaluator_type=evaluator_type,
                        error_type=type(exc).__qualname__,
                    )
                    continue

                produced: list[EvalScore] = []
                if isinstance(result, EvalScore):
                    produced.append(result)
                elif isinstance(result, list):
                    produced.extend(s for s in result if isinstance(s, EvalScore))
                elif isinstance(result, bool):
                    produced.append(EvalScore(name=eval_name, value=float(result), passed=result))
                elif isinstance(result, (int, float)):
                    produced.append(EvalScore(name=eval_name, value=float(result)))
                produced = [_apply_spec(s, spec) for s in produced]
                for s in produced:
                    if s.value is not None:
                        case_scores[s.name] = s.value
                case_assessments.update(_assessments(produced))
                _emit_scores(
                    grouping,
                    produced,
                    context=event_ctx,
                    evaluator=evaluator,
                    evaluator_type=evaluator_type,
                )

        # Case status: error on any task or scorer error (TestCaseResult.status
        # "fail"), else fail when any score has a fail verdict (explicit
        # assessment or threshold), else pass. No declared verdicts: pass.
        errored = error_msg is not None or bool(scorer_errors)
        verdict_failed = "fail" in case_assessments.values()
        result_status = "fail" if errored or verdict_failed else "pass"
        event_status = "error" if errored else ("fail" if verdict_failed else "pass")
        duration_ms = (time.monotonic() - case_start) * 1000

        _emit_case_event(
            grouping,
            status=event_status,
            context=event_ctx,
            error_type=error_type,
            io=io,
            duration_ms=duration_ms,
            metadata=safe_meta,
        )

        ev_trace_id, ev_span_id = _span_ids(event_ctx)
        statuses.append(event_status)
        case_results.append(
            TestCaseResult(
                case_id=case_id,
                case_name=case_name_val,
                input=input_val,
                output=output_val,
                expected=expected_val,
                scores=case_scores,
                error=error_msg,
                status=result_status,
                scorer_errors=scorer_errors or None,
                error_type=error_type,
                duration_ms=duration_ms,
                trace_id=ev_trace_id,
                span_id=ev_span_id,
                assessments=case_assessments or None,
            )
        )

    summary = _compute_summary(name, run_id, case_results, start_time)
    _emit_run_event(
        name,
        run_id,
        statuses,
        duration_ms=summary.duration_ms,
        metadata=safe_meta,
        # Explicit, else the agent every case agreed on, else register() defaults.
        agent={
            **resolve_agent(),
            **_run_agent(
                case_agents,
                {
                    k: v
                    for k, v in ((AGENT_NAME_KEY, agent_name), (AGENT_VERSION_KEY, agent_version))
                    if v
                },
            ),
        },
    )

    print(str(summary))
    logger.info("evaluate() completed: %s", run_id)

    return BenchmarkResult(summary=summary, cases=case_results)
