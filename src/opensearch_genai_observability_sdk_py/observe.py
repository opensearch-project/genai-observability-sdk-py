# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Unified tracing primitive for GenAI observability.

Provides ``observe()`` — a single function that works as both a decorator
and a context manager, replacing the previous @workflow, @task, @agent,
@tool decorators with one spec-aligned primitive.

As a decorator::

    @observe(name="planner", op=Op.INVOKE_AGENT)
    def plan(query: str) -> str:
        enrich(model="gpt-4.1")
        return ...

As a context manager::

    with observe("thinking", op=Op.CHAT) as span:
        enrich(model="gpt-4.1", input_tokens=1500)

The ``op`` parameter maps directly to ``gen_ai.operation.name`` from the
`OpenTelemetry GenAI semantic conventions
<https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.
Use ``Op`` constants for well-known values, or pass any custom string.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar, overload

from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_TRACER_NAME = "opensearch-genai-observability-sdk-py"


class Op:
    """Well-known values for ``gen_ai.operation.name``.

    Use these constants with the ``op`` parameter of ``observe()``
    for IDE autocomplete and typo prevention. Any custom string is
    also accepted when none of these fit.

    Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    CHAT = "chat"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    EXECUTE_TOOL = "execute_tool"
    RETRIEVAL = "retrieval"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    TEXT_COMPLETION = "text_completion"


# Span naming: ops that prefix the name (e.g. "invoke_agent planner")
_PREFIXED_OPS = {
    Op.INVOKE_AGENT,
    Op.CREATE_AGENT,
    Op.EXECUTE_TOOL,
    Op.CHAT,
    Op.RETRIEVAL,
    Op.EMBEDDINGS,
    Op.GENERATE_CONTENT,
    Op.TEXT_COMPLETION,
}

# Name attribute key per op
_NAME_ATTR = {
    Op.EXECUTE_TOOL: "gen_ai.tool.name",
}
_DEFAULT_NAME_ATTR = "gen_ai.agent.name"


def _make_span_name(name: str, op: str | None) -> str:
    """Build the span name from op and entity name."""
    if op and op in _PREFIXED_OPS:
        return f"{op} {name}"
    return name


def _set_span_op_attributes(span: trace.Span, name: str, op: str | None) -> None:
    """Set gen_ai semantic convention attributes based on op."""
    if op is None:
        return
    span.set_attribute("gen_ai.operation.name", op)
    attr_key = _NAME_ATTR.get(op, _DEFAULT_NAME_ATTR)
    span.set_attribute(attr_key, name)
    if op == Op.EXECUTE_TOOL:
        span.set_attribute("gen_ai.tool.type", "function")


class _Observe:
    """Object returned by ``observe()`` that works as both decorator and context manager."""

    def __init__(
        self,
        name: str | None = None,
        *,
        op: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        name_from: str | None = None,
    ) -> None:
        self._name = name
        self._op = op
        self._kind = kind
        self._name_from = name_from
        self._span: trace.Span | None = None
        self._ctx_manager: Any = None

    # -- Context manager usage: with observe("chat", op=Op.CHAT) as span: --

    def __enter__(self) -> trace.Span:
        entity_name = self._name or "unnamed"
        span_name = _make_span_name(entity_name, self._op)
        tracer = trace.get_tracer(_TRACER_NAME)
        self._span = tracer.start_span(span_name, kind=self._kind)
        _set_span_op_attributes(self._span, entity_name, self._op)
        self._ctx_manager = trace.use_span(self._span, end_on_exit=False)
        self._ctx_manager.__enter__()
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span is None:
            return
        try:
            if exc_val is not None:
                self._span.set_status(trace.StatusCode.ERROR, str(exc_val))
                self._span.record_exception(exc_val)
        finally:
            self._span.end()
            if self._ctx_manager is not None:
                self._ctx_manager.__exit__(exc_type, exc_val, exc_tb)

    # -- Decorator usage: @observe(name="planner", op=Op.INVOKE_AGENT) --

    def __call__(self, fn: F) -> F:
        entity_name = self._name or fn.__qualname__
        op = self._op
        kind = self._kind
        name_from = self._name_from
        sig = inspect.signature(fn)

        if inspect.iscoroutinefunction(fn):
            return _wrap_async(fn, entity_name, op, kind, sig, name_from)  # type: ignore[return-value]
        elif inspect.isasyncgenfunction(fn):
            return _wrap_async_gen(fn, entity_name, op, kind, sig, name_from)  # type: ignore[return-value]
        elif inspect.isgeneratorfunction(fn):
            return _wrap_gen(fn, entity_name, op, kind, sig, name_from)  # type: ignore[return-value]
        else:
            return _wrap_sync(fn, entity_name, op, kind, sig, name_from)  # type: ignore[return-value]


def _resolve_name(
    static_name: str,
    name_from: str | None,
    sig: inspect.Signature,
    args: tuple,
    kwargs: dict,
) -> str:
    """Resolve entity name at call time, optionally from a function parameter."""
    if not name_from:
        return static_name
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        runtime_val = bound.arguments.get(name_from)
        if runtime_val is not None:
            return str(runtime_val)
    except TypeError:
        pass
    return static_name


def _set_input(
    span: trace.Span,
    op: str | None,
    sig: inspect.Signature,
    args: tuple,
    kwargs: dict,
) -> None:
    """Capture function input as a span attribute."""
    try:
        if not args and not kwargs:
            return
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            value = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
        except TypeError:
            value = {"args": list(args), "kwargs": kwargs}

        serialized = json.dumps(value, default=str)
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"

        if op == Op.EXECUTE_TOOL:
            attr_key = "gen_ai.tool.call.arguments"
        else:
            attr_key = "gen_ai.input.messages"
        span.set_attribute(attr_key, serialized)
    except Exception:  # noqa: S110
        pass


def _set_output(span: trace.Span, op: str | None, result: Any) -> None:
    """Capture function output as a span attribute."""
    try:
        if result is None:
            return

        attr_key = "gen_ai.tool.call.result" if op == Op.EXECUTE_TOOL else "gen_ai.output.messages"

        existing = getattr(span, "attributes", None)
        if existing and attr_key in existing:
            return

        serialized = json.dumps(result, default=str)
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"

        span.set_attribute(attr_key, serialized)
    except Exception:  # noqa: S110
        pass


def _wrap_sync(
    fn: F,
    name: str,
    op: str | None,
    kind: SpanKind,
    sig: inspect.Signature,
    name_from: str | None = None,
) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        entity_name = _resolve_name(name, name_from, sig, args, kwargs)
        span_name = _make_span_name(entity_name, op)
        tracer = trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            _set_span_op_attributes(span, entity_name, op)
            _set_input(span, op, sig, args, kwargs)
            try:
                result = fn(*args, **kwargs)
                _set_output(span, op, result)
                return result
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return wrapper  # type: ignore[return-value]


def _wrap_async(
    fn: F,
    name: str,
    op: str | None,
    kind: SpanKind,
    sig: inspect.Signature,
    name_from: str | None = None,
) -> F:
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        entity_name = _resolve_name(name, name_from, sig, args, kwargs)
        span_name = _make_span_name(entity_name, op)
        tracer = trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            _set_span_op_attributes(span, entity_name, op)
            _set_input(span, op, sig, args, kwargs)
            try:
                result = await fn(*args, **kwargs)
                _set_output(span, op, result)
                return result
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return wrapper  # type: ignore[return-value]


def _wrap_gen(
    fn: F,
    name: str,
    op: str | None,
    kind: SpanKind,
    sig: inspect.Signature,
    name_from: str | None = None,
) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        entity_name = _resolve_name(name, name_from, sig, args, kwargs)
        span_name = _make_span_name(entity_name, op)
        tracer = trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            _set_span_op_attributes(span, entity_name, op)
            _set_input(span, op, sig, args, kwargs)
            try:
                collected = []
                for item in fn(*args, **kwargs):
                    collected.append(item)
                    yield item
                _set_output(span, op, collected)
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return wrapper  # type: ignore[return-value]


def _wrap_async_gen(
    fn: F,
    name: str,
    op: str | None,
    kind: SpanKind,
    sig: inspect.Signature,
    name_from: str | None = None,
) -> F:
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        entity_name = _resolve_name(name, name_from, sig, args, kwargs)
        span_name = _make_span_name(entity_name, op)
        tracer = trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            _set_span_op_attributes(span, entity_name, op)
            _set_input(span, op, sig, args, kwargs)
            try:
                collected = []
                async for item in fn(*args, **kwargs):
                    collected.append(item)
                    yield item
                _set_output(span, op, collected)
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return wrapper  # type: ignore[return-value]


@overload
def observe(fn: Callable[..., Any], /) -> Callable[..., Any]: ...


@overload
def observe(
    name: str | None = None,
    *,
    op: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    name_from: str | None = None,
) -> _Observe: ...


def observe(
    name: str | Callable[..., Any] | None = None,
    *,
    op: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    name_from: str | None = None,
) -> _Observe | Callable[..., Any]:
    """Create a traced span — as a decorator or context manager.

    As a **decorator**, wraps a function so every call creates an OTel
    span that automatically captures input arguments and return values::

        @observe(name="planner", op=Op.INVOKE_AGENT)
        def plan(query: str) -> str:
            return ...

    As a **context manager**, traces an inline code block::

        with observe("thinking", op=Op.CHAT) as span:
            result = call_llm(prompt)

    Can also be used without parentheses for the simplest case::

        @observe
        def my_function():
            ...

    Dynamic span naming with ``name_from``::

        @observe(op=Op.EXECUTE_TOOL, name_from="tool_name")
        def run_tool(tool_name: str, args: dict) -> dict:
            ...
        run_tool("web_search", {"q": "hi"})  # span: "execute_tool web_search"

    Args:
        name: Span name. For decorators, defaults to the function's
            qualified name. For context managers, defaults to "unnamed".
        op: The ``gen_ai.operation.name`` value. Use ``Op`` constants
            for well-known values (``Op.INVOKE_AGENT``, ``Op.CHAT``,
            ``Op.EXECUTE_TOOL``, etc.) or any custom string.
            When ``None``, no ``gen_ai.operation.name`` or name attributes
            are set on the span. Input/output capture still occurs.
        kind: OTel ``SpanKind``. Defaults to ``INTERNAL``.
            Use ``SpanKind.CLIENT`` when calling external services.
        name_from: Name of a function parameter whose runtime value
            becomes the span name. Useful for dispatcher functions where
            the logical name isn't known until call time. Only applies
            when used as a decorator.

    Returns:
        An ``_Observe`` instance that acts as both decorator and
        context manager.
    """
    # Support @observe without parentheses
    if callable(name):
        fn = name
        obs = _Observe(name=None, op=op, kind=kind, name_from=name_from)
        return obs(fn)

    return _Observe(name=name, op=op, kind=kind, name_from=name_from)
