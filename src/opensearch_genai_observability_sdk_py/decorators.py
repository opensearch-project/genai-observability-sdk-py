# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Decorators for tracing custom functions as OTEL spans.

Provides @workflow, @task, @agent, and @tool decorators that create
standard OpenTelemetry spans. These are the user-facing API for
tracing custom application logic — the gap that pure auto-instrumentors
don't cover.

All decorators produce standard OTEL spans with gen_ai semantic
convention attributes. Zero lock-in: remove the decorator and
your code still works.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Internal decorator type identifiers (used for attribute routing and span naming)
SPAN_KIND_WORKFLOW = "workflow"
SPAN_KIND_TASK = "task"
SPAN_KIND_AGENT = "invoke_agent"
SPAN_KIND_TOOL = "execute_tool"

# gen_ai.operation.name values per OTEL GenAI semantic conventions
# workflow and task both map to invoke_agent (no workflow/task values in semconv)
_OPERATION_NAME = {
    SPAN_KIND_WORKFLOW: "invoke_agent",
    SPAN_KIND_TASK: "invoke_agent",
    SPAN_KIND_AGENT: "invoke_agent",
    SPAN_KIND_TOOL: "execute_tool",
}

# Default OTel SpanKind per decorator type.
# agent uses CLIENT because it typically represents a call out to an LLM/agent service.
# tool uses INTERNAL because tool execution happens within the process.
_DEFAULT_OTEL_KIND = {
    SPAN_KIND_WORKFLOW: SpanKind.INTERNAL,
    SPAN_KIND_TASK: SpanKind.INTERNAL,
    SPAN_KIND_AGENT: SpanKind.CLIENT,
    SPAN_KIND_TOOL: SpanKind.INTERNAL,
}

_TRACER_NAME = "opensearch-genai-observability-sdk-py"


def workflow(
    name: str | None = None,
    version: int | None = None,
    kind: SpanKind | None = None,
    name_from: str | None = None,
) -> Callable[[F], F]:
    """Trace a function as a workflow span.

    Use for top-level orchestration functions that coordinate
    multiple tasks, agents, or tool calls.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.
        kind: Override the OTel SpanKind. Defaults to INTERNAL.
        name_from: Name of a function parameter whose runtime value is used
            as the entity name. Useful for dispatcher functions where the
            logical name isn't known until call time.

    Example:
        @workflow(name="qa_pipeline")
        def run_pipeline(query: str) -> str:
            plan = plan_steps(query)
            result = execute(plan)
            return result
    """
    return _make_decorator(
        name=name,
        version=version,
        span_kind=SPAN_KIND_WORKFLOW,
        otel_kind=kind,
        name_from=name_from,
    )


def task(
    name: str | None = None,
    version: int | None = None,
    kind: SpanKind | None = None,
    name_from: str | None = None,
) -> Callable[[F], F]:
    """Trace a function as a task span.

    Use for individual units of work within a workflow.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.
        kind: Override the OTel SpanKind. Defaults to INTERNAL.
        name_from: Name of a function parameter whose runtime value is used
            as the entity name.

    Example:
        @task(name="summarize")
        def summarize_text(text: str) -> str:
            return llm.generate(f"Summarize: {text}")
    """
    return _make_decorator(
        name=name, version=version, span_kind=SPAN_KIND_TASK, otel_kind=kind, name_from=name_from
    )


def agent(
    name: str | None = None,
    version: int | None = None,
    kind: SpanKind | None = None,
    name_from: str | None = None,
) -> Callable[[F], F]:
    """Trace a function as an agent span (SpanKind.CLIENT by default).

    Use for autonomous agent logic that makes decisions and invokes tools.
    Defaults to SpanKind.CLIENT because agent invocations typically represent
    a call out to an external LLM or agent service.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.
        kind: Override the OTel SpanKind. Defaults to CLIENT.
        name_from: Name of a function parameter whose runtime value is used
            as the entity name.

    Example:
        @agent(name="research_agent")
        def research(query: str) -> str:
            while not done:
                action = decide_next_action(query)
                result = execute_action(action)
            return result
    """
    return _make_decorator(
        name=name, version=version, span_kind=SPAN_KIND_AGENT, otel_kind=kind, name_from=name_from
    )


def tool(
    name: str | None = None,
    version: int | None = None,
    kind: SpanKind | None = None,
    name_from: str | None = None,
) -> Callable[[F], F]:
    """Trace a function as a tool span.

    Use for tool/function calls invoked by agents.

    Args:
        name: Span name. Defaults to the function's qualified name.
        version: Optional version number for tracking changes.
        kind: Override the OTel SpanKind. Defaults to INTERNAL.
        name_from: Name of a function parameter whose runtime value is used
            as the entity name and span name. Useful for dispatcher methods
            where the actual tool name is a runtime argument.

    Example — static tool:
        @tool(name="web_search")
        def search(query: str) -> list[dict]:
            return search_api.query(query)

    Example — dispatcher with dynamic tool name:
        @tool(name_from="tool_name")
        def execute_tool(self, tool_name: str, arguments: dict) -> dict:
            ...
    """
    return _make_decorator(
        name=name, version=version, span_kind=SPAN_KIND_TOOL, otel_kind=kind, name_from=name_from
    )


def _make_decorator(
    name: str | None,
    version: int | None,
    span_kind: str,
    otel_kind: SpanKind | None,
    name_from: str | None,
) -> Callable[[F], F]:
    """Create a decorator that wraps a function in an OTEL span."""

    resolved_otel_kind = otel_kind if otel_kind is not None else _DEFAULT_OTEL_KIND[span_kind]

    def decorator(fn: F) -> F:
        static_entity_name = name or fn.__qualname__
        fn_doc = fn.__doc__
        sig = inspect.signature(fn)
        # NOTE: tracer is intentionally fetched inside each wrapper (at call
        # time), NOT here at decoration time.  The OTEL ProxyTracer caches
        # the backing tracer on first use; fetching it here would lock in
        # whatever provider is active at import time (often the no-op default)
        # before register() has been called.  Fetching at call time ensures
        # the wrapper always uses the provider that is current when the
        # function is actually invoked.

        def _resolve_names(args: Any, kwargs: Any) -> tuple[str, str]:
            """Resolve entity name and span name at call time."""
            entity = static_entity_name
            if name_from:
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    runtime_val = bound.arguments.get(name_from)
                    if runtime_val is not None:
                        entity = str(runtime_val)
                except TypeError:
                    pass
            if span_kind in (SPAN_KIND_AGENT, SPAN_KIND_TOOL):
                span_name = f"{span_kind} {entity}"
            else:
                span_name = entity
            return entity, span_name

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                entity_name, span_name = _resolve_names(args, kwargs)
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span(span_name, kind=resolved_otel_kind) as span:
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
                    try:
                        result = await fn(*args, **kwargs)
                        _set_output(span, span_kind, result)
                        return result
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return async_wrapper  # type: ignore[return-value]

        elif inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                entity_name, span_name = _resolve_names(args, kwargs)
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span(span_name, kind=resolved_otel_kind) as span:
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
                    try:
                        collected = []
                        for item in fn(*args, **kwargs):
                            collected.append(item)
                            yield item
                        _set_output(span, span_kind, collected)
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return gen_wrapper  # type: ignore[return-value]

        elif inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                entity_name, span_name = _resolve_names(args, kwargs)
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span(span_name, kind=resolved_otel_kind) as span:
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
                    try:
                        collected = []
                        async for item in fn(*args, **kwargs):
                            collected.append(item)
                            yield item
                        _set_output(span, span_kind, collected)
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return async_gen_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                entity_name, span_name = _resolve_names(args, kwargs)
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span(span_name, kind=resolved_otel_kind) as span:
                    _set_span_attributes(
                        span, span_kind, entity_name, version, sig, args, kwargs, fn_doc
                    )
                    try:
                        result = fn(*args, **kwargs)
                        _set_output(span, span_kind, result)
                        return result
                    except Exception as exc:
                        span.set_status(trace.StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _set_span_attributes(
    span: trace.Span,
    span_kind: str,
    entity_name: str,
    version: int | None,
    sig: inspect.Signature,
    args: tuple,
    kwargs: dict,
    fn_doc: str | None = None,
) -> None:
    """Set standard attributes on a span."""
    span.set_attribute("gen_ai.operation.name", _OPERATION_NAME[span_kind])

    # Use type-specific name attributes matching gen_ai semantic conventions
    # workflow and task use gen_ai.agent.name (no workflow/task name attrs in semconv)
    _name_attr = {
        SPAN_KIND_WORKFLOW: "gen_ai.agent.name",
        SPAN_KIND_TASK: "gen_ai.agent.name",
        SPAN_KIND_AGENT: "gen_ai.agent.name",
        SPAN_KIND_TOOL: "gen_ai.tool.name",
    }
    span.set_attribute(_name_attr[span_kind], entity_name)

    if version is not None:
        span.set_attribute("gen_ai.agent.version", str(version))

    # Tool-specific attributes from semconv
    if span_kind == SPAN_KIND_TOOL:
        span.set_attribute("gen_ai.tool.type", "function")
        if fn_doc:
            # Use first non-empty line of docstring
            first_line = next(
                (ln.strip() for ln in fn_doc.splitlines() if ln.strip()), fn_doc[:200]
            )
            span.set_attribute("gen_ai.tool.description", first_line)

    # Capture input (best-effort, don't fail if serialization fails)
    _set_input(span, span_kind, sig, args, kwargs)


def _set_input(
    span: trace.Span, span_kind: str, sig: inspect.Signature, args: tuple, kwargs: dict
) -> None:
    """Attempt to capture function input as a span attribute.

    Binds positional and keyword arguments to their parameter names
    so the trace shows {"city": "Paris"} instead of just "Paris".
    Skips 'self' and 'cls' parameters for class methods.
    """
    try:
        if not args and not kwargs:
            return

        # Bind args to parameter names for readable output
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # Skip 'self' and 'cls' — not useful in traces and not serializable
            value = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
        except TypeError:
            # Fallback if binding fails (e.g., *args/**kwargs signatures)
            value = {"args": list(args), "kwargs": kwargs}

        serialized = json.dumps(value, default=str)
        # Truncate to avoid oversized attributes
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"

        # Tool spans use semconv attribute name; all others use gen_ai.input.messages
        attr_key = (
            "gen_ai.tool.call.arguments" if span_kind == SPAN_KIND_TOOL else "gen_ai.input.messages"
        )
        span.set_attribute(attr_key, serialized)
    except Exception:  # noqa: S110
        pass


def _set_output(span: trace.Span, span_kind: str, result: Any) -> None:
    """Attempt to capture function output as a span attribute.

    Skips setting the attribute if the user already set it inside the function
    body (via trace.get_current_span().set_attribute(...)), so that custom
    formatting (e.g. genai role/parts schema) is not overwritten.
    """
    try:
        if result is None:
            return

        attr_key = (
            "gen_ai.tool.call.result" if span_kind == SPAN_KIND_TOOL else "gen_ai.output.messages"
        )

        # Don't overwrite a value the user already set inside the function body.
        # span.attributes is the public ReadableSpan property, accessible on a live
        # recording span. getattr is used because trace.Span (the interface) does not
        # declare attributes — only the SDK implementation does.
        existing = getattr(span, "attributes", None)
        if existing and attr_key in existing:
            return

        serialized = json.dumps(result, default=str)
        if len(serialized) > 10_000:
            serialized = serialized[:10_000] + "...(truncated)"

        span.set_attribute(attr_key, serialized)
    except Exception:  # noqa: S110
        pass
