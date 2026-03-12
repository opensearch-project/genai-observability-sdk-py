# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for observe() — decorator, context manager, and Op constants."""

import json

import pytest
from opentelemetry.trace import SpanKind, StatusCode

from opensearch_genai_observability_sdk_py.observe import Op, observe

# ---------------------------------------------------------------------------
# Decorator — sync
# ---------------------------------------------------------------------------


@observe(name="my_agent", op=Op.INVOKE_AGENT)
def sync_invoke_agent(query: str) -> str:
    return f"answer to {query}"


@observe(name="my_tool", op=Op.EXECUTE_TOOL)
def sync_execute_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@observe(name="my_chat", op=Op.CHAT)
def sync_chat(prompt: str) -> str:
    return f"response to {prompt}"


@observe(name="my_retrieval", op=Op.RETRIEVAL)
def sync_retrieval(query: str) -> list:
    return ["doc1", "doc2"]


@observe(name="plain_fn")
def sync_no_op(x: int) -> int:
    return x + 1


@observe
def bare_observe(x: int) -> int:
    return x * 2


@observe(name="error_fn", op=Op.INVOKE_AGENT)
def sync_error_fn():
    raise ValueError("something went wrong")


@observe(name="parent", op=Op.INVOKE_AGENT)
def parent_fn():
    return child_fn("hello")


@observe(name="child", op=Op.EXECUTE_TOOL)
def child_fn(msg: str) -> str:
    return msg.upper()


@observe(name="kwargs_fn", op=Op.INVOKE_AGENT)
def kwargs_fn(*, key: str, value: int) -> dict:
    return {"key": key, "value": value}


class TestDecoratorSync:
    def test_invoke_agent_span(self, exporter):
        result = sync_invoke_agent("test")
        assert result == "answer to test"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "invoke_agent my_agent"
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes["gen_ai.agent.name"] == "my_agent"

    def test_execute_tool_span(self, exporter):
        result = sync_execute_tool(3, 4)
        assert result == 7

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "execute_tool my_tool"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "my_tool"

    def test_chat_span(self, exporter):
        result = sync_chat("hello")
        assert result == "response to hello"

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "chat my_chat"
        assert span.attributes["gen_ai.operation.name"] == "chat"
        assert span.attributes["gen_ai.agent.name"] == "my_chat"

    def test_retrieval_span(self, exporter):
        result = sync_retrieval("query")
        assert result == ["doc1", "doc2"]

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "retrieval my_retrieval"
        assert span.attributes["gen_ai.operation.name"] == "retrieval"

    def test_no_op_span(self, exporter):
        result = sync_no_op(5)
        assert result == 6

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "plain_fn"
        assert "gen_ai.operation.name" not in span.attributes

    def test_bare_observe_without_parens(self, exporter):
        result = bare_observe(5)
        assert result == 10

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "bare_observe"
        assert "gen_ai.operation.name" not in span.attributes

    def test_auto_name_uses_qualname(self, exporter):
        bare_observe(1)
        spans = exporter.get_finished_spans()
        assert spans[0].name == "bare_observe"

    def test_input_capture(self, exporter):
        sync_invoke_agent("test query")
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.input.messages"])
        assert captured == {"query": "test query"}

    def test_tool_input_uses_call_arguments(self, exporter):
        sync_execute_tool(3, 4)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.tool.call.arguments"])
        assert captured == {"a": 3, "b": 4}
        assert "gen_ai.input.messages" not in span.attributes

    def test_output_capture(self, exporter):
        sync_invoke_agent("test")
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert json.loads(span.attributes["gen_ai.output.messages"]) == "answer to test"

    def test_tool_output_uses_call_result(self, exporter):
        sync_execute_tool(3, 4)
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert json.loads(span.attributes["gen_ai.tool.call.result"]) == 7
        assert "gen_ai.output.messages" not in span.attributes

    def test_kwargs_input_capture(self, exporter):
        kwargs_fn(key="x", value=99)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = json.loads(span.attributes["gen_ai.input.messages"])
        assert captured == {"key": "x", "value": 99}

    def test_error_sets_span_status(self, exporter):
        with pytest.raises(ValueError, match="something went wrong"):
            sync_error_fn()

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "something went wrong" in span.status.description

    def test_error_records_exception_event(self, exporter):
        with pytest.raises(ValueError):
            sync_error_fn()

        spans = exporter.get_finished_spans()
        span = spans[0]
        exc_events = [e for e in span.events if e.name == "exception"]
        assert len(exc_events) >= 1
        assert "ValueError" in exc_events[0].attributes["exception.type"]


class TestParentChild:
    def test_nested_spans(self, exporter):
        result = parent_fn()
        assert result == "HELLO"

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        child_span = next(s for s in spans if s.name == "execute_tool child")
        parent_span = next(s for s in spans if s.name == "invoke_agent parent")

        assert child_span.context.trace_id == parent_span.context.trace_id
        assert child_span.parent.span_id == parent_span.context.span_id


# ---------------------------------------------------------------------------
# Decorator — async
# ---------------------------------------------------------------------------


@observe(name="async_agent", op=Op.INVOKE_AGENT)
async def async_invoke_agent(query: str) -> str:
    return f"async answer to {query}"


@observe(name="async_tool", op=Op.EXECUTE_TOOL)
async def async_execute_tool(a: int, b: int) -> int:
    return a + b


@observe(name="async_error", op=Op.INVOKE_AGENT)
async def async_error_fn():
    raise RuntimeError("async boom")


class TestDecoratorAsync:
    @pytest.mark.asyncio
    async def test_async_invoke_agent(self, exporter):
        result = await async_invoke_agent("hello")
        assert result == "async answer to hello"

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "invoke_agent async_agent"
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"

    @pytest.mark.asyncio
    async def test_async_execute_tool(self, exporter):
        result = await async_execute_tool(10, 20)
        assert result == 30

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "execute_tool async_tool"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"

    @pytest.mark.asyncio
    async def test_async_error(self, exporter):
        with pytest.raises(RuntimeError, match="async boom"):
            await async_error_fn()

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Decorator — generators
# ---------------------------------------------------------------------------


@observe(name="gen_tool", op=Op.EXECUTE_TOOL)
def generator_fn(n: int):
    yield from range(n)


@observe(name="async_gen_tool", op=Op.EXECUTE_TOOL)
async def async_generator_fn(n: int):
    for i in range(n):
        yield i


@observe(name="gen_error", op=Op.EXECUTE_TOOL)
def generator_error_fn():
    yield 1
    raise ValueError("gen error")


@observe(name="async_gen_error", op=Op.EXECUTE_TOOL)
async def async_generator_error_fn():
    yield 1
    raise ValueError("async gen error")


class TestGenerators:
    def test_sync_generator(self, exporter):
        items = list(generator_fn(3))
        assert items == [0, 1, 2]

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "execute_tool gen_tool"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"

    def test_sync_generator_output(self, exporter):
        items = list(generator_fn(3))
        assert items == [0, 1, 2]
        span = exporter.get_finished_spans()[0]
        assert json.loads(span.attributes["gen_ai.tool.call.result"]) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_async_generator(self, exporter):
        items = []
        async for item in async_generator_fn(3):
            items.append(item)
        assert items == [0, 1, 2]

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "execute_tool async_gen_tool"

    def test_sync_generator_error(self, exporter):
        gen = generator_error_fn()
        assert next(gen) == 1
        with pytest.raises(ValueError, match="gen error"):
            next(gen)

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_async_generator_error(self, exporter):
        agen = async_generator_error_fn()
        first = await agen.__anext__()
        assert first == 1
        with pytest.raises(ValueError, match="async gen error"):
            await agen.__anext__()

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_basic_context_manager(self, exporter):
        with observe("my_block", op=Op.INVOKE_AGENT) as span:
            span.set_attribute("custom", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "invoke_agent my_block"
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes["custom"] == "value"

    def test_context_manager_no_op(self, exporter):
        with observe("plain_block"):
            pass

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "plain_block"
        assert "gen_ai.operation.name" not in span.attributes

    def test_context_manager_error(self, exporter):
        with pytest.raises(ValueError, match="ctx error"):
            with observe("error_block", op=Op.CHAT):
                raise ValueError("ctx error")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

    def test_nested_context_managers(self, exporter):
        with observe("outer", op=Op.INVOKE_AGENT):
            with observe("inner", op=Op.EXECUTE_TOOL):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        inner = next(s for s in spans if s.name == "execute_tool inner")
        outer = next(s for s in spans if s.name == "invoke_agent outer")

        assert inner.context.trace_id == outer.context.trace_id
        assert inner.parent.span_id == outer.context.span_id

    def test_context_manager_with_custom_op(self, exporter):
        with observe("check_safety", op="guardrail"):
            pass

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.name == "check_safety"
        assert span.attributes["gen_ai.operation.name"] == "guardrail"


# ---------------------------------------------------------------------------
# Op constants
# ---------------------------------------------------------------------------


class TestOpConstants:
    def test_well_known_values(self):
        assert Op.CHAT == "chat"
        assert Op.CREATE_AGENT == "create_agent"
        assert Op.INVOKE_AGENT == "invoke_agent"
        assert Op.EXECUTE_TOOL == "execute_tool"
        assert Op.RETRIEVAL == "retrieval"
        assert Op.EMBEDDINGS == "embeddings"
        assert Op.GENERATE_CONTENT == "generate_content"
        assert Op.TEXT_COMPLETION == "text_completion"


# ---------------------------------------------------------------------------
# functools.wraps preservation
# ---------------------------------------------------------------------------


class TestFunctoolsWraps:
    def test_sync_preserves_name(self):
        assert sync_invoke_agent.__name__ == "sync_invoke_agent"

    def test_async_preserves_name(self):
        assert async_invoke_agent.__name__ == "async_invoke_agent"

    def test_generator_preserves_name(self):
        assert generator_fn.__name__ == "generator_fn"

    def test_bare_preserves_name(self):
        assert bare_observe.__name__ == "bare_observe"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_return_no_output_attribute(self, exporter):
        @observe(name="none_return", op=Op.INVOKE_AGENT)
        def returns_none():
            return None

        returns_none()
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.output.messages" not in span.attributes

    def test_no_args_no_input_attribute(self, exporter):
        @observe(name="no_args", op=Op.INVOKE_AGENT)
        def no_args_fn():
            return 42

        no_args_fn()
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert "gen_ai.input.messages" not in span.attributes

    def test_large_input_is_truncated(self, exporter):
        @observe(name="big_input", op=Op.INVOKE_AGENT)
        def big_input_fn(data: str) -> str:
            return "ok"

        big_input_fn("x" * 20_000)
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = span.attributes["gen_ai.input.messages"]
        assert len(captured) <= 10_100
        assert "truncated" in captured

    def test_large_output_is_truncated(self, exporter):
        @observe(name="big_output", op=Op.INVOKE_AGENT)
        def big_output_fn() -> str:
            return "y" * 20_000

        big_output_fn()
        spans = exporter.get_finished_spans()
        span = spans[0]
        captured = span.attributes["gen_ai.output.messages"]
        assert len(captured) <= 10_100
        assert "truncated" in captured

    def test_non_serializable_input_does_not_crash(self, exporter):
        @observe(name="non_serial", op=Op.INVOKE_AGENT)
        def non_serial_fn(obj):
            return "ok"

        class Weird:
            pass

        non_serial_fn(Weird())
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

    def test_non_serializable_output_does_not_crash(self, exporter):
        @observe(name="non_serial_out", op=Op.INVOKE_AGENT)
        def returns_weird():
            class Weird:
                pass

            return Weird()

        returns_weird()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

    def test_span_kind_override(self, exporter):
        @observe(name="remote_agent", op=Op.INVOKE_AGENT, kind=SpanKind.CLIENT)
        def remote_call():
            return "ok"

        remote_call()
        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.kind == SpanKind.CLIENT


# ---------------------------------------------------------------------------
# name_from — dynamic span naming
# ---------------------------------------------------------------------------


class TestNameFrom:
    def test_name_from_parameter(self, exporter):
        @observe(op=Op.EXECUTE_TOOL, name_from="tool_name")
        def run_tool(tool_name: str, args: dict) -> str:
            return "result"

        run_tool("web_search", {"q": "hello"})
        span = exporter.get_finished_spans()[0]
        assert span.name == "execute_tool web_search"
        assert span.attributes["gen_ai.tool.name"] == "web_search"

    def test_name_from_with_static_fallback(self, exporter):
        @observe(name="fallback", op=Op.EXECUTE_TOOL, name_from="tool_name")
        def run_tool(tool_name: str) -> str:
            return "result"

        run_tool("calculator")
        span = exporter.get_finished_spans()[0]
        assert span.name == "execute_tool calculator"
        assert span.attributes["gen_ai.tool.name"] == "calculator"

    def test_name_from_missing_param_uses_static_name(self, exporter):
        @observe(name="dispatcher", op=Op.EXECUTE_TOOL, name_from="tool_name")
        def run_tool(args: dict) -> str:
            return "result"

        run_tool({"q": "hello"})
        span = exporter.get_finished_spans()[0]
        assert span.name == "execute_tool dispatcher"

    def test_name_from_none_value_uses_static_name(self, exporter):
        @observe(name="dispatcher", op=Op.EXECUTE_TOOL, name_from="tool_name")
        def run_tool(tool_name: str = None) -> str:
            return "result"

        run_tool()
        span = exporter.get_finished_spans()[0]
        assert span.name == "execute_tool dispatcher"

    def test_name_from_with_invoke_agent(self, exporter):
        @observe(op=Op.INVOKE_AGENT, name_from="agent_name")
        def dispatch(agent_name: str, query: str) -> str:
            return "done"

        dispatch("research-agent", "find papers")
        span = exporter.get_finished_spans()[0]
        assert span.name == "invoke_agent research-agent"
        assert span.attributes["gen_ai.agent.name"] == "research-agent"


# ---------------------------------------------------------------------------
# Tool type attribute
# ---------------------------------------------------------------------------


class TestToolTypeAttribute:
    def test_execute_tool_sets_tool_type(self, exporter):
        @observe(name="my_tool", op=Op.EXECUTE_TOOL)
        def tool_fn() -> str:
            return "ok"

        tool_fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.tool.type"] == "function"

    def test_non_tool_op_does_not_set_tool_type(self, exporter):
        @observe(name="my_agent", op=Op.INVOKE_AGENT)
        def agent_fn() -> str:
            return "ok"

        agent_fn()
        span = exporter.get_finished_spans()[0]
        assert "gen_ai.tool.type" not in span.attributes
