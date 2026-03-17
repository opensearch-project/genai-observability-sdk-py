# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for enrich() — span enrichment with GenAI semantic convention attributes."""

import json

from opensearch_genai_observability_sdk_py.enrich import enrich
from opensearch_genai_observability_sdk_py.observe import Op, observe

# ---------------------------------------------------------------------------
# enrich() inside @observe decorator
# ---------------------------------------------------------------------------


class TestEnrichInDecorator:
    def test_model_and_provider(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(model="gpt-4.1", provider="openai")
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.request.model"] == "gpt-4.1"
        assert span.attributes["gen_ai.provider.name"] == "openai"

    def test_token_usage(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(input_tokens=1500, output_tokens=200, total_tokens=1700)
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.usage.input_tokens"] == 1500
        assert span.attributes["gen_ai.usage.output_tokens"] == 200
        assert span.attributes["gen_ai.usage.total_tokens"] == 1700

    def test_response_metadata(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(response_id="chatcmpl-123", finish_reason="stop")
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.response.id"] == "chatcmpl-123"
        assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)

    def test_request_params(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(temperature=0.7, max_tokens=1024)
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.request.temperature"] == 0.7
        assert span.attributes["gen_ai.request.max_tokens"] == 1024

    def test_session_and_agent_id(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(session_id="conv-001", agent_id="agent-001", agent_description="A helpful agent")
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.conversation.id"] == "conv-001"
        assert span.attributes["gen_ai.agent.id"] == "agent-001"
        assert span.attributes["gen_ai.agent.description"] == "A helpful agent"

    def test_tool_definitions(self, exporter):
        tools = [
            {"type": "function", "function": {"name": "search", "description": "Search the web"}}
        ]

        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(tool_definitions=tools)
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        captured = json.loads(span.attributes["gen_ai.tool.definitions"])
        assert captured[0]["function"]["name"] == "search"

    def test_system_instructions(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(system_instructions="You are a helpful assistant.")
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.system_instructions"] == "You are a helpful assistant."

    def test_input_output_messages(self, exporter):
        input_msgs = [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
        output_msgs = [{"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}]

        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(input_messages=input_msgs, output_messages=output_msgs)
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert json.loads(span.attributes["gen_ai.input.messages"]) == input_msgs
        assert json.loads(span.attributes["gen_ai.output.messages"]) == output_msgs

    def test_extra_kwargs(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(model="gpt-4.1", destination="Paris", customer_tier="premium")
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.request.model"] == "gpt-4.1"
        assert span.attributes["destination"] == "Paris"
        assert span.attributes["customer_tier"] == "premium"

    def test_none_values_are_skipped(self, exporter):
        @observe(name="agent", op=Op.INVOKE_AGENT)
        def fn():
            enrich(model="gpt-4.1", provider=None, input_tokens=None)
            return "ok"

        fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.request.model"] == "gpt-4.1"
        assert "gen_ai.provider.name" not in span.attributes
        assert "gen_ai.usage.input_tokens" not in span.attributes


# ---------------------------------------------------------------------------
# enrich() inside context manager
# ---------------------------------------------------------------------------


class TestEnrichInContextManager:
    def test_enrich_in_context_manager(self, exporter):
        with observe("thinking", op=Op.CHAT):
            enrich(model="gpt-4.1", input_tokens=500, finish_reason="tool_calls")

        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.request.model"] == "gpt-4.1"
        assert span.attributes["gen_ai.usage.input_tokens"] == 500
        assert span.attributes["gen_ai.response.finish_reasons"] == ("tool_calls",)

    def test_multiple_enrich_calls(self, exporter):
        with observe("agent", op=Op.INVOKE_AGENT):
            enrich(model="gpt-4.1", provider="openai")
            # ... later, after LLM responds:
            enrich(input_tokens=1500, output_tokens=200, response_id="chatcmpl-456")

        span = exporter.get_finished_spans()[0]
        assert span.attributes["gen_ai.request.model"] == "gpt-4.1"
        assert span.attributes["gen_ai.provider.name"] == "openai"
        assert span.attributes["gen_ai.usage.input_tokens"] == 1500
        assert span.attributes["gen_ai.usage.output_tokens"] == 200
        assert span.attributes["gen_ai.response.id"] == "chatcmpl-456"


# ---------------------------------------------------------------------------
# enrich() outside any span (no-op, should not crash)
# ---------------------------------------------------------------------------


class TestEnrichNoSpan:
    def test_enrich_without_span_does_not_crash(self, exporter):
        # No active span — enrich should silently no-op
        enrich(model="gpt-4.1", input_tokens=100)
        spans = exporter.get_finished_spans()
        assert len(spans) == 0
