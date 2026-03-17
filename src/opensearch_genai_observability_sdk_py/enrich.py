# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Span enrichment for GenAI observability.

Provides ``enrich()`` — a convenience function to add GenAI semantic
convention attributes to the currently active span. Use it inside
``@observe``-decorated functions or ``with observe(...)`` blocks to
record model details, token usage, and other metadata without needing
to remember exact attribute names.

Example::

    @observe(name="planner", op=Op.INVOKE_AGENT)
    def plan(query: str) -> str:
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": query}],
        )
        enrich(
            model=response.model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            response_id=response.id,
            finish_reason=response.choices[0].finish_reason,
        )
        return response.choices[0].message.content
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import trace

logger = logging.getLogger(__name__)


def enrich(
    *,
    model: str | None = None,
    provider: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    response_id: str | None = None,
    finish_reason: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    agent_description: str | None = None,
    tool_definitions: list[dict] | None = None,
    system_instructions: str | None = None,
    input_messages: Any | None = None,
    output_messages: Any | None = None,
    **extra: Any,
) -> None:
    """Add GenAI semantic convention attributes to the current active span.

    Call this from inside an ``@observe``-decorated function or a
    ``with observe(...)`` block. It finds the current span automatically
    via OpenTelemetry's context and sets the appropriate ``gen_ai.*``
    attributes.

    Works with **any** tracer provider — our ``register()``, Strands'
    ``StrandsTelemetry``, or any OTel-compatible setup.

    Args:
        model: Model identifier (e.g. ``"gpt-4.1"``, ``"claude-sonnet-4-6"``).
            Sets ``gen_ai.request.model``.
        provider: Provider/system name (e.g. ``"openai"``, ``"anthropic"``).
            Sets ``gen_ai.provider.name``.
        input_tokens: Number of input/prompt tokens.
            Sets ``gen_ai.usage.input_tokens``.
        output_tokens: Number of output/completion tokens.
            Sets ``gen_ai.usage.output_tokens``.
        total_tokens: Total token count.
            Sets ``gen_ai.usage.total_tokens``.
        response_id: LLM response/completion ID.
            Sets ``gen_ai.response.id``.
        finish_reason: Why the model stopped (e.g. ``"stop"``, ``"tool_calls"``).
            Sets ``gen_ai.response.finish_reasons``.
        temperature: Sampling temperature used.
            Sets ``gen_ai.request.temperature``.
        max_tokens: Max tokens requested.
            Sets ``gen_ai.request.max_tokens``.
        session_id: Conversation/session identifier.
            Sets ``gen_ai.conversation.id``.
        agent_id: Agent instance identifier.
            Sets ``gen_ai.agent.id``.
        agent_description: Human-readable agent description.
            Sets ``gen_ai.agent.description``.
        tool_definitions: List of tool/function definitions available to the agent.
            Sets ``gen_ai.tool.definitions`` (JSON-serialized).
        system_instructions: System prompt / instructions.
            Sets ``gen_ai.system_instructions``.
        input_messages: Structured input messages (will be JSON-serialized).
            Sets ``gen_ai.input.messages``.
        output_messages: Structured output messages (will be JSON-serialized).
            Sets ``gen_ai.output.messages``.
        **extra: Additional key-value pairs set directly as span attributes.
            Use for custom or business-specific attributes.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return

    attr_map: dict[str, Any] = {
        "gen_ai.request.model": model,
        "gen_ai.provider.name": provider,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "gen_ai.usage.total_tokens": total_tokens,
        "gen_ai.response.id": response_id,
        "gen_ai.request.temperature": temperature,
        "gen_ai.request.max_tokens": max_tokens,
        "gen_ai.conversation.id": session_id,
        "gen_ai.agent.id": agent_id,
        "gen_ai.agent.description": agent_description,
        "gen_ai.system_instructions": system_instructions,
    }

    for attr_name, value in attr_map.items():
        if value is not None:
            span.set_attribute(attr_name, value)

    if finish_reason is not None:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])

    if tool_definitions is not None:
        try:
            span.set_attribute("gen_ai.tool.definitions", json.dumps(tool_definitions, default=str))
        except Exception:  # noqa: S110
            pass

    if input_messages is not None:
        try:
            serialized = json.dumps(input_messages, default=str)
            span.set_attribute("gen_ai.input.messages", serialized)
        except Exception:  # noqa: S110
            pass

    if output_messages is not None:
        try:
            serialized = json.dumps(output_messages, default=str)
            span.set_attribute("gen_ai.output.messages", serialized)
        except Exception:  # noqa: S110
            pass

    for key, value in extra.items():
        if value is not None:
            span.set_attribute(key, value)
