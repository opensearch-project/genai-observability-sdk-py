# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Travel assistant on LlamaIndex LLMs, traced with OpenSearch.

    python agent.py "Plan a one-day trip to Paris."

This file drives a LlamaIndex LLM (Claude on Amazon Bedrock through
BedrockConverse) with a small tool-calling loop. Tracing:
- register() turns on the LlamaIndex instrumentor (installed by the
  [llamaindex] extra): every LLM chat call becomes a `chat` span.
- @observe adds the `invoke_agent travel_assistant` span around the loop and
  an `execute_tool <name>` span per tool call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.bedrock_converse import BedrockConverse

from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

# OTLP endpoint: OTEL_EXPORTER_OTLP_ENDPOINT, else http://localhost:4318.
register(service_name="travel-assistant-llamaindex", auto_instrument=True)

DATA = json.loads((Path(__file__).parent / "../../shared/travel_data.json").read_text())
SYSTEM_PROMPT = DATA["system_prompt"]
CITIES = DATA["cities"]

# Any LlamaIndex LLM that supports tool calling works here (OpenAI, Anthropic, ...).
# BedrockConverse reads AWS_PROFILE like any AWS SDK.
llm = BedrockConverse(
    model=os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    region_name=os.environ.get("AWS_REGION", "us-west-2"),
    max_tokens=1024,
)
MAX_TURNS = 5


@observe(name="get_weather", op=Op.EXECUTE_TOOL)
def get_weather(city: str) -> dict:
    """Weather forecast for a city."""
    data = CITIES.get(city.strip().lower())
    if data is None:
        return {"error": f"no data for {city}"}
    return {"city": data["name"], **data["weather"]}


@observe(name="find_events", op=Op.EXECUTE_TOOL)
def find_events(city: str) -> dict:
    """Events happening in a city."""
    data = CITIES.get(city.strip().lower())
    if data is None:
        return {"error": f"no data for {city}"}
    return {"city": data["name"], "events": data["events"]}


TOOLS = {"get_weather": get_weather, "find_events": find_events}
TOOL_SPECS = [FunctionTool.from_defaults(fn=fn, name=name) for name, fn in TOOLS.items()]


@observe(name="travel_assistant", op=Op.INVOKE_AGENT)
def travel_assistant(question: str) -> str:
    # The instrumentor records the messages of each chat call. On the agent span,
    # record the user question and the final answer as GenAI messages.
    enrich(
        system_instructions=SYSTEM_PROMPT,
        input_messages=[{"role": "user", "parts": [{"type": "text", "content": question}]}],
    )
    history = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=question),
    ]
    answer = ""
    for _ in range(MAX_TURNS):
        response = llm.chat_with_tools(
            TOOL_SPECS, chat_history=history, allow_parallel_tool_calls=True
        )
        history.append(response.message)
        calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        if not calls:
            answer = response.message.content or ""
            break
        for call in calls:
            tool = TOOLS.get(call.tool_name)
            result = tool(**call.tool_kwargs) if tool else {"error": f"unknown {call.tool_name}"}
            history.append(
                ChatMessage(
                    role="tool",
                    content=json.dumps(result),
                    additional_kwargs={"tool_call_id": call.tool_id},
                )
            )
    enrich(output_messages=[{"role": "assistant", "parts": [{"type": "text", "content": answer}]}])
    return answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Plan a one-day trip to Paris."
    print(travel_assistant(question))
