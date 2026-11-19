# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Travel assistant on the Anthropic Python SDK, traced with OpenSearch.

    python agent.py "Plan a one-day trip to Paris."

The Anthropic SDK has no agent loop, so this file has a small one. Tracing:
- register() turns on the Anthropic instrumentor (installed by the [anthropic]
  extra): every messages.create call becomes a `chat` span.
- @observe adds the `invoke_agent travel_assistant` span around the loop and
  an `execute_tool <name>` span per tool call.

With ANTHROPIC_API_KEY set it calls the Anthropic API, otherwise Claude on
Amazon Bedrock (AnthropicBedrock, using your AWS credentials).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic, AnthropicBedrock

from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

# OTLP endpoint: OTEL_EXPORTER_OTLP_ENDPOINT, else http://localhost:4318.
register(service_name="travel-assistant-anthropic-py", auto_instrument=True)

DATA = json.loads((Path(__file__).parent / "../../shared/travel_data.json").read_text())
SYSTEM_PROMPT = DATA["system_prompt"]
CITIES = DATA["cities"]

# anthropic>=1.14 also emits its own call spans when a tracer provider is set.
# open_telemetry=False turns them off so each model call is one `chat` span,
# from the instrumentor, which also records the messages and tool definitions.
if os.environ.get("ANTHROPIC_API_KEY"):
    client = Anthropic(open_telemetry=False)
    MODEL_ID = os.environ.get("MODEL_ID", "claude-haiku-4-5")
else:
    # Reads AWS_PROFILE / AWS_REGION like any AWS SDK.
    client = AnthropicBedrock(open_telemetry=False)
    MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
MAX_TURNS = 5


@observe(name="get_weather", op=Op.EXECUTE_TOOL)
def get_weather(city: str) -> dict:
    data = CITIES.get(city.strip().lower())
    if data is None:
        return {"error": f"no data for {city}"}
    return {"city": data["name"], **data["weather"]}


@observe(name="find_events", op=Op.EXECUTE_TOOL)
def find_events(city: str) -> dict:
    data = CITIES.get(city.strip().lower())
    if data is None:
        return {"error": f"no data for {city}"}
    return {"city": data["name"], "events": data["events"]}


TOOLS = {"get_weather": get_weather, "find_events": find_events}
CITY_PARAM = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}},
    "required": ["city"],
}
TOOL_SPECS = [
    {
        "name": "get_weather",
        "description": "Weather forecast for a city.",
        "input_schema": CITY_PARAM,
    },
    {
        "name": "find_events",
        "description": "Events happening in a city.",
        "input_schema": CITY_PARAM,
    },
]


@observe(name="travel_assistant", op=Op.INVOKE_AGENT)
def travel_assistant(question: str) -> str:
    # The instrumentor records the messages of each chat call. On the agent span,
    # record the user question and the final answer as GenAI messages.
    enrich(
        system_instructions=SYSTEM_PROMPT,
        input_messages=[{"role": "user", "parts": [{"type": "text", "content": question}]}],
    )
    messages = [{"role": "user", "content": question}]
    answer = ""
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL_ID,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOL_SPECS,
            max_tokens=1024,
        )
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            answer = "".join(block.text for block in response.content if block.type == "text")
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for call in tool_uses:
            tool = TOOLS.get(call.name)
            result = tool(**call.input) if tool else {"error": f"unknown tool {call.name}"}
            results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result)}
            )
        messages.append({"role": "user", "content": results})
    enrich(output_messages=[{"role": "assistant", "parts": [{"type": "text", "content": answer}]}])
    return answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Plan a one-day trip to Paris."
    print(travel_assistant(question))
