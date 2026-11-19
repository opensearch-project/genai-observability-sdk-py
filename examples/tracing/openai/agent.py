# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Travel assistant on the OpenAI Python SDK, traced with OpenSearch.

    python agent.py "Plan a one-day trip to Paris."

The OpenAI SDK has no agent loop, so this file has a small one. Tracing:
- register() turns on the OpenAI instrumentor (installed by the [openai] extra):
  every chat.completions.create call becomes a `chat` span.
- @observe adds the `invoke_agent travel_assistant` span around the loop and
  an `execute_tool <name>` span per tool call.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

# OTLP endpoint: OTEL_EXPORTER_OTLP_ENDPOINT, else http://localhost:4318.
register(service_name="travel-assistant-openai-py", auto_instrument=True)

DATA = json.loads((Path(__file__).parent / "../../shared/travel_data.json").read_text())
SYSTEM_PROMPT = DATA["system_prompt"]
CITIES = DATA["cities"]

# OpenAI() reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
MODEL_ID = os.environ.get("MODEL_ID", "gpt-4o-mini")
MAX_TURNS = 5
client = OpenAI()


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
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather forecast for a city.",
            "parameters": CITY_PARAM,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_events",
            "description": "Events happening in a city.",
            "parameters": CITY_PARAM,
        },
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = ""
    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL_ID, messages=messages, tools=TOOL_SPECS, max_tokens=1024
        )
        message = response.choices[0].message
        if not message.tool_calls:
            answer = message.content or ""
            break
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            tool = TOOLS.get(call.function.name)
            result = tool(**args) if tool else {"error": f"unknown tool {call.function.name}"}
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
    enrich(output_messages=[{"role": "assistant", "parts": [{"type": "text", "content": answer}]}])
    return answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Plan a one-day trip to Paris."
    print(travel_assistant(question))
