"""Travel assistant on the Amazon Bedrock Converse API (boto3), traced with the
OpenSearch GenAI observability SDK.

There is no agent framework here: the script runs the tool loop itself.
- register() activates the Bedrock instrumentor (from the SDK's [bedrock] extra), which
  traces each converse() call as a chat span with Bedrock's token usage.
- The SDK's @observe adds the agent span (invoke_agent) and the tool spans (execute_tool).

Run:
    pip install -r requirements.txt
    python agent.py                      # plans a day in Paris
    python agent.py "Plan a trip to Tokyo."
"""

import json
import os
import sys
import uuid
from pathlib import Path

import boto3

from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")

DATA = json.loads((Path(__file__).resolve().parents[2] / "shared" / "travel_data.json").read_text())
CITIES = DATA["cities"]

# Sends traces to the local OTel Collector (http://localhost:4318) by default.
provider = register(service_name="travel-assistant-bedrock")
client = boto3.client("bedrock-runtime", region_name=REGION)


@observe(op=Op.EXECUTE_TOOL)  # records the arguments and the result
def get_weather(city: str) -> dict:
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], **info["weather"]}


@observe(op=Op.EXECUTE_TOOL)
def find_events(city: str) -> dict:
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], "events": info["events"]}


TOOLS = {"get_weather": get_weather, "find_events": find_events}
CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the weather forecast for a city.",
                "inputSchema": {"json": CITY_SCHEMA},
            }
        },
        {
            "toolSpec": {
                "name": "find_events",
                "description": "Find events happening in a city.",
                "inputSchema": {"json": CITY_SCHEMA},
            }
        },
    ]
}


def _text(role: str, text: str) -> list[dict]:
    """A message in the GenAI semantic-convention format that Agent Traces shows."""
    return [{"role": role, "parts": [{"type": "text", "content": text}]}]


@observe(name="travel_assistant", op=Op.INVOKE_AGENT)
def ask(messages: list, question: str, conversation_id: str) -> str:
    enrich(
        model=MODEL_ID,
        provider="aws.bedrock",
        session_id=conversation_id,
        input_messages=_text("user", question),
    )
    messages.append({"role": "user", "content": [{"text": question}]})
    input_tokens = output_tokens = 0
    while True:  # each converse() call is a chat span from the Bedrock instrumentor
        response = client.converse(
            modelId=MODEL_ID,
            system=[{"text": DATA["system_prompt"]}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
        )
        input_tokens += response["usage"]["inputTokens"]
        output_tokens += response["usage"]["outputTokens"]
        message = response["output"]["message"]
        messages.append(message)
        if response["stopReason"] != "tool_use":
            break
        results = []
        for block in message["content"]:
            if "toolUse" in block:
                call = block["toolUse"]
                output = TOOLS[call["name"]](**call["input"])
                results.append(
                    {"toolResult": {"toolUseId": call["toolUseId"], "content": [{"json": output}]}}
                )
        messages.append({"role": "user", "content": results})
    answer = "".join(block.get("text", "") for block in message["content"]).strip()
    enrich(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_messages=_text("assistant", answer),
    )
    return answer


def main() -> None:
    questions = sys.argv[1:] or ["Plan a one-day trip to Paris."]
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    messages: list = []  # the Converse history, shared by the questions of this run
    for question in questions:
        print(f"> {question}\n{ask(messages, question, conversation_id)}\n")
    provider.force_flush()


if __name__ == "__main__":
    main()
