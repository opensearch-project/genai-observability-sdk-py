# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""The travel assistant under evaluation.

A Strands agent on Amazon Bedrock with two tools, get_weather and find_events, that read
the static data in ../../shared/travel_data.json. Strands' own instrumentation records the
invoke_agent, chat and execute_tool spans; register() (in the caller) sends them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Strands records the latest GenAI semantic conventions (messages as span attributes)
# only with this opt-in. Set it before strands is imported.
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental,gen_ai_span_attributes_only"
)

from strands import Agent, tool  # noqa: E402
from strands.models.bedrock import BedrockModel  # noqa: E402

SHARED = Path(__file__).resolve().parent / ".." / ".." / "shared"
TRAVEL = json.loads((SHARED / "travel_data.json").read_text())
SYSTEM_PROMPT = TRAVEL["system_prompt"]
CITIES = TRAVEL["cities"]

AGENT_NAME = "travel_assistant"
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")


def load_test_suite() -> list[dict]:
    """The shared test cases as evaluate() rows (case_id, case_name, input, expected)."""
    rows = []
    for line in (SHARED / "test_suite.jsonl").read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            rows.append(
                {
                    "case_id": case["id"],
                    "case_name": case["name"],
                    "input": case["input"],
                    "expected": case["expected"],
                }
            )
    return rows


@tool
def get_weather(city: str) -> str:
    """Get the weather forecast for a city."""
    data = CITIES.get(city.strip().lower())
    if not data:
        return json.dumps({"error": f"no data for {city}"})
    return json.dumps({"city": data["name"], **data["weather"]})


@tool
def find_events(city: str) -> str:
    """Find events happening in a city."""
    data = CITIES.get(city.strip().lower())
    if not data:
        return json.dumps({"error": f"no data for {city}"})
    return json.dumps({"city": data["name"], "events": data["events"]})


@dataclass
class AgentRun:
    answer: str
    tool_results: list[str]  # what the tools returned, the context the answer must use
    trace_id: str  # the agent run, to link evaluation results to it
    span_id: str


def run_agent(
    question: str, *, system_prompt: str = SYSTEM_PROMPT, version: str = "v1"
) -> AgentRun:
    """Run the agent once. Each call is one trace with an invoke_agent root span."""
    agent = Agent(
        name=AGENT_NAME,
        model=BedrockModel(model_id=MODEL_ID, region_name=REGION),
        tools=[get_weather, find_events],
        system_prompt=system_prompt,
        callback_handler=None,
        trace_attributes={"gen_ai.agent.version": version},
    )
    answer = str(agent(question)).strip()
    tool_results = [
        part["text"]
        for message in agent.messages
        for block in message["content"]
        if "toolResult" in block
        for part in block["toolResult"]["content"]
        if "text" in part
    ]
    ctx = agent.trace_span.get_span_context()  # Strands' invoke_agent span
    return AgentRun(answer, tool_results, format(ctx.trace_id, "032x"), format(ctx.span_id, "016x"))
