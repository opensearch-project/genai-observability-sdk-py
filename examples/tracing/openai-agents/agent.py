# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Travel assistant on the OpenAI Agents SDK, traced with the OpenSearch GenAI Observability SDK.

register() activates the OpenAI Agents instrumentor (from the SDK's [openai] extra), which
traces the agent run, every model call and every tool call.

The model is OpenAI when OPENAI_API_KEY is set; otherwise Claude on Amazon Bedrock
through LiteLLM (no OpenAI account needed).

Run:
    pip install -r requirements.txt
    python agent.py                      # plans a day in Paris
    python agent.py "Plan a trip to Tokyo."
"""

import json
import os
import sys
from pathlib import Path

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from opensearch_genai_observability_sdk_py import register

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
os.environ.setdefault("AWS_REGION", "us-west-2")

DATA = json.loads((Path(__file__).resolve().parents[2] / "shared" / "travel_data.json").read_text())
CITIES = DATA["cities"]

# Sends traces to the local OTel Collector (http://localhost:4318) by default.
# auto_instrument=False, then only the Agents SDK instrumentor: the [openai] extra also installs
# the OpenAI client instrumentor, and the Agents SDK calls the model through that client, so
# auto-instrumentation would trace each model call twice (and a Bedrock instrumentor, if
# installed, would do the same on the LiteLLM path).
provider = register(service_name="travel-assistant-openai-agents-py", auto_instrument=False)
OpenAIAgentsInstrumentor().instrument()


@function_tool
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], **info["weather"]}


@function_tool
def find_events(city: str) -> dict:
    """Find events happening in a city."""
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], "events": info["events"]}


if os.environ.get("OPENAI_API_KEY"):
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
else:
    model = LitellmModel(model=f"bedrock/{MODEL_ID}")  # LiteLLM calls Bedrock through boto3

agent = Agent(
    name="travel_assistant",  # becomes gen_ai.agent.name
    instructions=DATA["system_prompt"],
    model=model,
    tools=[get_weather, find_events],
)


def main() -> None:
    questions = sys.argv[1:] or ["Plan a one-day trip to Paris."]
    history: list = []
    for question in questions:
        result = Runner.run_sync(agent, history + [{"role": "user", "content": question}])
        history = result.to_input_list()  # the next question sees this conversation
        print(f"> {question}\n{result.final_output.strip()}\n")
    provider.force_flush()


if __name__ == "__main__":
    main()
