# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Travel assistant on CrewAI, traced with the OpenSearch GenAI Observability SDK.

register() activates the CrewAI and Bedrock instrumentors. The CrewAI instrumentor
records the crew, task and agent; the Bedrock instrumentor records each model call.
The CrewAI instrumentor records no tool calls, so the tools use the SDK's observe().

Run:
    pip install -r requirements.txt
    python agent.py                      # plans a day in Paris
    python agent.py "Plan a trip to Tokyo."
"""

import json
import logging
import os
import sys
from pathlib import Path

from crewai import Agent, Crew, Task
from crewai.tools import tool

from opensearch_genai_observability_sdk_py import Op, observe, register

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
os.environ.setdefault("AWS_REGION", "us-west-2")

DATA = json.loads((Path(__file__).resolve().parents[2] / "shared" / "travel_data.json").read_text())
CITIES = DATA["cities"]
# CrewAI warns on every Bedrock call that it estimates tokens; the spans use Bedrock's counts.
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

# Sends traces to the local OTel Collector (http://localhost:4318) by default.
provider = register(service_name="travel-assistant-crewai-py")


@tool("get_weather")
@observe(name="get_weather", op=Op.EXECUTE_TOOL)  # below @tool: records arguments and result
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], **info["weather"]}


@tool("find_events")
@observe(name="find_events", op=Op.EXECUTE_TOOL)  # below @tool: records arguments and result
def find_events(city: str) -> dict:
    """Find events happening in a city."""
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], "events": info["events"]}


assistant = Agent(
    role="travel_assistant",
    goal="Plan short trips using only the weather and events the tools return",
    backstory=DATA["system_prompt"],
    llm=f"bedrock/{MODEL_ID}",
    tools=[get_weather, find_events],
)


def main() -> None:
    questions = sys.argv[1:] or ["Plan a one-day trip to Paris."]
    for question in questions:
        task = Task(description=question, expected_output="A short trip plan", agent=assistant)
        answer = Crew(agents=[assistant], tasks=[task]).kickoff()
        print(f"> {question}\n{str(answer).strip()}\n")
    provider.force_flush()


if __name__ == "__main__":
    main()
