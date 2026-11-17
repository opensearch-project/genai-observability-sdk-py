"""Travel assistant on Strands Agents, traced with the OpenSearch GenAI Observability SDK.

Strands has built-in OpenTelemetry tracing. It uses the global tracer provider that
register() installs, so there is nothing else to wire up.

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

# Strands reads this when it builds its tracer, so set it before importing Strands:
# latest GenAI semantic conventions, with messages stored as span attributes.
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN",
    "gen_ai_latest_experimental,gen_ai_span_attributes_only",
)

from strands import Agent, tool  # noqa: E402
from strands.models.bedrock import BedrockModel  # noqa: E402

from opensearch_genai_observability_sdk_py import register  # noqa: E402

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")

DATA = json.loads((Path(__file__).resolve().parents[2] / "shared" / "travel_data.json").read_text())
CITIES = DATA["cities"]

# Sends traces to the local OTel Collector (http://localhost:4318) by default.
# Strands traces itself, so auto_instrument=False keeps an installed Bedrock (botocore)
# instrumentor from adding a second chat span for every model call.
provider = register(service_name="travel-assistant-strands", auto_instrument=False)


@tool
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], **info["weather"]}


@tool
def find_events(city: str) -> dict:
    """Find events happening in a city."""
    info = CITIES.get(city.strip().lower())
    if info is None:
        return {"error": f"no data for {city}"}
    return {"city": info["name"], "events": info["events"]}


def main() -> None:
    questions = sys.argv[1:] or ["Plan a one-day trip to Paris."]
    agent = Agent(
        name="travel_assistant",  # becomes gen_ai.agent.name on the invoke_agent span
        model=BedrockModel(model_id=MODEL_ID, region_name=REGION),
        system_prompt=DATA["system_prompt"],
        tools=[get_weather, find_events],
        # Groups the turns of this run into one session in Agent Traces.
        trace_attributes={"gen_ai.conversation.id": f"conv-{uuid.uuid4().hex[:8]}"},
        callback_handler=None,  # do not stream the answer to stdout
    )
    for question in questions:
        print(f"> {question}\n{str(agent(question)).strip()}\n")
    provider.force_flush()


if __name__ == "__main__":
    main()
