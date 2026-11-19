"""Travel assistant on LangGraph, traced with the OpenSearch GenAI Observability SDK.

register() activates the LangChain instrumentor (from the SDK's [langchain] extra),
which traces the graph, every model call and every tool call.

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
from typing import Annotated, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from opensearch_genai_observability_sdk_py import register

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.environ.get("AWS_REGION", "us-west-2")

DATA = json.loads((Path(__file__).resolve().parents[2] / "shared" / "travel_data.json").read_text())
CITIES = DATA["cities"]

# Sends traces to the local OTel Collector (http://localhost:4318) by default.
provider = register(service_name="travel-assistant-langgraph-py")


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


TOOLS = [get_weather, find_events]
llm = ChatBedrockConverse(model=MODEL_ID, region_name=REGION).bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: State) -> dict:
    return {"messages": [llm.invoke([SystemMessage(DATA["system_prompt"]), *state["messages"]])]}


graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)
graph.add_edge("tools", "model")
# The graph name becomes gen_ai.agent.name; the checkpointer keeps each thread's history.
app = graph.compile(name="travel_assistant", checkpointer=InMemorySaver())


def main() -> None:
    questions = sys.argv[1:] or ["Plan a one-day trip to Paris."]
    config = {"configurable": {"thread_id": f"conv-{uuid.uuid4().hex[:8]}"}}
    for question in questions:
        result = app.invoke({"messages": [HumanMessage(question)]}, config=config)
        print(f"> {question}\n{result['messages'][-1].text.strip()}\n")
    provider.force_flush()


if __name__ == "__main__":
    main()
