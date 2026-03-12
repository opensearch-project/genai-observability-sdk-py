#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Multi-Agent Travel Planner — instrumented with opensearch-genai-observability-sdk-py.

This is an SDK-instrumented version of the multi-agent-planner from the
observability-stack repo. It replaces ~100 lines of manual OTel setup
(TracerProvider, Resource, BatchSpanProcessor, OTLPSpanExporter, manual
span.set_attribute() calls) with register() + @observe + enrich().

Usage:
    # Terminal 1: start the mini collector
    python examples/mini_collector_http.py

    # Terminal 2: run this example
    python examples/multi_agent_planner/run.py

Span tree produced:

    invoke_agent travel-planner
    ├── chat planning                          (LLM "thinking" step)
    ├── invoke_agent weather-agent
    │   ├── chat weather-reasoning             (LLM decides which tool)
    │   └── execute_tool fetch_weather_api     (tool execution)
    ├── invoke_agent events-agent
    │   ├── chat events-reasoning              (LLM decides which tool)
    │   └── execute_tool fetch_events_api      (tool execution)
    └── chat summarize                         (LLM final response)
"""

import random
import time

from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

# --- Setup: one line replaces ~30 lines of manual OTel config ---
register(
    endpoint="http://localhost:4318/v1/traces",
    service_name="travel-planner",
)

# --- Shared data ---

MODELS = [
    "claude-sonnet-4.5", "gpt-4.1", "gpt-4o-mini", "gemini-2.5-flash", "nova-pro",
]
SYSTEMS = {
    "claude-sonnet-4.5": "anthropic",
    "gpt-4.1": "openai", "gpt-4o-mini": "openai",
    "gemini-2.5-flash": "google",
    "nova-pro": "amazon",
}

SAMPLE_WEATHER = {
    "paris": {"temperature": "57°F", "condition": "rainy", "humidity": "85%"},
    "tokyo": {"temperature": "72°F", "condition": "sunny", "humidity": "60%"},
    "london": {"temperature": "50°F", "condition": "cloudy", "humidity": "75%"},
    "new york": {"temperature": "65°F", "condition": "partly cloudy", "humidity": "55%"},
}

SAMPLE_EVENTS = {
    "paris": [
        {"name": "Louvre Late Night", "type": "museum", "venue": "Louvre Museum"},
        {"name": "Seine River Cruise", "type": "tour", "venue": "Port de la Bourdonnais"},
    ],
    "tokyo": [
        {"name": "Shibuya Night Walk", "type": "tour", "venue": "Shibuya"},
        {"name": "Tsukiji Outer Market", "type": "food", "venue": "Tsukiji"},
    ],
    "london": [
        {"name": "West End Show", "type": "theater", "venue": "Various"},
        {"name": "Borough Market Food Tour", "type": "food", "venue": "Borough Market"},
    ],
    "new york": [
        {"name": "Broadway Show", "type": "theater", "venue": "Times Square"},
        {"name": "Central Park Walk", "type": "tour", "venue": "Central Park"},
    ],
}


def _pick_model():
    model = random.choice(MODELS)    return model, SYSTEMS[model]


# --- Tools (leaf-level) ---

@observe(name="fetch_weather_api", op=Op.EXECUTE_TOOL)
def fetch_weather(location: str) -> dict:
    """Fetch weather data for a location."""
    time.sleep(random.uniform(0.05, 0.15))    key = location.lower()
    data = SAMPLE_WEATHER.get(key, {"temperature": "68°F", "condition": "sunny", "humidity": "50%"})
    return {"location": location, **data}


@observe(name="fetch_events_api", op=Op.EXECUTE_TOOL)
def fetch_events(destination: str) -> list[dict]:
    """Fetch local events for a destination."""
    time.sleep(random.uniform(0.05, 0.15))    key = destination.lower()
    default = [{"name": f"{destination} Walking Tour", "type": "tour", "venue": "City Center"}]
    return SAMPLE_EVENTS.get(key, default)


# --- Sub-agents ---

@observe(name="weather-agent", op=Op.INVOKE_AGENT)
def weather_agent(destination: str) -> dict:
    """Weather sub-agent: reasons about query, then calls weather tool."""
    model, provider = _pick_model()
    enrich(
        model=model,
        provider=provider,
        agent_id="weather-agent-001",
    )

    # LLM "thinking" step — decides which tool to call
    with observe("weather-reasoning", op=Op.CHAT):
        input_tokens = random.randint(100, 500)        output_tokens = random.randint(50, 200)        enrich(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="tool_calls",
        )
        time.sleep(random.uniform(0.05, 0.15))
    # Tool execution
    weather = fetch_weather(destination)

    return {
        "response": f"The weather in {destination} is {weather['condition']} "
                     f"at {weather['temperature']} with {weather['humidity']} humidity.",
        **weather,
    }


@observe(name="events-agent", op=Op.INVOKE_AGENT)
def events_agent(destination: str) -> dict:
    """Events sub-agent: reasons about query, then calls events tool."""
    model, provider = _pick_model()
    enrich(
        model=model,
        provider=provider,
        agent_id="events-agent-001",
    )

    # LLM "thinking" step
    with observe("events-reasoning", op=Op.CHAT):
        input_tokens = random.randint(100, 500)        output_tokens = random.randint(50, 200)        enrich(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="tool_calls",
        )
        time.sleep(random.uniform(0.05, 0.15))
    # Tool execution
    events = fetch_events(destination)

    return {"destination": destination, "events": events}


# --- Orchestrator ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a destination via the weather agent",
            "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events",
            "description": "Get local events for a destination via the events agent",
            "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}},
        },
    },
]


@observe(name="travel-planner", op=Op.INVOKE_AGENT)
def plan_trip(destination: str) -> dict:
    """Orchestrator agent: plans a trip by calling weather + events sub-agents."""
    model, provider = _pick_model()
    enrich(
        model=model,
        provider=provider,
        agent_id="travel-planner-001",
        tool_definitions=TOOL_DEFINITIONS,
        input_messages=[{"role": "user", "parts": [{"type": "text", "content": f"Plan a trip to {destination}"}]}],
    )

    # Step 1: LLM "planning" step — decides to call both sub-agents
    with observe("planning", op=Op.CHAT):
        input_tokens = random.randint(500, 2000)        output_tokens = random.randint(100, 500)        enrich(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="tool_calls",
        )
        time.sleep(random.uniform(0.1, 0.3))
    # Step 2: Fan out to sub-agents
    weather_data = weather_agent(destination)
    events_data = events_agent(destination)

    # Step 3: LLM summarize step — generates final recommendation
    with observe("summarize", op=Op.CHAT):
        input_tokens = random.randint(200, 800)
        output_tokens = random.randint(50, 200)
        enrich(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
        )
        time.sleep(random.uniform(0.05, 0.15))

    # Build recommendation
    parts = [f"Great choice! {destination} looks wonderful."]
    parts.append(weather_data["response"])
    if events_data["events"]:
        event_names = [e["name"] for e in events_data["events"][:2]]
        parts.append(f"Check out {', '.join(event_names)}.")
    recommendation = " ".join(parts)

    enrich(
        output_messages=[{"role": "assistant", "parts": [{"type": "text", "content": recommendation}]}],
    )

    return {
        "destination": destination,
        "weather": weather_data,
        "events": events_data["events"],
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Travel Planner — SDK Instrumented")
    print("=" * 60)
    print()

    destinations = ["Paris", "Tokyo", "London", "New York"]
    for dest in destinations:
        print(f"Planning trip to {dest}...")
        result = plan_trip(dest)
        print(f"  -> {result['recommendation']}")
        print()

    # Give BatchSpanProcessor time to flush
    time.sleep(2)
    print("Done! Check the mini_collector_http.py output for traces.")
