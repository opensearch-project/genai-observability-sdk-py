# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Demo: Run 3 agent versions through evaluate() and view in agent-health UI.

This script runs the same test cases against 3 simulated agent versions,
sending all traces to the agent-health OTLP receiver on localhost:4001.

After running, open http://localhost:4001/experiments to see results.

Usage:
    python examples/09_local_demo.py

Prerequisites:
    pip install opensearch-genai-observability-sdk-py
    # agent-health running on port 4001 (npm run server)
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from opensearch_genai_observability_sdk_py import (
    Op,
    Score,
    enrich,
    evaluate,
    observe,
    register,
)


class OTLPJsonExporter(SpanExporter):
    """Sends spans as OTLP JSON (not protobuf) so agent-health can parse them."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        resource_spans = self._build_resource_spans(spans)
        payload = {"resourceSpans": resource_spans}
        try:
            resp = requests.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                return SpanExportResult.SUCCESS
            return SpanExportResult.FAILURE
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def _build_resource_spans(self, spans: Sequence[ReadableSpan]) -> list[dict]:
        # Group spans by resource
        by_resource: dict[str, list[ReadableSpan]] = {}
        for span in spans:
            key = str(id(span.resource))
            if key not in by_resource:
                by_resource[key] = []
            by_resource[key].append(span)

        result = []
        for _, group in by_resource.items():
            resource_attrs = []
            if group[0].resource:
                for k, v in group[0].resource.attributes.items():
                    resource_attrs.append({"key": k, "value": self._attr_value(v)})

            scope_spans = []
            for span in group:
                scope_spans.append(self._convert_span(span))

            result.append({
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [{"spans": scope_spans}],
            })
        return result

    def _convert_span(self, span: ReadableSpan) -> dict:
        ctx = span.get_span_context()
        attrs = []
        for k, v in (span.attributes or {}).items():
            attrs.append({"key": k, "value": self._attr_value(v)})

        events = []
        for event in span.events or []:
            event_attrs = []
            for k, v in (event.attributes or {}).items():
                event_attrs.append({"key": k, "value": self._attr_value(v)})
            events.append({
                "name": event.name,
                "timeUnixNano": str(event.timestamp or 0),
                "attributes": event_attrs,
            })

        parent_id = ""
        if span.parent and span.parent.span_id:
            parent_id = format(span.parent.span_id, '016x')

        return {
            "traceId": format(ctx.trace_id, '032x'),
            "spanId": format(ctx.span_id, '016x'),
            "parentSpanId": parent_id,
            "name": span.name,
            "startTimeUnixNano": str(span.start_time or 0),
            "endTimeUnixNano": str(span.end_time or 0),
            "attributes": attrs,
            "events": events,
            "status": {
                "code": 2 if span.status and span.status.status_code.value == 2 else 0,
            },
        }

    def _attr_value(self, v: Any) -> dict:
        if isinstance(v, str):
            return {"stringValue": v}
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        if isinstance(v, (list, tuple)):
            return {"arrayValue": {"values": [self._attr_value(x) for x in v]}}
        return {"stringValue": str(v)}

# ---------------------------------------------------------------------------
# Setup — point SDK at agent-health's OTLP receiver
# ---------------------------------------------------------------------------

register(
    exporter=OTLPJsonExporter("http://localhost:4001/v1/traces"),
    service_name="rag-agent-demo",
    batch=False,  # Flush immediately for demo
)

# ---------------------------------------------------------------------------
# Agent versions — each simulates different quality levels
# ---------------------------------------------------------------------------

PROMPTS = {
    "v1": "Answer briefly.",
    "v2": "Answer accurately with detail.",
    "v3": "Answer precisely, cite sources, be thorough.",
}

MODELS = {
    "v1": "gpt-4o-mini",
    "v2": "gpt-4o-mini",
    "v3": "gpt-4.1",
}

RESPONSES = {
    "What is Python?": {
        "v1": "A programming language.",
        "v2": "Python is a high-level programming language for web dev, data science, and automation.",
        "v3": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. Used for web development (Django, Flask), data science (pandas, NumPy), and automation. Source: python.org",
    },
    "What causes rain?": {
        "v1": "Water falls from clouds.",
        "v2": "Rain occurs when water vapor condenses into droplets heavy enough to fall.",
        "v3": "Rain forms through the water cycle: evaporation creates water vapor, which rises, cools, and condenses around particles to form clouds. When droplets coalesce and become heavy, they fall as precipitation. Source: NOAA Weather Guide",
    },
    "Capital of France?": {
        "v1": "Paris",
        "v2": "The capital of France is Paris.",
        "v3": "The capital of France is Paris, which has been the country's capital since the 10th century. Source: Encyclopaedia Britannica",
    },
    "How does photosynthesis work?": {
        "v1": "Plants use sunlight.",
        "v2": "Plants convert sunlight, water, and CO2 into glucose and oxygen.",
        "v3": "Photosynthesis converts light energy into chemical energy: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. It occurs in chloroplasts via light-dependent reactions and the Calvin cycle. Source: Campbell Biology, 12th Ed.",
    },
}


def make_agent(version: str):
    @observe(op=Op.CHAT)
    def agent(question: str) -> str:
        enrich(model=MODELS[version], system=PROMPTS[version])
        return RESPONSES.get(question, {}).get(version, "I don't know.")

    return agent


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"input": "What is Python?", "expected": "programming language", "case_id": "python_basics", "case_name": "Python definition"},
    {"input": "What causes rain?", "expected": "water vapor condenses", "case_id": "rain_science", "case_name": "Rain explanation"},
    {"input": "Capital of France?", "expected": "Paris", "case_id": "france_capital", "case_name": "France capital"},
    {"input": "How does photosynthesis work?", "expected": "sunlight", "case_id": "photosynthesis", "case_name": "Photosynthesis"},
]

# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def correctness(input: str, output: str, expected: str) -> Score:
    keywords = [k.strip().lower() for k in expected.split(",")]
    matches = sum(1 for k in keywords if k in output.lower())
    value = matches / len(keywords) if keywords else 0.0
    return Score(name="correctness", value=value, label="pass" if value >= 0.5 else "fail")


def detail_score(input: str, output: str, expected: str) -> Score:
    wc = len(output.split())
    if wc >= 30:
        value = 1.0
    elif wc >= 15:
        value = 0.7
    elif wc >= 5:
        value = 0.4
    else:
        value = 0.1
    return Score(name="detail", value=value)


def has_source(input: str, output: str, expected: str) -> Score:
    has_cite = any(m in output.lower() for m in ["source:", "reference:", ".org", ".com"])
    return Score(name="has_source", value=1.0 if has_cite else 0.0, label="yes" if has_cite else "no")


# ---------------------------------------------------------------------------
# Run 3 versions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for version in ["v1", "v2", "v3"]:
        print(f"\n{'='*60}")
        print(f"  Running experiment for agent {version}")
        print(f"  Prompt: {PROMPTS[version]}")
        print(f"  Model:  {MODELS[version]}")
        print(f"{'='*60}\n")

        result = evaluate(
            name="rag-agent",
            task=make_agent(version),
            data=TEST_CASES,
            scores=[correctness, detail_score, has_source],
            metadata={
                "agent_version": version,
                "model": MODELS[version],
                "prompt": PROMPTS[version],
            },
            record_io=True,
        )

        # Small delay between runs so timestamps are distinct
        time.sleep(1)

    print(f"\n{'='*60}")
    print("  All 3 runs complete!")
    print("  Open http://localhost:4001/experiments to view results")
    print(f"{'='*60}")
