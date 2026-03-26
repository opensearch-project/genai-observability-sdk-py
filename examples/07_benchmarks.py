# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Example: Run benchmarks to compare agent versions.

Run the same test cases across different prompt versions, with full traces
and scores stored in OpenSearch for comparison.

Usage:
    AGENT_VERSION=v1 python examples/07_benchmarks.py
    AGENT_VERSION=v2 python examples/07_benchmarks.py
    # Open agent-health UI → Benchmarks → select runs → Compare

Prerequisites:
    pip install opensearch-genai-observability-sdk-py
"""

from __future__ import annotations

import os

from opensearch_genai_observability_sdk_py import (
    EvalScore,
    Op,
    enrich,
    evaluate,
    observe,
    register,
)

register(
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"),
    service_name="my-rag-agent",
)

AGENT_VERSION = os.environ.get("AGENT_VERSION", "v1")

PROMPTS = {
    "v1": "Answer briefly.",
    "v2": "Answer accurately. Include sources when possible.",
}


@observe(op=Op.CHAT)
def my_agent(question: str) -> str:
    """Your agent — instrumented with @observe for tracing."""
    enrich(model="gpt-4o-mini", system=PROMPTS.get(AGENT_VERSION, PROMPTS["v1"]))
    return _simulate(question, AGENT_VERSION)


def _simulate(question: str, version: str) -> str:
    responses = {
        "What is Python?": {
            "v1": "A programming language.",
            "v2": "Python is a high-level language for web dev, data science, and automation. Source: python.org",
        },
        "What causes rain?": {
            "v1": "Water falls from clouds.",
            "v2": "Water vapor condenses into droplets that fall as precipitation. Source: NOAA",
        },
    }
    return responses.get(question, {}).get(version, "I don't know.")


TEST_CASES = [
    {"input": "What is Python?", "expected": "programming language", "case_id": "python"},
    {"input": "What causes rain?", "expected": "water vapor condenses", "case_id": "rain"},
]


def correctness(input: str, output: str, expected: str) -> EvalScore:
    keywords = [k.strip().lower() for k in expected.split(",")]
    matches = sum(1 for k in keywords if k in output.lower())
    return EvalScore(name="correctness", value=matches / len(keywords) if keywords else 0.0)


def has_source(input: str, output: str, expected: str) -> EvalScore:
    has_citation = any(m in output.lower() for m in ["source:", ".org", ".com"])
    return EvalScore(name="has_source", value=1.0 if has_citation else 0.0)


if __name__ == "__main__":
    print(f"\nRunning benchmark: agent {AGENT_VERSION}\n")

    result = evaluate(
        name="rag-agent",
        task=my_agent,
        data=TEST_CASES,
        scores=[correctness, has_source],
        metadata={"agent_version": AGENT_VERSION},
        record_io=True,
    )

    for case in result.cases:
        print(
            f"  {case.case_id:<10s} {case.status:<6s} "
            f"correctness={case.scores.get('correctness', 0):.2f} "
            f"has_source={case.scores.get('has_source', 0):.2f}"
        )
