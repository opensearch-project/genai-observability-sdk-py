# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Example: Run experiments to compare agent versions.

This example shows how an agent builder iterates on their agent during
development — running the same test cases across different prompt/model
versions, with full traces and scores stored in OpenSearch for comparison.

Workflow:
    1. Define your agent (instrumented with @observe)
    2. Define your test dataset
    3. Define your scorers
    4. Run evaluate() — traces + scores flow to OpenSearch
    5. Change your agent (prompt, model, tools, etc.)
    6. Run evaluate() again with a new experiment name
    7. Open agent-health UI to compare runs side-by-side

Usage:
    # First run — baseline with simple prompt
    AGENT_VERSION=v1 python examples/07_experiments.py

    # Second run — improved prompt
    AGENT_VERSION=v2 python examples/07_experiments.py

    # Third run — different model
    AGENT_VERSION=v3 python examples/07_experiments.py

    # Open agent-health UI → Experiments → select runs → Compare

Prerequisites:
    pip install opensearch-genai-observability-sdk-py openai

    # Local observability stack (OTel Collector → OpenSearch):
    #   OTel Collector on http://localhost:4318
    #   OpenSearch on http://localhost:9200
    #   Agent-health UI on http://localhost:4001
"""

from __future__ import annotations

import os

from opensearch_genai_observability_sdk_py import (
    Op,
    Score,
    enrich,
    evaluate,
    observe,
    register,
)

# ---------------------------------------------------------------------------
# 1. Setup — one line to connect to your local observability stack
# ---------------------------------------------------------------------------

register(
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"),
    service_name="my-rag-agent",
)

AGENT_VERSION = os.environ.get("AGENT_VERSION", "v1")

# ---------------------------------------------------------------------------
# 2. Your agent — changes between versions
# ---------------------------------------------------------------------------

# Simulate different agent versions via prompt/model changes
PROMPTS = {
    "v1": "Answer the question briefly.",
    "v2": "Answer the question accurately. Include sources when possible.",
    "v3": "You are an expert assistant. Answer precisely and cite sources.",
}

MODELS = {
    "v1": "gpt-4o-mini",
    "v2": "gpt-4o-mini",
    "v3": "gpt-4.1",
}


@observe(op=Op.CHAT)
def my_agent(question: str) -> str:
    """Your RAG agent — instrumented with @observe for full tracing."""
    prompt = PROMPTS.get(AGENT_VERSION, PROMPTS["v1"])
    model = MODELS.get(AGENT_VERSION, MODELS["v1"])

    # Enrich the span with model info (visible in traces)
    enrich(model=model, system=prompt)

    # --- In real code, this would be your actual agent logic ---
    # docs = retrieve(question)
    # response = call_llm(model, prompt, question, docs)
    # return response

    # Simulated responses for the example
    return _simulate_response(question, AGENT_VERSION)


@observe(op=Op.EXECUTE_TOOL)
def retrieve(query: str) -> list[str]:
    """Simulated retrieval tool — also traced."""
    enrich(tool_name="vector_search")
    return [f"Document about {query}"]


def _simulate_response(question: str, version: str) -> str:
    """Simulate different quality responses per agent version."""
    responses = {
        "What is Python?": {
            "v1": "A programming language.",
            "v2": "Python is a high-level programming language used for web development, data science, and automation.",
            "v3": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It is widely used for web development (Django, Flask), data science (pandas, NumPy), and automation. Source: python.org",
        },
        "What causes rain?": {
            "v1": "Water falls from clouds.",
            "v2": "Rain occurs when water vapor in the atmosphere condenses into droplets that become heavy enough to fall.",
            "v3": "Rain forms through the water cycle: evaporation from surfaces creates water vapor, which rises and cools in the atmosphere, condensing around particles to form clouds. When droplets coalesce and become heavy enough, they fall as precipitation. Source: NOAA Weather Guide",
        },
        "Capital of France?": {
            "v1": "Paris",
            "v2": "The capital of France is Paris.",
            "v3": "The capital of France is Paris, which has been the country's capital since the 10th century. Source: Encyclopaedia Britannica",
        },
        "How does photosynthesis work?": {
            "v1": "Plants use sunlight.",
            "v2": "Plants convert sunlight, water, and CO2 into glucose and oxygen through photosynthesis.",
            "v3": "Photosynthesis converts light energy into chemical energy: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. It occurs in chloroplasts via light-dependent reactions (thylakoids) and the Calvin cycle (stroma). Source: Campbell Biology, 12th Ed.",
        },
    }
    return responses.get(question, {}).get(version, "I don't know.")


# ---------------------------------------------------------------------------
# 3. Test dataset — same across all runs for fair comparison
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "input": "What is Python?",
        "expected": "programming language",
        "case_id": "python_basics",
        "case_name": "Python definition",
    },
    {
        "input": "What causes rain?",
        "expected": "water vapor condenses",
        "case_id": "rain_science",
        "case_name": "Rain explanation",
    },
    {
        "input": "Capital of France?",
        "expected": "Paris",
        "case_id": "france_capital",
        "case_name": "France capital",
    },
    {
        "input": "How does photosynthesis work?",
        "expected": "sunlight",
        "case_id": "photosynthesis",
        "case_name": "Photosynthesis explanation",
    },
]

# ---------------------------------------------------------------------------
# 4. Scorers — evaluate output quality
# ---------------------------------------------------------------------------


def correctness(input: str, output: str, expected: str) -> Score:
    """Check if expected keywords appear in the output."""
    keywords = [k.strip().lower() for k in expected.split(",")]
    matches = sum(1 for k in keywords if k in output.lower())
    value = matches / len(keywords) if keywords else 0.0
    return Score(
        name="correctness",
        value=value,
        label="pass" if value >= 0.5 else "fail",
    )


def detail_score(input: str, output: str, expected: str) -> Score:
    """Score response detail/thoroughness by word count."""
    word_count = len(output.split())
    if word_count >= 30:
        value = 1.0
    elif word_count >= 15:
        value = 0.7
    elif word_count >= 5:
        value = 0.4
    else:
        value = 0.1
    return Score(name="detail", value=value)


def has_source(input: str, output: str, expected: str) -> Score:
    """Check if response cites a source."""
    has_citation = any(
        marker in output.lower()
        for marker in ["source:", "reference:", "according to", ".org", ".com"]
    )
    return Score(
        name="has_source",
        value=1.0 if has_citation else 0.0,
        label="yes" if has_citation else "no",
    )


# ---------------------------------------------------------------------------
# 5. Run the experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nRunning experiment for agent {AGENT_VERSION}...")
    print(f"  Prompt: {PROMPTS[AGENT_VERSION]}")
    print(f"  Model:  {MODELS[AGENT_VERSION]}")
    print()

    result = evaluate(
        name="rag-agent",  # Same name across runs for comparison
        task=my_agent,
        data=TEST_CASES,
        scores=[correctness, detail_score, has_source],
        metadata={
            "agent_version": AGENT_VERSION,
            "model": MODELS[AGENT_VERSION],
            "prompt": PROMPTS[AGENT_VERSION],
        },
        record_io=True,  # Store input/output for debugging in UI
    )

    # Print per-case results
    print("\nPer-case results:")
    print(f"  {'Case':<25s} {'Status':<8s} {'Correct':<10s} {'Detail':<10s} {'Source':<10s}")
    print("  " + "-" * 63)
    for case in result.cases:
        print(
            f"  {case.case_name or case.case_id:<25s} "
            f"{case.status:<8s} "
            f"{case.scores.get('correctness', 0):<10.2f} "
            f"{case.scores.get('detail', 0):<10.2f} "
            f"{case.scores.get('has_source', 0):<10.2f}"
        )

    print(f"""
What to do next:
  1. Run again with a different version:
       AGENT_VERSION=v2 python examples/07_experiments.py

  2. Open agent-health UI:
       http://localhost:4001

  3. Go to Experiments tab:
       - See all runs of "rag-agent"
       - Click a run to see score summary + individual cases
       - Select two runs → Compare to see deltas

  4. Go to Traces tab:
       - Each case has a full trace waterfall
       - See model, tokens, latency, tool calls per case
""")
