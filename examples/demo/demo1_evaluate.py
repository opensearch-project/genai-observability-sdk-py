# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Demo 1: Built-in eval runner — evaluate() runs your agent + scores it.

USE CASE:
    You're iterating on your agent during development. You change prompts,
    swap models, add tools — and want to measure the impact on output quality
    across a test suite. The SDK handles everything: running the agent,
    scoring outputs, emitting OTel spans with traces AND scores.

WHAT HAPPENS:
    1. Runs the same 5 test cases against two agent versions:
       - "basic" — short prompt, Claude 3 Haiku
       - "detailed" — thorough prompt, Claude 3 Haiku
    2. Custom Python scorers evaluate each output (no LLM judge needed)
    3. All traces + scores flow as OTel spans to agent-health
    4. Open the UI to compare the two runs side-by-side

RUN:
    python examples/demo/demo1_evaluate.py

THEN:
    Open http://localhost:4001/experiments → click "opensearch-qa"
    → select both runs → Compare
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import OTLPJsonExporter, TEST_CASES, make_agent

from opensearch_genai_observability_sdk_py import Score, evaluate, register

# ---------------------------------------------------------------------------
# Setup — send spans to agent-health on localhost:4001
# ---------------------------------------------------------------------------

register(
    exporter=OTLPJsonExporter(),
    service_name="opensearch-qa-agent",
    batch=False,  # Flush immediately for demo
)

# ---------------------------------------------------------------------------
# Scorers — pure Python functions, no LLM needed
# ---------------------------------------------------------------------------


def keyword_coverage(input: str, output: str, expected: str) -> Score:
    """What fraction of expected keywords appear in the output?"""
    expected_words = set(w.lower().strip(".,()") for w in expected.split() if len(w) > 3)
    output_lower = output.lower()
    matches = sum(1 for w in expected_words if w in output_lower)
    value = matches / len(expected_words) if expected_words else 0.0
    return Score(name="keyword_coverage", value=round(value, 2))


def completeness(input: str, output: str, expected: str) -> Score:
    """Is the answer thorough? Based on word count."""
    words = len(output.split())
    if words >= 80:
        value = 1.0
    elif words >= 40:
        value = 0.7
    elif words >= 15:
        value = 0.4
    else:
        value = 0.1
    return Score(name="completeness", value=value)


def has_specifics(input: str, output: str, expected: str) -> Score:
    """Does the answer mention specific tools, versions, or features?"""
    specifics = [
        "apache 2.0", "k-nn", "knn", "faiss", "nmslib", "lucene",
        "opentelemetry", "otel", "jaeger", "zipkin", "ldap", "saml",
        "rbac", "bedrock", "sagemaker", "ml commons", "7.10",
        "anomaly detection", "dashboards", "prometheus",
    ]
    found = sum(1 for s in specifics if s in output.lower())
    value = min(found / 3.0, 1.0)  # 3+ specifics = perfect score
    return Score(name="has_specifics", value=round(value, 2))


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scorers = [keyword_coverage, completeness, has_specifics]

    for version in ["basic", "detailed"]:
        print(f"\n{'='*60}")
        print(f"  Running: opensearch-qa / {version}")
        print(f"{'='*60}\n")

        result = evaluate(
            name="opensearch-qa",
            task=make_agent(prompt_version=version),
            data=TEST_CASES,
            scores=scorers,
            metadata={
                "agent_version": version,
                "model": os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0"),
                "prompt": version,
            },
            record_io=True,
        )

        print(f"\nPer-case results ({version}):")
        print(f"  {'Case':<28s} {'Keywords':<12s} {'Complete':<12s} {'Specifics':<12s}")
        print("  " + "-" * 64)
        for case in result.cases:
            print(
                f"  {(case.case_name or case.case_id):<28s} "
                f"{case.scores.get('keyword_coverage', 0):<12.0%} "
                f"{case.scores.get('completeness', 0):<12.0%} "
                f"{case.scores.get('has_specifics', 0):<12.0%}"
            )

        time.sleep(1)  # Distinct timestamps between runs

    print(f"""
{'='*60}
  Done! Open the UI to see results:

  1. http://localhost:4001/experiments
     → Click "opensearch-qa"
     → See runs for "basic" and "detailed" versions
     → Click a run to see donut chart + score metrics

  2. Select both runs → click "Compare"
     → See side-by-side delta badges (green = better, red = worse)

  3. Click "View Trace" on any test case
     → See the full agent execution: retrieve → LLM call
     → Inspect tokens, latency, model used
{'='*60}
""")
