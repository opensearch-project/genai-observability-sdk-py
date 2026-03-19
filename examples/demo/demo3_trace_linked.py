# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Demo 3: Trace-linked experiments — connect eval scores to agent traces.

USE CASE:
    Your agent runs in production or CI. Traces are already flowing to
    OpenSearch. Now you want to evaluate those outputs and link the scores
    back to the original execution traces.

    This is the key differentiator: click "View Trace" on a failed test case
    and land directly on the agent execution waterfall — see exactly which
    LLM call or tool invocation caused the bad output.

WHAT HAPPENS:
    1. Runs the agent on test cases, capturing the trace ID from each run
    2. Scores the outputs (simple scorers — could also be DeepEval)
    3. Uploads results via Experiment.log(trace_id=...) to create OTel span links
    4. In the UI, each test case has a "View Trace" button that navigates
       to the original agent execution

RUN:
    python examples/demo/demo3_trace_linked.py

THEN:
    Open http://localhost:4001/experiments → click "opensearch-qa-traced"
    → Click "View Trace" on any test case
    → See the full agent execution waterfall (retrieve → LLM call)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import OTLPJsonExporter, TEST_CASES, make_agent

from opentelemetry import trace

from opensearch_genai_observability_sdk_py import Experiment, Score, register

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

register(
    exporter=OTLPJsonExporter(),
    service_name="opensearch-qa-agent",
    batch=False,
)


# ---------------------------------------------------------------------------
# Scorers (same simple ones — could also use DeepEval here)
# ---------------------------------------------------------------------------


def keyword_coverage(output: str, expected: str) -> float:
    expected_words = set(w.lower().strip(".,()") for w in expected.split() if len(w) > 3)
    output_lower = output.lower()
    matches = sum(1 for w in expected_words if w in output_lower)
    return round(matches / len(expected_words), 2) if expected_words else 0.0


def completeness(output: str) -> float:
    words = len(output.split())
    if words >= 80:
        return 1.0
    if words >= 40:
        return 0.7
    if words >= 15:
        return 0.4
    return 0.1


# ---------------------------------------------------------------------------
# Step 1: Run the agent and capture trace IDs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = make_agent(prompt_version="detailed")
    tracer = trace.get_tracer("demo-runner")

    print("Step 1: Running agent and capturing trace IDs...\n")

    results = []
    for tc in TEST_CASES:
        question = tc["input"]

        # Create a parent span so we can capture the trace ID
        # The agent's @observe spans become children of this span
        with tracer.start_as_current_span(f"production_run {tc['case_name']}") as span:
            output = agent(question)
            ctx = span.get_span_context()
            trace_id = format(ctx.trace_id, '032x')
            span_id = format(ctx.span_id, '016x')

        results.append({
            "case": tc,
            "output": output,
            "trace_id": trace_id,
            "span_id": span_id,
        })
        print(f"  [{tc['case_name']}] trace_id={trace_id[:16]}... output={output[:60]}...")

    # ---------------------------------------------------------------------------
    # Step 2: Score the outputs
    # ---------------------------------------------------------------------------

    print("\nStep 2: Scoring outputs...\n")

    for r in results:
        expected = r["case"]["expected"]
        r["scores"] = {
            "keyword_coverage": keyword_coverage(r["output"], expected),
            "completeness": completeness(r["output"]),
        }
        print(
            f"  [{r['case']['case_name']}] "
            f"keyword_coverage={r['scores']['keyword_coverage']:.0%}  "
            f"completeness={r['scores']['completeness']:.0%}"
        )

    # ---------------------------------------------------------------------------
    # Step 3: Upload with trace links
    # ---------------------------------------------------------------------------

    print("\nStep 3: Uploading results with trace links...\n")

    with Experiment(
        name="opensearch-qa-traced",
        metadata={
            "agent_prompt": "detailed",
            "model": os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0"),
            "source": "ci-pipeline",
        },
        record_io=True,
    ) as exp:
        for r in results:
            exp.log(
                input=r["case"]["input"],
                output=r["output"],
                expected=r["case"]["expected"],
                scores=r["scores"],
                case_id=r["case"]["case_id"],
                case_name=r["case"]["case_name"],
                trace_id=r["trace_id"],   # Links to the agent execution trace
                span_id=r["span_id"],     # Links to the specific parent span
            )

    print(f"""
{'='*60}
  Done! Results uploaded with trace links.

  Open http://localhost:4001/experiments
  → Click "opensearch-qa-traced"
  → Click "View Trace" on any test case
  → See the full agent execution waterfall:
      production_run
       └── invoke_agent opensearch-qa-agent
            ├── retrieval retrieve
            └── chat call_bedrock

  This is the key feature: eval scores linked to production traces.
  A failed score → one click → see exactly what went wrong.
{'='*60}
""")
