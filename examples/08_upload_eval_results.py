# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Example: Upload evaluation results from external frameworks.

Most teams already have eval pipelines (RAGAS, DeepEval, pytest, custom).
This example shows how to upload those results to OpenSearch via the SDK,
so you can visualize and compare them in the agent-health UI.

Three patterns:
    A. Upload from a list of dicts (any eval framework)
    B. Upload from RAGAS results
    C. Upload with trace links (connect eval results to agent traces)

Usage:
    python examples/08_upload_eval_results.py

Prerequisites:
    pip install opensearch-genai-observability-sdk-py
"""

from __future__ import annotations

import os

from opensearch_genai_observability_sdk_py import Experiment, register

register(
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"),
    service_name="my-rag-agent",
)


# ---------------------------------------------------------------------------
# Pattern A: Upload from any eval framework (list of dicts)
# ---------------------------------------------------------------------------

def upload_custom_results():
    """Upload results from your own eval pipeline."""

    # Your eval results — could come from pytest, a notebook, CI, etc.
    results = [
        {
            "question": "What is Python?",
            "answer": "A high-level programming language",
            "ground_truth": "programming language",
            "accuracy": 1.0,
            "relevance": 0.95,
        },
        {
            "question": "Capital of France?",
            "answer": "Paris",
            "ground_truth": "Paris",
            "accuracy": 1.0,
            "relevance": 1.0,
        },
        {
            "question": "How does DNA replicate?",
            "answer": "Through cell division",
            "ground_truth": "semiconservative replication with helicase and polymerase",
            "accuracy": 0.3,
            "relevance": 0.5,
        },
    ]

    with Experiment(
        name="custom-eval",
        metadata={"framework": "custom", "model": "gpt-4o-mini"},
        record_io=True,
    ) as exp:
        for r in results:
            exp.log(
                input=r["question"],
                output=r["answer"],
                expected=r["ground_truth"],
                scores={
                    "accuracy": r["accuracy"],
                    "relevance": r["relevance"],
                },
            )


# ---------------------------------------------------------------------------
# Pattern B: Upload from RAGAS (commented out — requires ragas package)
# ---------------------------------------------------------------------------

def upload_ragas_results():
    """Upload results from a RAGAS evaluation run.

    Uncomment and install ragas to use:
        pip install ragas
    """
    # from ragas import evaluate as ragas_evaluate
    # from ragas.metrics import faithfulness, answer_relevancy
    #
    # # Run RAGAS evaluation
    # ragas_result = ragas_evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy],
    # )
    # df = ragas_result.to_pandas()
    #
    # # Upload to OpenSearch
    # with Experiment(
    #     name="ragas-eval",
    #     metadata={"framework": "ragas", "model": "gpt-4o"},
    #     record_io=True,
    # ) as exp:
    #     for _, row in df.iterrows():
    #         exp.log(
    #             input=row["question"],
    #             output=row["answer"],
    #             expected=row.get("ground_truth"),
    #             scores={
    #                 "faithfulness": row["faithfulness"],
    #                 "answer_relevancy": row["answer_relevancy"],
    #             },
    #         )
    pass


# ---------------------------------------------------------------------------
# Pattern C: Upload with trace links
# ---------------------------------------------------------------------------

def upload_with_trace_links():
    """Upload eval results linked to existing agent traces.

    When your eval pipeline captures trace IDs from the agent run,
    you can link eval results back to the original traces. This lets
    you click through from a failed test case to the full agent trace
    waterfall in the UI.
    """

    # Imagine your CI pipeline ran the agent and captured trace IDs
    ci_results = [
        {
            "question": "Why is checkout slow?",
            "answer": "Payment service has high latency due to connection pool exhaustion",
            "accuracy": 1.0,
            "trace_id": "6ebb9835f43af1552f2cebb9f5165e39",
            "span_id": "89829115c2128845",
        },
        {
            "question": "Why are orders failing?",
            "answer": "Database connection timeout in order-service",
            "accuracy": 0.8,
            "trace_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "span_id": "1234567890abcdef",
        },
    ]

    with Experiment(
        name="ci-nightly",
        metadata={"framework": "pytest", "commit": "abc123", "branch": "main"},
        record_io=True,
    ) as exp:
        for r in ci_results:
            exp.log(
                input=r["question"],
                output=r["answer"],
                scores={"accuracy": r["accuracy"]},
                trace_id=r["trace_id"],  # Links to the original agent trace
                span_id=r["span_id"],    # Links to the specific agent span
                case_id=r["question"][:20].replace(" ", "_").lower(),
            )


# ---------------------------------------------------------------------------
# Run all patterns
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pattern A: Upload custom eval results")
    upload_custom_results()
    print()

    print("Pattern B: Upload RAGAS results (skipped — requires ragas)")
    upload_ragas_results()
    print()

    print("Pattern C: Upload with trace links")
    upload_with_trace_links()
    print()

    print("""
All results uploaded! Open agent-health UI to explore:
  - Experiments tab: see "custom-eval" and "ci-nightly" runs
  - Click a run to see scores per test case
  - For ci-nightly: click "View trace" to see the original agent trace
""")
