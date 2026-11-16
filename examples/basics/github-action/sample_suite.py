# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Sample evaluation suite for the GitHub Action (mock agent, no LLM calls).

Point the action at your own suite the same way: a function with no
arguments that calls ``evaluate()`` and returns its ``BenchmarkResult``.
``run_suite.py`` calls ``register()`` before importing this module, so
``@observe`` spans and evaluation events go to the configured endpoints.
"""

from __future__ import annotations

import os

from opensearch_genai_observability_sdk_py import EvalScore, Op, evaluate, observe

ANSWERS = {
    "What is OpenSearch?": "OpenSearch is an open source search and analytics suite.",
    "What does OTLP stand for?": "OpenTelemetry Protocol.",
    "Capital of France?": "Paris is the capital of France.",
    "Who wrote Hamlet?": "I am not sure.",
}

DATA = [
    {"case_id": "opensearch", "input": "What is OpenSearch?", "expected": "search"},
    {"case_id": "otlp", "input": "What does OTLP stand for?", "expected": "OpenTelemetry"},
    {"case_id": "france", "input": "Capital of France?", "expected": "Paris"},
    {"case_id": "hamlet", "input": "Who wrote Hamlet?", "expected": "Shakespeare"},
]


@observe(name="mock_agent", op=Op.INVOKE_AGENT)
def mock_agent(question: str) -> str:
    """Deterministic stand-in for a real agent."""
    return ANSWERS.get(question, "I don't know.")


def contains_expected(input, output, expected) -> EvalScore:
    hit = expected.lower() in output.lower()
    return EvalScore(
        name="contains_expected",
        value=1.0 if hit else 0.0,
        label="pass" if hit else "fail",
        explanation=f"expected {expected!r} {'found' if hit else 'missing'} in output",
    )


def answer_length(input, output, expected) -> float:
    """Toy quality score in [0, 1]: longer answers up to 40 characters."""
    return min(len(output) / 40.0, 1.0)


def suite():
    return evaluate(
        name=os.environ.get("EVAL_SUITE_NAME", "sample-agent-suite"),
        task=mock_agent,
        data=DATA,
        scores=[contains_expected, answer_length],
        metadata={
            "vcs.ref.head.revision": os.environ.get("GITHUB_SHA", "local"),
            "vcs.ref.head.name": os.environ.get("GITHUB_REF_NAME", "local"),
        },
    )
