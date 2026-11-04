# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Scoring with opensearch-genai-observability-sdk-py.

Each score() call emits one OTel log-based event named
``gen_ai.evaluation.result`` (OTel GenAI semantic conventions). The event
carries the trace context of the evaluated span and is exported over OTLP
logs (``/v1/logs``) on the same endpoint and auth as traces. Data Prepper
routes these events to the ``otel-v1-genai-evaluation`` index.

Levels: span (explicit ids or the active span), trace (trace id only),
session (conversation id).
"""

from opensearch_genai_observability_sdk_py import Op, observe, register, score

# --- Setup: traces to /v1/traces, evaluation events to /v1/logs ---
provider = register(endpoint="http://localhost:21890/opentelemetry")


# --- Inline: score the active span while it is running ---
@observe(op=Op.CHAT)
def answer(question: str) -> str:
    response = "Paris is the capital of France."
    # The event is parented to this chat span automatically.
    score(
        name="relevance",
        value=0.95,
        label="relevant",
        explanation="Answer directly addresses the question with correct facts",
        evaluator="relevance_judge",
        evaluator_version="1.0.0",
        evaluator_type="llm_judge",
        score_min=0.0,
        score_max=1.0,
        direction="maximize",
    )
    return response


answer("What is the capital of France?")


# --- Offline / async: score a stored span by its ids ---
score(
    name="faithfulness",
    value=0.4,
    label="fail",
    trace_id="0af7651916cd43dd8448eb211c80319c",
    span_id="b7ad6b7169203331",
    evaluator="faithfulness_judge",
    evaluator_type="llm_judge",
    judge_trace_id="4bf92f3577b34da6a3ce929d0e0e4736",  # the judge's own trace, if traced
)


# --- Trace-level: only a trace id is known. No span id is invented; the
# event carries the trace id as its own traceId (no spanId) and level=trace. ---
score(
    name="quality",
    value=0.88,
    label="good",
    trace_id="0af7651916cd43dd8448eb211c80319c",
    evaluator_type="human",
    explanation="Reviewed by QA team, response is accurate and well-formatted",
)


# --- Session-level: score a whole conversation ---
score(
    name="user_satisfaction",
    value=1.0,
    label="thumbs_up",
    conversation_id="session-456",
    level="session",
    evaluator_type="human",
)


# --- Evaluator failure: record error.type, no score ---
score(name="toxicity", error_type="timeout", evaluator="toxicity_classifier")


# --- Extra attributes (for example test.* grouping keys) pass through as-is ---
score(
    name="latency_check",
    value=1.0,
    trace_id="0af7651916cd43dd8448eb211c80319c",
    span_id="b7ad6b7169203331",
    evaluator_type="code",
    attributes={"test.suite.run.id": "run_001", "test.case.id": "case_001"},
)

# Flush spans and evaluation events before exit.
provider.force_flush()

# Each score() call emits a LogRecord like:
#
#   event_name     = "gen_ai.evaluation.result"
#   trace_id/span_id = the evaluated span
#   scope          = opensearch-genai-observability-sdk-py <SDK version>
#   attributes:
#     gen_ai.evaluation.name                 = "relevance"
#     gen_ai.evaluation.score.value          = 0.95
#     gen_ai.evaluation.score.label          = "relevant"
#     gen_ai.evaluation.explanation          = "Answer directly addresses..."
#     opensearch.gen_ai.evaluation.level            = "span"
#     opensearch.gen_ai.evaluation.evaluator.name   = "relevance_judge"
#     opensearch.gen_ai.evaluation.evaluator.type   = "llm_judge"
#     opensearch.gen_ai.evaluation.score.min / .max = 0.0 / 1.0
#     opensearch.gen_ai.evaluation.score.direction  = "maximize"
