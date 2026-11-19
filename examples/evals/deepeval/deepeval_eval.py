# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""DeepEval metrics on the travel assistant, results sent to OpenSearch.

    python deepeval_eval.py

evaluate() runs the shared test suite through the agent; each scorer runs one DeepEval
metric (judge: Claude on Amazon Bedrock) on the agent's answer and returns its score and
reason. Each result names its evaluator (deepeval.<Metric>) and the judge model, and is
linked to the agent run's root span. A metric that raises becomes an evaluator error result.
"""

from __future__ import annotations

import os

from agent import AGENT_NAME, REGION, AgentRun, load_test_suite, run_agent
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.models import AmazonBedrockModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from opensearch_genai_observability_sdk_py import EvalScore, evaluate, register, scorer

JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
THRESHOLD = 0.7

# Strands emits its own chat spans; auto_instrument=False keeps an installed Bedrock
# (botocore) instrumentor from tracing each model call a second time.
provider = register(
    auto_instrument=False,
    service_name="travel-assistant-evals-deepeval-py",
    agent_name=AGENT_NAME,
    agent_version="v1",
)

judge = AmazonBedrockModel(model=JUDGE_MODEL_ID, region=REGION)  # or any DeepEval model
metrics = {
    "answer_relevancy": AnswerRelevancyMetric(model=judge, threshold=THRESHOLD),
    "faithfulness": FaithfulnessMetric(model=judge, threshold=THRESHOLD),
    "correctness": GEval(
        name="Correctness",
        criteria=(
            "The actual output meets the expected output. Weather and events must come from "
            "the context (the tool results); extra details from the context are fine."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        model=judge,
        threshold=THRESHOLD,
    ),
}


def deepeval_scorer(name: str):
    """A scorer that runs one DeepEval metric and returns its score and reason."""
    metric = metrics[name]

    @scorer(
        name=name,
        threshold=THRESHOLD,
        score_min=0,
        score_max=1,
        evaluator=f"deepeval.{type(metric).__name__}",
        evaluator_type="llm_judge",
        judge_model=JUDGE_MODEL_ID,
    )
    def run_metric(input: str, output: AgentRun, expected: str) -> EvalScore:
        tool_results = output.tool_results or ["No tool was called."]
        metric.measure(
            LLMTestCase(
                input=input,
                actual_output=output.answer,
                expected_output=expected,
                context=tool_results,  # what the answer must be grounded in
                retrieval_context=tool_results,  # read by FaithfulnessMetric
            )
        )
        return EvalScore(name=name, value=metric.score, explanation=metric.reason)

    return run_metric


result = evaluate(
    name="travel-assistant-suite-deepeval",
    task=run_agent,  # returns AgentRun: the answer and the tool results
    data=load_test_suite(),
    scores=[deepeval_scorer(name) for name in metrics],
    run_name="deepeval judges",
    run_description=f"DeepEval {', '.join(metrics)}; judge {JUDGE_MODEL_ID}",
)
for case in result.cases:
    print(f"  {case.case_id:<14} {case.status:<5} {case.scores}  trace {case.trace_id}")

provider.force_flush()
