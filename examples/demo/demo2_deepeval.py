# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Demo 2: External eval framework — DeepEval scores → Experiment.log().

USE CASE:
    Your team already uses DeepEval (or RAGAS, or pytest) for LLM evaluation.
    You don't want to rewrite your eval pipeline. You just want to see those
    scores in OpenSearch, alongside your agent traces, in the same UI where
    you debug production issues.

    The SDK is a bridge: scores from any framework become OTel spans in
    OpenSearch, viewable and comparable in the agent-health UI.

WHAT HAPPENS:
    1. Runs the agent on 5 test cases (with @observe tracing)
    2. Scores each output using DeepEval's LLM-as-judge metrics:
       - AnswerRelevancyMetric — is the answer relevant to the question?
       - GEval (correctness) — custom LLM judge criteria
    3. Uploads all results via Experiment.log() with scores
    4. Open the UI to see DeepEval scores visualized

SETUP:
    pip install deepeval

RUN:
    python examples/demo/demo2_deepeval.py

THEN:
    Open http://localhost:4001/experiments → click "opensearch-qa-deepeval"
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import AWS_REGION, BEDROCK_MODEL, OTLPJsonExporter, TEST_CASES, make_agent

import boto3
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from opensearch_genai_observability_sdk_py import Experiment, register

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

register(
    exporter=OTLPJsonExporter(),
    service_name="opensearch-qa-agent",
    batch=False,
)


# ---------------------------------------------------------------------------
# Bedrock model adapter for DeepEval
# ---------------------------------------------------------------------------


class BedrockJudge(DeepEvalBaseLLM):
    """Wraps AWS Bedrock Claude as a DeepEval judge model."""

    def __init__(self, model_id: str = BEDROCK_MODEL, region: str = AWS_REGION):
        self.model_id = model_id
        self.region = region
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def load_model(self):
        return self.model_id

    def generate(self, prompt: str, schema=None) -> str:
        system = ""
        if schema:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            system = (
                "You must respond with valid JSON only. No markdown, no explanation. "
                f"Your response must conform to this JSON schema:\n{schema_json}"
            )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        if system:
            body["system"] = system

        response = self._client.invoke_model(
            modelId=self.model_id, body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        if schema:
            # Extract JSON from response and parse into schema
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return schema.model_validate_json(json_match.group())
            return schema.model_validate_json(text)

        return text

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_id


# ---------------------------------------------------------------------------
# DeepEval metrics
# ---------------------------------------------------------------------------

judge = BedrockJudge()

answer_relevancy = AnswerRelevancyMetric(model=judge, threshold=0.5)

correctness = GEval(
    name="correctness",
    model=judge,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "Check if the actual output covers the key points in the expected output.",
        "Verify factual accuracy of the claims made.",
        "Penalize if important information from the expected output is missing.",
    ],
    threshold=0.5,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = make_agent(prompt_version="detailed")

    print("Running agent on test cases and scoring with DeepEval...\n")

    with Experiment(
        name="opensearch-qa-deepeval",
        metadata={
            "framework": "deepeval",
            "judge_model": BEDROCK_MODEL,
            "agent_prompt": "detailed",
            "model": BEDROCK_MODEL,
        },
        record_io=True,
    ) as exp:
        for tc in TEST_CASES:
            question = tc["input"]
            expected = tc["expected"]
            case_name = tc["case_name"]
            case_id = tc["case_id"]

            print(f"  [{case_name}] Running agent...")
            output = agent(question)

            print(f"  [{case_name}] Scoring with DeepEval...")
            deepeval_case = LLMTestCase(
                input=question,
                actual_output=output,
                expected_output=expected,
                retrieval_context=[],  # Could add retrieved docs here
            )

            scores = {}

            # Answer relevancy
            try:
                answer_relevancy.measure(deepeval_case)
                scores["answer_relevancy"] = answer_relevancy.score
                print(f"    answer_relevancy: {answer_relevancy.score:.2f} — {answer_relevancy.reason[:80]}")
            except Exception as e:
                print(f"    answer_relevancy: ERROR — {e}")

            # Correctness (GEval)
            try:
                correctness.measure(deepeval_case)
                scores["correctness"] = correctness.score
                print(f"    correctness:      {correctness.score:.2f} — {correctness.reason[:80]}")
            except Exception as e:
                print(f"    correctness: ERROR — {e}")

            # Upload to OpenSearch via Experiment.log()
            exp.log(
                input=question,
                output=output,
                expected=expected,
                scores=scores,
                case_id=case_id,
                case_name=case_name,
            )
            print()

    print(f"""
{'='*60}
  Done! DeepEval scores uploaded to OpenSearch.

  Open http://localhost:4001/experiments
  → Click "opensearch-qa-deepeval"
  → See LLM-as-judge scores (answer_relevancy, correctness)
  → Compare with Demo 1's custom scorer results
{'='*60}
""")
