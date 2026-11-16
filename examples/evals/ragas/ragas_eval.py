# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""RAGAS metrics on the travel assistant, uploaded to OpenSearch with Benchmark.

    python ragas_eval.py

1. Run the agent on every case of the shared test suite (one trace per case).
2. Score all answers at once with ragas.evaluate() (judge: Claude on Amazon Bedrock).
3. Upload one Benchmark.log() per case, linked to the case's agent run.
"""

from __future__ import annotations

import math
import os

from agent import AGENT_NAME, REGION, load_test_suite, run_agent
from datasets import Dataset
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness

from opensearch_genai_observability_sdk_py import Benchmark, EvalScore, register

JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBEDDINGS_MODEL_ID = os.environ.get("EMBEDDINGS_MODEL_ID", "amazon.titan-embed-text-v2:0")
THRESHOLDS = {"faithfulness": 0.7, "answer_relevancy": 0.6, "answer_correctness": 0.5}

provider = register(
    service_name="travel-assistant-evals-ragas", agent_name=AGENT_NAME, agent_version="v1"
)

# 1. Run the agent. Each run keeps its trace and span ids to link the results.
cases = load_test_suite()
runs = [run_agent(case["input"]) for case in cases]

# 2. RAGAS uses OpenAI by default; pass llm= and embeddings= for Bedrock.
df = ragas_evaluate(
    dataset=Dataset.from_dict(
        {
            "question": [case["input"] for case in cases],
            "answer": [run.answer for run in runs],
            "contexts": [run.tool_results or ["No tool was called."] for run in runs],
            "ground_truth": [case["expected"] for case in cases],
        }
    ),
    metrics=[faithfulness, answer_relevancy, answer_correctness],
    llm=LangchainLLMWrapper(ChatBedrockConverse(model=JUDGE_MODEL_ID, region_name=REGION)),
    embeddings=LangchainEmbeddingsWrapper(
        BedrockEmbeddings(model_id=EMBEDDINGS_MODEL_ID, region_name=REGION)
    ),
    raise_exceptions=True,  # by default a failed judge call becomes NaN
).to_pandas()

# 3. Upload. The run metadata records the judge model on every result.
with Benchmark(
    "travel-assistant-suite-ragas",
    run_name="ragas judges",
    run_description=f"RAGAS {', '.join(THRESHOLDS)}",
    metadata={"gen_ai.request.model": JUDGE_MODEL_ID},
) as benchmark:
    for case, run, (_, row) in zip(cases, runs, df.iterrows(), strict=True):
        scores = []
        for metric, threshold in THRESHOLDS.items():
            value = float(row[metric])
            if math.isnan(value):  # a metric RAGAS could not compute
                continue
            scores.append(
                EvalScore(
                    name=metric,
                    value=value,
                    threshold=threshold,
                    score_min=0,
                    score_max=1,
                    evaluator=f"ragas.{metric}",
                    evaluator_type="llm_judge",
                )
            )
        benchmark.log(
            case_id=case["case_id"],
            case_name=case["case_name"],
            input=case["input"],
            output=run.answer,
            expected=case["expected"],
            scores=scores,
            trace_id=run.trace_id,  # link the results to the agent run
            span_id=run.span_id,
        )

provider.force_flush()
