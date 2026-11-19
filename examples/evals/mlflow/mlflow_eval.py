# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""MLflow 3 judges on the travel assistant, uploaded to OpenSearch with Benchmark.

    python mlflow_eval.py

mlflow.genai.evaluate() runs the agent on each case of the shared test suite (predict_fn)
and grades the answers with MLflow's built-in LLM judges on Amazon Bedrock. Then one
Benchmark.log() per case uploads the verdicts and rationales, linked to the agent run.
"""

from __future__ import annotations

import os

import boto3
import mlflow
from agent import AGENT_NAME, AgentRun, load_test_suite, run_agent
from mlflow.genai.scorers import Correctness, RelevanceToQuery

from opensearch_genai_observability_sdk_py import Benchmark, EvalScore, register

JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
JUDGE = f"bedrock:/{JUDGE_MODEL_ID}"  # any MLflow judge provider, for example "openai:/gpt-4o"
SCORERS = {"correctness": "Correctness", "relevance_to_query": "RelevanceToQuery"}

# MLflow's bedrock:/ judge reads AWS keys from environment variables only, not from a
# profile. Export the keys of the current boto3 session (AWS_PROFILE, SSO, ...) for it.
if "AWS_ACCESS_KEY_ID" not in os.environ:
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token

# Strands emits its own chat spans; auto_instrument=False keeps an installed Bedrock
# (botocore) instrumentor from tracing each model call a second time.
provider = register(
    auto_instrument=False,
    service_name="travel-assistant-evals-mlflow-py",
    agent_name=AGENT_NAME,
    agent_version="v1",
)

cases = load_test_suite()
runs: dict[str, AgentRun] = {}  # question -> the agent run, for the trace ids


def predict_fn(question: str) -> str:
    runs[question] = run_agent(question)  # one trace per call
    return runs[question].answer


result = mlflow.genai.evaluate(
    data=[
        {"inputs": {"question": c["input"]}, "expectations": {"expected_response": c["expected"]}}
        for c in cases
    ],
    predict_fn=predict_fn,
    scorers=[Correctness(model=JUDGE), RelevanceToQuery(model=JUDGE)],
)
df = result.result_df

# MLflow records a failed judge call in the table instead of raising: do not upload it.
errors = {
    n: df[f"{n}/error_message"].dropna().tolist() for n in SCORERS if f"{n}/error_message" in df
}
if any(errors.values()):
    raise RuntimeError(f"MLflow judges failed: {errors}")


# The run metadata records the judge model on every result.
with Benchmark(
    "travel-assistant-suite-mlflow",
    run_name="mlflow judges",
    run_description=f"MLflow {', '.join(SCORERS.values())}",
    metadata={"gen_ai.request.model": JUDGE_MODEL_ID},
) as benchmark:
    for _, row in df.iterrows():
        # Match the row to its case by the expectation (each case has its own).
        case = next(c for c in cases if c["expected"] == row["expected_response/value"])
        run = runs[case["input"]]
        benchmark.log(
            case_id=case["case_id"],
            case_name=case["case_name"],
            input=case["input"],
            output=run.answer,
            expected=case["expected"],
            trace_id=run.trace_id,  # link the results to the agent run
            span_id=run.span_id,
            scores=[
                EvalScore(
                    name=name,
                    value=1.0 if row[f"{name}/value"] == "yes" else 0.0,
                    label="pass" if row[f"{name}/value"] == "yes" else "fail",  # yes / no
                    explanation=row.get(f"{name}/rationale"),
                    evaluator=f"mlflow.{cls}",
                    evaluator_type="llm_judge",
                )
                for name, cls in SCORERS.items()
            ],
        )

provider.force_flush()
