# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Offline evaluation: run the shared test suite twice, once per prompt version.

    python offline_eval.py

Each run is one evaluate() call with its own run_name, so the two runs can be compared
in OpenSearch Dashboards (Evaluations > the suite > Compare). Every case is its own agent
trace; each scorer result is one gen_ai.evaluation.result linked to the agent's root span.
"""

from __future__ import annotations

from agent import AGENT_NAME, PROMPTS, load_test_suite, run_agent
from scorers import (
    JUDGE_DESCRIPTION,
    JUDGE_MODEL_ID,
    JUDGE_PROMPT_NAME,
    JUDGE_PROMPT_VERSION,
    judge_plan,
    uses_tool_data,
)

from opensearch_genai_observability_sdk_py import EvalScore, evaluate, register, scorer

SUITE = "travel-assistant-suite"

# Strands and the traced judge emit their own chat spans; auto_instrument=False keeps an
# installed Bedrock (botocore) instrumentor from tracing each model call a second time.
provider = register(
    auto_instrument=False, service_name="travel-assistant-evals-native-py", agent_name=AGENT_NAME
)


@scorer(
    name="uses_tool_data",
    evaluator="uses_tool_data",
    evaluator_type="code",
    description="Names the city's weather and one of its events, or says there is no data",
)
def tool_data(input, output, expected) -> EvalScore:
    passed, why = uses_tool_data(input, output)
    return EvalScore(name="uses_tool_data", value=float(passed), passed=passed, explanation=why)


@scorer(
    name="answer_length",
    threshold=500,
    direction="minimize",
    evaluator="answer_length",
    evaluator_type="code",
    description="Answer length in characters, at most 500",
)
def answer_length(input, output, expected) -> float:
    return float(len(output))


@scorer(
    name="plan_quality",
    threshold=4,
    score_min=1,
    score_max=5,
    evaluator="travel_plan_judge",
    evaluator_type="llm_judge",
    description=JUDGE_DESCRIPTION,
    judge_model=JUDGE_MODEL_ID,
    judge_prompt_name=JUDGE_PROMPT_NAME,
    judge_prompt_version=JUDGE_PROMPT_VERSION,
)
def plan_quality(input, output, expected) -> EvalScore:
    value, reason, judge_trace_id = judge_plan(input, expected, output)
    return EvalScore(
        name="plan_quality", value=value, explanation=reason, judge_trace_id=judge_trace_id
    )


for version, prompt in PROMPTS.items():

    def task(question: str, prompt: str = prompt, version: str = version) -> str:
        return run_agent(question, system_prompt=prompt, version=version).answer

    result = evaluate(
        name=SUITE,
        task=task,
        data=load_test_suite(),
        scores=[tool_data, answer_length, plan_quality],
        agent_version=version,
        run_name=f"{version} prompt",
        run_description=f"System prompt {version}: {prompt}",
    )
    for case in result.cases:
        print(f"  {case.case_id:<14} {case.status:<5} {case.scores}  trace {case.trace_id}")

provider.force_flush()
