# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Online evaluation: judge recent agent traces stored in OpenSearch and post the results back.

    python traffic.py      # a few live requests
    python online_eval.py  # judge the last hour of traces that have no judge result yet

For each trace: read the question and answer from the agent's root span, the tool calls
from its execute_tool spans, run the checks, and record each result with
score(trace_id=..., span_id=...) on the root span. Online results have no test.suite.run.id.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from scorers import (
    JUDGE_DESCRIPTION,
    JUDGE_MODEL_ID,
    JUDGE_PROMPT_NAME,
    JUDGE_PROMPT_VERSION,
    judge_plan,
    uses_tool_data,
)

from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever, register, score

SERVICE = "travel-assistant-evals-native"
EXPECTED = (
    "A short plan that uses only the tool data: the city's weather and at least one event, "
    "or says there is no data for the city."
)

provider = register(service_name=f"{SERVICE}-judge")
retriever = OpenSearchTraceRetriever(
    host=os.environ.get("OPENSEARCH_URL", "https://localhost:9200"),
    auth=(os.environ.get("OPENSEARCH_USER", "admin"), os.environ["OPENSEARCH_PASSWORD"]),
    verify_certs=os.environ.get("OPENSEARCH_VERIFY_CERTS", "false") == "true",  # local stack
)

since = datetime.now(timezone.utc) - timedelta(hours=1)
roots = retriever.list_root_spans(services=[SERVICE], since=since, max_results=20)
done = retriever.find_evaluated_trace_ids(
    [r.trace_id for r in roots], evaluator="travel_plan_judge"
)


def is_test_run(root) -> bool:
    """Offline evaluate() cases carry test.suite.run.id; they are scored by their run."""
    return "test.suite.run.id" in root.raw.get("attributes", {})


todo = [
    r
    for r in roots
    if r.trace_id not in done and not is_test_run(r) and r.input_messages and r.output_messages
]
print(f"{len(roots)} recent traces, {len(done)} already judged, {len(todo)} to judge")

for root in todo:
    question = root.input_messages[-1].content
    answer = root.output_messages[-1].content
    spans = retriever.get_traces(root.trace_id).traces[0].spans
    tool_spans = [s for s in spans if s.operation_name == "execute_tool"]
    tools = sorted({s.tool_name for s in tool_spans})
    tool_results = "; ".join(f"{s.tool_name}: {s.tool_call_result}" for s in tool_spans)
    target = {
        "trace_id": root.trace_id,
        "span_id": root.span_id,  # the agent's root span
        "agent_name": root.agent_name or None,
        "agent_version": root.agent_version or None,
    }

    value, reason = judge_plan(question, EXPECTED, answer, tool_results)
    score(
        "plan_quality",
        value,
        **target,
        explanation=reason,
        threshold=4,
        score_min=1,
        score_max=5,
        evaluator="travel_plan_judge",
        evaluator_type="llm_judge",
        evaluator_description=JUDGE_DESCRIPTION,
        judge_model=JUDGE_MODEL_ID,
        judge_prompt_name=JUDGE_PROMPT_NAME,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
    )

    passed, why = uses_tool_data(question, answer)
    score(
        "uses_tool_data",
        float(passed),
        **target,
        label="pass" if passed else "fail",
        explanation=f"{why}; tools called: {tools or 'none'}",
        evaluator="uses_tool_data",
        evaluator_type="code",
    )
    print(f"  {root.trace_id}  plan_quality={value} ({reason})  uses_tool_data={passed}")

provider.force_flush()
