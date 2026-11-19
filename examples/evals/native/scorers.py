# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Checks shared by the offline and the online evaluation.

- uses_tool_data: deterministic. The plan names the city's weather and one of its events,
  or, for a city with no data, says so.
- judge_plan: an LLM judge on Amazon Bedrock that grades the plan 1 to 5 against the
  expected behavior. The judge call is traced as its own `chat` span (judge prompt as the
  system instructions, the request and verdict as messages), and its trace id goes on each
  result as judge_trace_id, so a result links to the judge call that produced it.
"""

from __future__ import annotations

import json
import os

import boto3
from agent import CITIES, REGION
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import Op, enrich, observe

JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
JUDGE_PROMPT_NAME = "travel-plan-quality"
JUDGE_PROMPT_VERSION = "v1"  # bump when JUDGE_PROMPT changes
JUDGE_DESCRIPTION = "Plan is grounded in the tool data and meets the expected behavior"
JUDGE_PROMPT = (
    "You grade a travel assistant. Given the user's request, the expected behavior, the "
    "tool results (when given) and the assistant's answer, score the answer from 1 (wrong "
    "or invented) to 5 (correct, grounded, short). Reply with JSON only: "
    '{"score": <1-5>, "reason": "<one sentence>"}'
)

_bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def uses_tool_data(question: str, answer: str) -> tuple[bool, str]:
    """(passed, why) for a plan, checked against the static travel data."""
    text = answer.lower()
    city = next((c for c in CITIES if c in question.lower()), None)
    if city is None:  # a city the tools have no data for
        refused = any(p in text for p in ("no data", "don't have", "do not have"))
        return refused, "says there is no data" if refused else "does not say there is no data"
    data = CITIES[city]
    weather = data["weather"]["condition"] in text
    events = [e["name"] for e in data["events"] if e["name"].split()[0].lower() in text]
    why = f"weather {'named' if weather else 'missing'}, events named: {events or 'none'}"
    return weather and bool(events), why


def _text(role: str, text: str) -> list[dict]:
    return [{"role": role, "parts": [{"type": "text", "content": text}]}]


@observe(name=JUDGE_MODEL_ID, op=Op.CHAT)
def judge_plan(
    question: str, expected: str, answer: str, tool_results: str = ""
) -> tuple[float, str, str]:
    """(score 1 to 5, reason, judge_trace_id) from the LLM judge.

    tool_results: what the tools returned. The call is its own `chat` span; judge_trace_id
    is that span's trace id.
    """
    context = f"\nTool results: {tool_results}" if tool_results else ""
    prompt = f"Request: {question}\nExpected: {expected}{context}\nAnswer: {answer}"
    enrich(
        model=JUDGE_MODEL_ID,
        provider="aws.bedrock",
        system_instructions=JUDGE_PROMPT,  # the judge prompt lives on its LLM span
        input_messages=_text("user", prompt),
    )
    reply = _bedrock.converse(
        modelId=JUDGE_MODEL_ID,
        system=[{"text": JUDGE_PROMPT}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 200, "temperature": 0},
    )
    text = reply["output"]["message"]["content"][0]["text"]
    usage = reply.get("usage", {})
    enrich(
        output_messages=_text("assistant", text),
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
    )
    # The first JSON object in the reply; the model may add text or a second object after it
    verdict, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    judge_trace_id = format(trace.get_current_span().get_span_context().trace_id, "032x")
    return float(verdict["score"]), verdict["reason"], judge_trace_id
