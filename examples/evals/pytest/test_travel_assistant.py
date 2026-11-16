# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""The shared test suite as pytest tests; each test's checks are sent as evaluation results.

    pytest -v

One test per case of ../../shared/test_suite.jsonl. Each test runs the agent, logs its
checks to the session's Benchmark run (linked to the agent run's trace), then asserts
them, so a failing check is both a failed test and a `fail` result in OpenSearch.
"""

from __future__ import annotations

import pytest
from agent import CITIES, load_test_suite, run_agent

from opensearch_genai_observability_sdk_py import EvalScore

MAX_CHARS = 800  # the system prompt asks for at most 5 sentences
CASES = load_test_suite()


def uses_tool_data(question: str, answer: str) -> tuple[bool, str]:
    """The plan names the city's weather and one of its events, or says there is no data."""
    text = answer.lower()
    city = next((c for c in CITIES if c in question.lower()), None)
    if city is None:
        refused = any(p in text for p in ("no data", "don't have", "do not have"))
        return refused, "says there is no data" if refused else "does not say there is no data"
    data = CITIES[city]
    weather = data["weather"]["condition"] in text
    events = [e["name"] for e in data["events"] if e["name"].split()[0].lower() in text]
    why = f"weather {'named' if weather else 'missing'}, events named: {events or 'none'}"
    return weather and bool(events), why


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_travel_assistant(case, eval_run):
    ids = {"case_id": case["case_id"], "case_name": case["case_name"]}
    try:
        run = run_agent(case["input"])
    except Exception as e:  # the agent failed: record an error result for each check
        eval_run.log(
            **ids, error_type=type(e).__name__, scores={"uses_tool_data": 0.0, "answer_length": 0.0}
        )
        raise

    grounded, why = uses_tool_data(case["input"], run.answer)
    eval_run.log(
        **ids,
        input=case["input"],
        output=run.answer,
        expected=case["expected"],
        trace_id=run.trace_id,  # link the results to the agent run
        span_id=run.span_id,
        evaluator_type="code",
        scores=[
            EvalScore(
                name="uses_tool_data",
                value=float(grounded),
                passed=grounded,
                explanation=why,
                evaluator="uses_tool_data",
            ),
            EvalScore(
                name="answer_length",
                value=float(len(run.answer)),
                threshold=MAX_CHARS,
                direction="minimize",
                evaluator="answer_length",
            ),
        ],
    )

    assert grounded, why
    assert len(run.answer) <= MAX_CHARS, f"{len(run.answer)} characters"
