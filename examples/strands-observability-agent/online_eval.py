# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Online evaluation: sample recent agent traces from OpenSearch, judge them, post results back.

    python online_eval.py                       # one pass over the last 60 minutes
    python online_eval.py --sample-rate 0.25 --minutes 30
    python online_eval.py --interval 300        # loop every 5 minutes

For each sampled trace that has no evaluation result yet:

- ``answer_quality``: the LLM judge (1 to 5, pass at 4), with the judge's model, prompt
  name / version and description recorded on the result.
- ``known_services``: a cheap deterministic check. Every service the answer names
  must exist in the span index, and an answer about services or errors must name at
  least one (or say there is no data).

Results are posted with ``score(..., trace_id=..., span_id=...)`` on the agent's
root span, so they show next to the trace and on the agent's Evaluations page
as online results (no test.suite.run.id).
"""

from __future__ import annotations

import argparse
import random
import re
import time
from datetime import datetime, timedelta, timezone

import queries
from config import (
    AGENT_NAME,
    EVALUATION_INDEX,
    SERVICE_NAME,
    SPAN_INDEX,
    judge_model_id,
    opensearch_auth,
    opensearch_url,
    verify_certs,
)
from judge import JUDGE_DESCRIPTION, JUDGE_PROMPT_NAME, JUDGE_PROMPT_VERSION, judge
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever, register, score

# The evaluator's own telemetry (the judge's traces and the results).
provider = register(service_name=f"{SERVICE_NAME}-evaluator")

retriever = OpenSearchTraceRetriever(
    host=opensearch_url(),
    index=SPAN_INDEX,
    evaluation_index=EVALUATION_INDEX,
    auth=opensearch_auth(),
    verify_certs=verify_certs(),
)

_SERVICE_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
_NO_DATA = ("no error", "no services", "no spans", "no data", "none", "zero")


def known_services_check(question: str, answer: str, live: set[str]) -> tuple[bool | None, str]:
    """``(verdict, why)``; verdict ``None`` when the check does not apply."""
    named = {t for t in _SERVICE_TOKEN.findall(answer) if t in live}
    unknown = {
        t
        for t in _SERVICE_TOKEN.findall(answer)
        if t not in live and ("-prod" in t or "-service" in t or t.endswith(("-ci", "-evals")))
    }
    about_services = any(w in question.lower() for w in ("service", "error"))
    if not about_services and not named and not unknown:
        return None, "not about services"
    if unknown:
        return False, f"names services that do not exist: {sorted(unknown)}"
    if about_services and not named and not any(w in answer.lower() for w in _NO_DATA):
        return False, "names no existing service"
    return True, f"names existing services: {sorted(named)}" if named else "says there is no data"


def one_pass(minutes: int, sample_rate: float, max_traces: int) -> int:
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    roots = retriever.list_root_spans(services=[SERVICE_NAME], since=since, max_results=200)
    # This agent's runs only, and not the offline eval runs (their root spans carry
    # test.suite.run.id, and they are scored by offline_eval.py).
    roots = [
        r
        for r in roots
        if r.agent_name == AGENT_NAME and "test.suite.run.id" not in r.raw.get("attributes", {})
    ]
    done = retriever.find_evaluated_trace_ids([r.trace_id for r in roots])
    todo = [r for r in roots if r.trace_id not in done]
    # random is fine for sampling (not security relevant).
    sampled = [r for r in todo if random.random() < sample_rate][:max_traces]  # noqa: S311
    print(
        f"{len(roots)} agent traces in the last {minutes} min, {len(done)} already evaluated, "
        f"{len(sampled)} sampled"
    )
    live = set(queries.services(max(minutes, 10080)))
    for root in sampled:
        question = root.input_messages[0].content if root.input_messages else ""
        answer = root.output_messages[-1].content if root.output_messages else ""
        version = root.raw.get("attributes", {}).get("gen_ai.agent.version")
        if not question or not answer:
            print(f"  {root.trace_id}: no input / output messages, skipped")
            continue
        common = {
            "trace_id": root.trace_id,
            "span_id": root.span_id,
            "agent_name": AGENT_NAME,
            "agent_version": version,
        }
        try:
            verdict = judge(question, answer)
        except Exception as exc:  # the judge failed: record an evaluator error
            score(
                "answer_quality",
                error_type=type(exc).__name__,
                evaluator="answer_quality_judge",
                evaluator_type="llm_judge",
                **common,
            )
            print(f"  {root.trace_id}: judge failed ({type(exc).__name__})")
            continue
        score(
            "answer_quality",
            verdict["score"],
            explanation=verdict["reason"],
            threshold=4,
            score_min=1,
            score_max=5,
            evaluator="answer_quality_judge",
            evaluator_version=JUDGE_PROMPT_VERSION,
            evaluator_type="llm_judge",
            evaluator_description=JUDGE_DESCRIPTION,
            judge_model=judge_model_id(),
            judge_prompt_name=JUDGE_PROMPT_NAME,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            judge_trace_id=verdict["judge_trace_id"],
            **common,
        )
        ok, why = known_services_check(question, answer, live)
        if ok is not None:
            score(
                "known_services",
                1.0 if ok else 0.0,
                label="pass" if ok else "fail",
                explanation=why,
                evaluator="known_services_check",
                evaluator_type="code",
                evaluator_description="Every service the answer names exists in the span index",
                **common,
            )
        print(f"  {root.trace_id}: answer_quality={verdict['score']:.0f} known_services={ok}")
    trace.get_tracer_provider().force_flush()  # type: ignore[attr-defined]
    return len(sampled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minutes", type=int, default=60, help="time window to sample from")
    parser.add_argument("--sample-rate", type=float, default=0.5, help="fraction to evaluate")
    parser.add_argument("--max-traces", type=int, default=20, help="cap per pass")
    parser.add_argument("--interval", type=int, default=0, help="loop every N seconds (0: once)")
    args = parser.parse_args()
    while True:
        one_pass(args.minutes, args.sample_rate, args.max_traces)
        if not args.interval:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
