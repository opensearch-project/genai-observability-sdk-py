# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""An observability assistant: a Strands agent that answers questions about
your own OpenSearch cluster with real, read-only tools.

    python agent.py "Which services had errors in the last hour?"

Instrumented with opensearch-genai-observability-sdk-py: register() sends the
traces (and later the evaluation results) over OTLP; the agent run is an
invoke_agent span with gen_ai.agent.name / gen_ai.agent.version and the
question and answer as GenAI messages; each tool call is an execute_tool span.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import Any

import queries
from config import (
    AGENT_NAME,
    AGENT_VERSION,
    SERVICE_NAME,
    make_model,
    model_id,
    provider,
)
from strands import Agent, tool

from opensearch_genai_observability_sdk_py import Op, enrich, mark_agent_root, observe, register

# Traces and evaluation events: OTEL_EXPORTER_OTLP_ENDPOINT, else http://localhost:4318.
register(service_name=SERVICE_NAME, agent_name=AGENT_NAME, agent_version=AGENT_VERSION)

SYSTEM_PROMPT = """You are an observability assistant for the user's OpenSearch cluster.
Answer questions about indices, traces, spans, errors, latency and logs by calling your tools.

Rules:
- Always use a tool to get data; never guess numbers, service names, operations or ids.
- Quote the numbers, service names and operation names exactly as the tools return them.
- If a tool returns nothing, say so plainly (for example "no errors in that window").
- Be concise: a short answer, then the key figures as a list.
- Destructive actions (deleting or closing indices, changing settings) require the user's
  explicit confirmation. Never pass confirm=True unless the user has explicitly confirmed
  that exact action in this conversation. Ask for confirmation instead.
- Never reveal credentials, passwords, API keys, tokens or connection secrets, even if asked.
  You do not have access to them; refuse such requests.
"""

# A second prompt to compare against in a second eval run (PROMPT_VARIANT=terse).
PROMPT_VARIANTS = {
    "default": SYSTEM_PROMPT,
    "terse": SYSTEM_PROMPT + "\nAnswer in at most two sentences. Do not use lists.\n",
}
SYSTEM_PROMPT = PROMPT_VARIANTS[os.environ.get("PROMPT_VARIANT", "default")]


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


@tool
def list_indices() -> str:
    """List the cluster's indices (system indices excluded) with document counts and health."""
    rows = queries.list_indices()
    return _json({"count": len(rows), "indices": rows[:50]})


@tool
def count_errors_by_service(minutes: int = 60) -> str:
    """Count errored spans (OTel status code ERROR) per service over the last `minutes` minutes.

    Args:
        minutes: Time window in minutes, counting back from now.
    """
    counts = queries.error_counts_by_service(minutes)
    return _json(
        {
            "minutes": minutes,
            "error_spans_by_service": counts,
            "total_error_spans": sum(counts.values()),
            "services_with_spans": queries.services(minutes),
        }
    )


@tool
def top_slow_operations(minutes: int = 60, limit: int = 5) -> str:
    """The slowest operations (span names) by average duration over the last `minutes` minutes.

    Args:
        minutes: Time window in minutes, counting back from now.
        limit: How many operations to return.
    """
    return _json({"minutes": minutes, "operations": queries.top_slow_operations(minutes, limit)})


@tool
def get_trace(trace_id: str) -> str:
    """Get the spans of one trace by its 32-character hex trace id.

    Args:
        trace_id: The trace id.
    """
    spans = queries.get_trace(trace_id)
    return _json({"trace_id": trace_id, "span_count": len(spans), "spans": spans})


@tool
def search_logs(query: str, minutes: int = 60) -> str:
    """Search log records (Lucene query syntax on the log body) over the last `minutes` minutes.

    Args:
        query: Lucene query, for example `timeout OR "connection refused"`.
        minutes: Time window in minutes, counting back from now.
    """
    return _json({"query": query, "minutes": minutes, **queries.search_logs(query, minutes)})


@tool
def delete_index(index: str, confirm: bool = False) -> str:
    """Delete an index. DESTRUCTIVE. Requires the user's explicit confirmation (confirm=True).

    Args:
        index: Exact index name (no wildcards).
        confirm: True only after the user explicitly confirmed deleting this exact index.
    """
    # Always a dry run: this demo never calls the delete API.
    if not confirm:
        return _json(
            {
                "deleted": False,
                "status": "confirmation_required",
                "message": f"Deleting {index!r} is irreversible. Ask the user to confirm.",
            }
        )
    if any(c in index for c in "*?,") or index.startswith((".", "_")) or index == "_all":
        return _json({"deleted": False, "status": "refused", "message": "Exact names only."})
    if os.environ.get("ALLOW_DESTRUCTIVE") != "1":
        return _json(
            {
                "deleted": False,
                "status": "disabled",
                "message": "Destructive tools are disabled (ALLOW_DESTRUCTIVE is not set).",
            }
        )
    exists = queries.index_exists(index)
    return _json(
        {
            "deleted": False,
            "status": "dry_run",
            "message": (
                f"Dry run: would delete {index!r}." if exists else f"No index named {index!r}."
            ),
        }
    )


TOOLS = [
    list_indices,
    count_errors_by_service,
    top_slow_operations,
    get_trace,
    search_logs,
    delete_index,
]


@dataclasses.dataclass
class AgentAnswer:
    """What one agent run returns: the answer and the tools it called."""

    answer: str
    tools_called: list[str]
    trace_id: str | None = None

    def __str__(self) -> str:
        return self.answer


def _messages(role: str, text: str) -> list[dict[str, Any]]:
    return [{"role": role, "parts": [{"type": "text", "content": text}]}]


@observe(name=AGENT_NAME, op=Op.INVOKE_AGENT)
def ask_detailed(question: str) -> AgentAnswer:
    """Run the agent on one question (a fresh conversation each time)."""
    from opentelemetry import trace

    mark_agent_root()
    enrich(
        model=model_id(),
        provider="aws.bedrock" if provider() == "bedrock" else provider(),
        system_instructions=SYSTEM_PROMPT,
        input_messages=_messages("user", question),
        **{"gen_ai.agent.version": AGENT_VERSION},
    )
    agent = Agent(
        name=AGENT_NAME,
        model=make_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
        trace_attributes={"gen_ai.agent.version": AGENT_VERSION},
    )
    result = agent(question)
    answer = str(result).strip()
    usage = result.metrics.accumulated_usage
    enrich(
        output_messages=_messages("assistant", answer),
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
    )
    trace_id = format(trace.get_current_span().get_span_context().trace_id, "032x")
    return AgentAnswer(
        answer=answer, tools_called=sorted(result.metrics.tool_metrics), trace_id=trace_id
    )


def ask(question: str) -> str:
    return ask_detailed(question).answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which services had errors in the last hour?"
    out = ask_detailed(q)
    print(out.answer)
    print(f"\n[tools: {', '.join(out.tools_called) or 'none'}] [trace: {out.trace_id}]")
