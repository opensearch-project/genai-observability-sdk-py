# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Basic tracing with opensearch-genai-observability-sdk-py.

Shows how to register the SDK, trace functions with @observe,
use observe() as a context manager, and enrich spans with
GenAI semantic convention attributes.
"""

from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

# --- Setup ---
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")


# --- Decorator: trace a tool ---
@observe(name="web_search", op=Op.EXECUTE_TOOL)
def search(query: str) -> list[dict]:
    """Simulated web search tool."""
    return [
        {"title": f"Result for: {query}", "url": "https://example.com"},
    ]


# --- Decorator: trace an agent with enrichment ---
@observe(name="research_agent", op=Op.INVOKE_AGENT)
def research(query: str) -> str:
    """Agent that searches, then summarizes via LLM."""
    results = search(query)
    titles = ", ".join(r["title"] for r in results)

    # Enrich the agent span with model details
    enrich(model="gpt-4.1", provider="openai", input_tokens=150, output_tokens=50)

    return f"Summary of: {titles[:100]}"


# --- Context manager: trace inline blocks ---
@observe(name="qa_pipeline", op=Op.INVOKE_AGENT)
def run_pipeline(question: str) -> str:
    """Top-level agent that orchestrates the research."""
    answer = research(question)

    # Trace an inline step (e.g., a guardrail check)
    with observe("safety_check", op="guardrail"):
        is_safe = True  # simulated check
        enrich(safe=is_safe)

    return answer


# --- Run ---
if __name__ == "__main__":
    result = run_pipeline("What is OpenSearch?")
    print(result)

    # Produces this span tree:
    #
    #   invoke_agent qa_pipeline             (@observe, op=INVOKE_AGENT)
    #   ├── invoke_agent research_agent      (@observe, op=INVOKE_AGENT)
    #   │   └── execute_tool web_search      (@observe, op=EXECUTE_TOOL)
    #   └── safety_check                     (context manager, op="guardrail")
