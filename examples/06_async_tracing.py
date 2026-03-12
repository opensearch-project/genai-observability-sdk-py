# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Async function tracing with opensearch-genai-observability-sdk-py.

All decorators support async functions natively — no special config.
"""

import asyncio

from opensearch_genai_observability_sdk_py import register, task, tool, workflow

# --- Setup ---
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")


@tool(name="async_search")
async def search(query: str) -> list[dict]:
    """Simulated async API call."""
    await asyncio.sleep(0.1)  # simulate network latency
    return [{"title": f"Result: {query}"}]


@task(name="async_summarize")
async def summarize(text: str) -> str:
    """Simulated async LLM call."""
    await asyncio.sleep(0.2)
    return f"Summary: {text[:50]}"


@workflow(name="async_pipeline")
async def run_pipeline(question: str) -> str:
    """Async workflow — decorators handle async transparently."""
    results = await search(question)
    summary = await summarize(str(results))
    return summary


if __name__ == "__main__":
    result = asyncio.run(run_pipeline("What is OpenSearch?"))
    print(result)
