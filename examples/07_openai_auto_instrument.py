# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Auto-instrumentation with opensearch-genai-observability-sdk-py.

register() auto-discovers and activates any installed OpenTelemetry
or OpenInference instrumentors. Just pip install them — no code changes.

This example shows OpenAI auto-instrumentation: every openai.chat.completions
call is automatically traced as an OTEL span with gen_ai.* attributes.
"""

# Step 1: pip install opensearch-genai-observability-sdk-py opentelemetry-instrumentation-openai
#   (or: pip install opensearch-genai-observability-sdk-py openinference-instrumentation-openai)
#
# Both entry point groups are discovered:
#   - opentelemetry_instrumentor  (OpenLLMetry ecosystem)
#   - openinference_instrumentor  (Phoenix/Arize ecosystem)

from opensearch_genai_observability_sdk_py import register, workflow

# Step 2: register() auto-instruments all installed libraries
register(
    endpoint="http://localhost:21890/opentelemetry/v1/traces",
    service_name="my-chatbot",
)

# Step 3: Use OpenAI as normal — calls are traced automatically
from openai import OpenAI  # noqa: E402

client = OpenAI()


@workflow(name="chat")
def chat(question: str) -> str:
    """Every OpenAI call inside this workflow is auto-traced."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    answer = chat("What is OpenSearch?")
    print(answer)

    # Span tree:
    #
    #   chat                           (workflow, from @workflow)
    #   └── openai.chat.completions    (auto-instrumented by OpenLLMetry/OpenInference)
    #       Attributes:
    #         gen_ai.system = "openai"
    #         gen_ai.request.model = "gpt-4o-mini"
    #         gen_ai.usage.input_tokens = 12
    #         gen_ai.usage.output_tokens = 87
    #         gen_ai.response.model = "gpt-4o-mini-2024-07-18"

    # Supported auto-instrumentors (just pip install):
    #
    # OpenLLMetry (opentelemetry_instrumentor group):
    #   opentelemetry-instrumentation-openai
    #   opentelemetry-instrumentation-anthropic
    #   opentelemetry-instrumentation-bedrock
    #   opentelemetry-instrumentation-langchain
    #   opentelemetry-instrumentation-llamaindex
    #   opentelemetry-instrumentation-chromadb
    #   ... 30+ more
    #
    # OpenInference (openinference_instrumentor group):
    #   openinference-instrumentation-openai
    #   openinference-instrumentation-langchain
    #   openinference-instrumentation-llama-index
    #   ... 15+ more
