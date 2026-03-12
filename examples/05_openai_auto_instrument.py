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
instrumentors. Just pip install them — no code changes.

This example shows OpenAI auto-instrumentation: every openai.chat.completions
call is automatically traced as an OTel span with gen_ai.* attributes.
"""

# Step 1: pip install opensearch-genai-observability-sdk-py[openai]

from opensearch_genai_observability_sdk_py import Op, observe, register

# Step 2: register() auto-instruments all installed libraries
register(
    endpoint="http://localhost:21890/opentelemetry/v1/traces",
    service_name="my-chatbot",
)

# Step 3: Use OpenAI as normal — calls are traced automatically
from openai import OpenAI  # noqa: E402

client = OpenAI()


@observe(name="chat", op=Op.INVOKE_AGENT)
def chat(question: str) -> str:
    """Every OpenAI call inside this is auto-traced by the instrumentor."""
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
    #   invoke_agent chat                  (@observe, op=INVOKE_AGENT)
    #   └── openai.chat.completions        (auto-instrumented)
    #       Attributes:
    #         gen_ai.system = "openai"
    #         gen_ai.request.model = "gpt-4o-mini"
    #         gen_ai.usage.input_tokens = 12
    #         gen_ai.usage.output_tokens = 87
