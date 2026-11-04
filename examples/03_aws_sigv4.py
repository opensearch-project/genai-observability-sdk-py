# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""AWS SigV4 authentication with opensearch-genai-observability-sdk-py.

When your Data Prepper or OpenSearch Ingestion pipeline is hosted on AWS,
pass sigv4= to register(): it builds SigV4-signed exporters for traces and
for evaluation events.

OpenSearch Ingestion (OSIS) has one pipeline per signal: traces go to the
traces pipeline, evaluation events (score(), evaluate(), Benchmark) go to a
logs pipeline. Pass both URLs, as traces_endpoint= and logs_endpoint=.

Requires: pip install opensearch-genai-observability-sdk-py[aws]
"""

from opensearch_genai_observability_sdk_py import Op, observe, register, score

TRACES_ENDPOINT = "https://my-traces-pipeline.us-east-1.osis.amazonaws.com/v1/traces"
LOGS_ENDPOINT = "https://my-logs-pipeline.us-east-1.osis.amazonaws.com/v1/logs"

# Uses the standard boto3 credential chain (env vars, ~/.aws/credentials, IAM role).
provider = register(
    project_name="my-llm-app",
    traces_endpoint=TRACES_ENDPOINT,
    logs_endpoint=LOGS_ENDPOINT,
    sigv4={
        "service": "osis",  # "osis" for OpenSearch Ingestion, "es" for OpenSearch Service
        # "region": "us-east-1",  # optional, auto-detected from botocore if not set
    },
)


@observe(name="qa_pipeline", op=Op.INVOKE_AGENT)
def run(question: str) -> str:
    answer = f"Answer to: {question}"
    # Inline evaluation: the event is parented to the active qa_pipeline span.
    score("relevance", 0.9, label="relevant", evaluator_type="human")
    return answer


if __name__ == "__main__":
    print(run("What is OpenSearch?"))
    provider.force_flush()
