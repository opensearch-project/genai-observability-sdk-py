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
pass AWSSigV4OTLPExporter directly to register() via the exporter= parameter.

Requires: pip install opensearch-genai-observability-sdk-py[aws]
"""

from opensearch_genai_observability_sdk_py import Op, observe, register, score
from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

ENDPOINT = "https://my-pipeline.us-east-1.osis.amazonaws.com/v1/traces"

# Pass AWSSigV4OTLPExporter explicitly.
# Uses the standard boto3 credential chain (env vars, ~/.aws/credentials, IAM role).
register(
    project_name="my-llm-app",
    exporter=AWSSigV4OTLPExporter(
        endpoint=ENDPOINT,
        service="osis",  # "osis" for OpenSearch Ingestion Service
        # region="us-east-1",   # optional — auto-detected from botocore if not set
    ),
)


@observe(name="qa_pipeline", op=Op.INVOKE_AGENT)
def run(question: str) -> str:
    return f"Answer to: {question}"


if __name__ == "__main__":
    result = run("What is OpenSearch?")
    print(result)

    score(
        name="relevance",
        value=0.9,
        trace_id="abc123",
        source="human",
    )
