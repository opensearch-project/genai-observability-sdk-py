# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Shared configuration: the OpenSearch client and the Strands model, from env vars.

OpenSearch (the cluster the agent answers questions about, and where the
traces and evaluation results land):

    OPENSEARCH_URL          default https://localhost:9200
    OPENSEARCH_USERNAME     basic auth user (default admin when a password is set)
    OPENSEARCH_PASSWORD     basic auth password
    OPENSEARCH_AWS_SIGV4    set to "es" (managed domain) or "aoss" (serverless) for SigV4
    OPENSEARCH_VERIFY_CERTS default false for https://localhost, else true
    AWS_REGION              region for SigV4 and Bedrock (default us-west-2)

Model (Strands model providers):

    MODEL_PROVIDER   bedrock (default), openai or anthropic
    MODEL_ID         model id; defaults per provider below
    OPENAI_API_KEY / ANTHROPIC_API_KEY for the API providers
    JUDGE_MODEL_ID   model id of the LLM judge (default: MODEL_ID)
"""

from __future__ import annotations

import os
import warnings
from typing import Any
from urllib.parse import urlparse

AGENT_NAME = "observability_assistant"
AGENT_VERSION = os.environ.get("AGENT_VERSION", "1.0.0")
SERVICE_NAME = "observability-assistant"

SPAN_INDEX = os.environ.get("SPAN_INDEX", "otel-v1-apm-span*")
LOG_INDEX = os.environ.get("LOG_INDEX", "logs-otel-v1*")
EVALUATION_INDEX = os.environ.get("EVALUATION_INDEX", "otel-v1-genai-evaluation*")

DEFAULT_MODEL_IDS = {
    # A cross-region inference profile; list yours with
    # aws bedrock list-inference-profiles --region us-west-2
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
}


def quiet_tls_warnings() -> None:
    """The local stack uses a self-signed certificate: silence the per-request warnings."""
    if not verify_certs():
        warnings.filterwarnings("ignore", message=".*verify_certs=False.*")
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"


def opensearch_url() -> str:
    return os.environ.get("OPENSEARCH_URL", "https://localhost:9200")


def verify_certs() -> bool:
    raw = os.environ.get("OPENSEARCH_VERIFY_CERTS")
    if raw is not None:
        return raw.lower() in ("1", "true", "yes")
    host = urlparse(opensearch_url()).hostname or ""
    return host not in ("localhost", "127.0.0.1")


def opensearch_auth() -> Any:
    """Basic auth tuple, a SigV4 signer, or None."""
    sigv4_service = os.environ.get("OPENSEARCH_AWS_SIGV4")
    if sigv4_service:
        import boto3
        from opensearchpy import RequestsAWSV4SignerAuth

        credentials = boto3.Session().get_credentials()
        return RequestsAWSV4SignerAuth(credentials, region(), sigv4_service)
    password = os.environ.get("OPENSEARCH_PASSWORD")
    if password:
        return (os.environ.get("OPENSEARCH_USERNAME", "admin"), password)
    return None


def opensearch_client() -> Any:
    """An opensearch-py client for the user's cluster."""
    from opensearchpy import OpenSearch, RequestsHttpConnection

    url = opensearch_url()
    kwargs: dict[str, Any] = {
        "hosts": [url],
        "use_ssl": url.startswith("https"),
        "verify_certs": verify_certs(),
        "ssl_show_warn": False,
        "timeout": 30,
    }
    auth = opensearch_auth()
    if auth is not None:
        kwargs["http_auth"] = auth
        if not isinstance(auth, tuple):
            kwargs["connection_class"] = RequestsHttpConnection
    return OpenSearch(**kwargs)


def provider() -> str:
    return os.environ.get("MODEL_PROVIDER", "bedrock").lower()


def model_id() -> str:
    return os.environ.get("MODEL_ID") or DEFAULT_MODEL_IDS[provider()]


def judge_model_id() -> str:
    return os.environ.get("JUDGE_MODEL_ID") or model_id()


def make_model(model: str | None = None, *, max_tokens: int = 1024) -> Any:
    """A Strands model for MODEL_PROVIDER (bedrock, openai or anthropic)."""
    name = model or model_id()
    p = provider()
    if p == "bedrock":
        from strands.models import BedrockModel

        return BedrockModel(model_id=name, region_name=region(), max_tokens=max_tokens)
    if p == "openai":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(
            client_args={"api_key": os.environ["OPENAI_API_KEY"]},
            model_id=name,
            params={"max_tokens": max_tokens},
        )
    if p == "anthropic":
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(
            client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
            model_id=name,
            max_tokens=max_tokens,
        )
    raise ValueError(f"MODEL_PROVIDER must be bedrock, openai or anthropic; got {p!r}")


quiet_tls_warnings()
