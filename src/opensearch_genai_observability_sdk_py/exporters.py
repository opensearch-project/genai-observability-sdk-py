# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""OTLP span exporters with AWS SigV4 support.

Provides a drop-in replacement for OTLPSpanExporter that signs
requests using AWS Signature Version 4 for AWS-hosted OpenSearch
and Data Prepper endpoints.

Design note
-----------
The previous implementation signed a *synthetic* request with an empty body
(``data=b""``) and patched the Authorization header onto the exporter's
internal ``_session``. That approach was incorrect: AWS SigV4 includes
``SHA256(body)`` in the canonical request, so signing over an empty body
while sending a non-empty protobuf payload produces a body-hash mismatch
and a ``403 SignatureDoesNotMatch`` from AWS.

The fix (same approach as aws-otel-python-instrumentation) is to subclass
``requests.Session`` and override ``request()``.  By the time ``request()``
is called, the OTLP exporter has already serialized the spans into protobuf
bytes and passed them as ``data=``.  We sign over that real payload, so the
body hash in the Authorization header always matches what AWS receives.

``OTLPSpanExporter`` (and its parent ``OTLPExporterMixin``) accept a
``session=`` constructor argument and use it for all HTTP calls, so there
is no need to override ``export()`` or touch ``_session`` after init.
"""

from __future__ import annotations

import logging
from typing import Any

import requests  # type: ignore[import-untyped]
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)


class _SigV4AuthSession(requests.Session):
    """A ``requests.Session`` that signs every request with AWS SigV4.

    Signing happens inside ``request()``, at which point the real
    serialized body (protobuf bytes) is available as ``data=``.  This
    ensures the body hash in the Authorization header always matches
    the payload that AWS actually receives.
    """

    def __init__(self, credentials: Any, service: str, region: str) -> None:
        super().__init__()
        self._credentials = credentials
        self._service = service
        self._region = region

    def request(
        self,
        method: str,
        url: str,
        *args: Any,
        data: Any = None,
        headers: Any = None,
        **kwargs: Any,
    ) -> Any:  # noqa: E501
        import botocore.auth
        from botocore.awsrequest import AWSRequest

        frozen = self._credentials.get_frozen_credentials()
        signer = botocore.auth.SigV4Auth(frozen, self._service, self._region)

        # Sign over the real body so the SHA256 payload hash is correct.
        aws_request = AWSRequest(
            method=method,
            url=url,
            data=data if data is not None else b"",
            headers={"Content-Type": "application/x-protobuf"},
        )
        signer.add_auth(aws_request)

        # Merge all SigV4 headers into the headers that will be sent.
        # Copy every header botocore added (Authorization, X-Amz-Date,
        # X-Amz-Security-Token, X-Amz-Content-Sha256, etc.) — OSIS requires
        # X-Amz-Content-Sha256 and will drop the connection if it is missing.
        if headers is None:
            headers = {}
        for key, value in aws_request.headers.items():
            if key.lower() != "content-type":  # already set by the OTLP exporter
                headers[key] = value

        return super().request(*args, method=method, url=url, data=data, headers=headers, **kwargs)


class AWSSigV4OTLPExporter(OTLPSpanExporter):
    """OTLP HTTP span exporter that signs requests with AWS SigV4.

    Uses botocore's credential chain (env vars, ~/.aws/credentials,
    IAM roles, IMDS) to resolve credentials automatically.

    A ``_SigV4AuthSession`` is passed to the parent ``OTLPSpanExporter``
    via the ``session=`` constructor argument.  All HTTP calls go through
    that session, so every export is signed over the real protobuf body.

    Args:
        endpoint: The OTLP endpoint URL.
        service: The AWS service name for signing. Use "osis" for
            OpenSearch Ingestion, "es" for OpenSearch Service direct.
        region: AWS region. Auto-detected from botocore if not provided.
        **kwargs: Additional arguments passed to OTLPSpanExporter.

    Example:
        exporter = AWSSigV4OTLPExporter(
            endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
            service="osis",
        )
    """

    def __init__(
        self,
        *args: Any,
        service: str = "osis",
        region: str | None = None,
        **kwargs: Any,
    ) -> None:
        import botocore.session

        botocore_session = botocore.session.get_session()
        credentials = botocore_session.get_credentials()
        resolved_region = region or botocore_session.get_config_variable("region")

        if not credentials:
            raise RuntimeError(
                "No AWS credentials found. Configure credentials via environment "
                "variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), "
                "~/.aws/credentials, or an IAM role."
            )
        if not resolved_region:
            raise RuntimeError(
                "No AWS region found. Set the region via the 'region' parameter, "
                "AWS_DEFAULT_REGION environment variable, or ~/.aws/config."
            )

        # Pass the signing session to OTLPSpanExporter.  The parent stores it
        # as self._session and routes all HTTP calls through it.
        kwargs["session"] = _SigV4AuthSession(
            credentials=credentials,
            service=service,
            region=resolved_region,
        )
        super().__init__(*args, **kwargs)

        logger.info(
            "AWSSigV4OTLPExporter initialized for service=%s region=%s",
            service,
            resolved_region,
        )
