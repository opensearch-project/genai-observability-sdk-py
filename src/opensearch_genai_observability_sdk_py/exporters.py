# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""OTLP span and log exporters with AWS SigV4 support.

Provides drop-in replacements for OTLPSpanExporter and OTLPLogExporter that
sign requests using AWS Signature Version 4 for AWS-hosted OpenSearch
and Data Prepper endpoints. The log exporter carries the evaluation events
(``gen_ai.evaluation.result``). AWS OSIS needs a separate logs pipeline for
them. ``register(sigv4=...)`` builds both exporters from the resolved
traces and logs URLs.

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
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
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
        url: str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response:
        import botocore.auth
        from botocore.awsrequest import AWSRequest

        frozen = self._credentials.get_frozen_credentials()
        signer = botocore.auth.SigV4Auth(frozen, self._service, self._region)

        # Sign over the real body so the SHA256 payload hash is correct.
        data = kwargs.get("data")
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
        headers = kwargs.get("headers") or {}
        for key, value in aws_request.headers.items():
            if key.lower() != "content-type":  # already set by the OTLP exporter
                headers[key] = value
        kwargs["headers"] = headers

        return super().request(method, url, *args, **kwargs)


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
        # Pass the signing session to OTLPSpanExporter, which routes all HTTP
        # calls through it. Keep our own reference: where the parent stores the
        # session is an implementation detail (1.45 moved it into a transport).
        self.signing_session = _make_signing_session(service, region)
        self.service = service
        self.region = self.signing_session._region
        kwargs["session"] = self.signing_session
        super().__init__(*args, **kwargs)

        logger.info(
            "AWSSigV4OTLPExporter initialized for service=%s region=%s",
            service,
            self.region,
        )


class AWSSigV4OTLPLogExporter(OTLPLogExporter):
    """OTLP HTTP log exporter that signs requests with AWS SigV4.

    Same credential and region resolution as ``AWSSigV4OTLPExporter``.
    ``register(sigv4=...)`` creates one for the resolved logs endpoint; build
    it directly and pass it as ``register(log_exporter=...)`` for full control.

    Args:
        endpoint: The OTLP logs endpoint URL (``.../v1/logs``).
        service: The AWS service name for signing (``"osis"`` or ``"es"``).
        region: AWS region. Auto-detected from botocore if not provided.
        **kwargs: Additional arguments passed to OTLPLogExporter.

    Example:
        log_exporter = AWSSigV4OTLPLogExporter(
            endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/logs",
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
        self.signing_session = _make_signing_session(service, region)
        self.service = service
        self.region = self.signing_session._region
        kwargs["session"] = self.signing_session
        super().__init__(*args, **kwargs)

        logger.info(
            "AWSSigV4OTLPLogExporter initialized for service=%s region=%s",
            service,
            self.region,
        )


def _make_signing_session(service: str, region: str | None) -> _SigV4AuthSession:
    """Resolve AWS credentials/region via botocore and build a signing session."""
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

    return _SigV4AuthSession(
        credentials=credentials,
        service=service,
        region=resolved_region,
    )
