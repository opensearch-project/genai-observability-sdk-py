# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""OTel pipeline setup for OpenSearch AI observability.

The register() function is the single entry point for configuring
tracing. It creates a TracerProvider, sets up the exporter, and
auto-discovers installed instrumentor packages.

Endpoint resolution (priority order)
-------------------------------------
1. ``endpoint=`` parameter — full URL, used as-is.
2. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` env var — full URL, used as-is.
3. ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var — base URL; ``/v1/traces`` is
   appended automatically (per the OTel spec).
4. Data Prepper default: ``http://localhost:21890/opentelemetry/v1/traces``

Protocol resolution (priority order)
--------------------------------------
1. ``protocol=`` parameter — ``"http"`` or ``"grpc"``.
2. ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`` env var.
3. ``OTEL_EXPORTER_OTLP_PROTOCOL`` env var.
4. Inferred from URL scheme: ``grpc://`` / ``grpcs://`` → gRPC;
   everything else (``http://``, ``https://``) → HTTP.

Supported protocols
--------------------
- ``http://`` / ``https://`` → HTTP OTLP (protobuf)
- ``grpc://`` → gRPC (insecure)
- ``grpcs://`` → gRPC with TLS

``http/json`` is not supported.

Raises ``ValueError`` if the explicit protocol contradicts an unambiguous
``grpc://`` or ``grpcs://`` URL scheme.

For authenticated endpoints (e.g. AWS OSIS), pass a custom exporter:

    from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

    register(
        exporter=AWSSigV4OTLPExporter(
            endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
            service="osis",
        )
    )

Note: AWSSigV4OTLPExporter is HTTP-only. AWS OSIS does not expose a gRPC
endpoint.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from typing import Literal
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanProcessor,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:21890/opentelemetry/v1/traces"

# Entry point group to discover instrumentors from.
# OpenLLMetry + official OTel instrumentors register under this group.
_INSTRUMENTOR_GROUPS = [
    "opentelemetry_instrumentor",
]


def register(
    *,
    endpoint: str | None = None,
    protocol: Literal["http", "grpc"] | None = None,
    project_name: str | None = None,
    service_name: str | None = None,
    service_version: str | None = None,
    batch: bool = True,
    auto_instrument: bool = True,
    exporter: SpanExporter | None = None,
    set_global: bool = True,
    headers: dict | None = None,
) -> TracerProvider:
    """Configure the OTel tracing pipeline for OpenSearch.

    One-line setup that creates a TracerProvider, configures an OTLP
    exporter, and auto-discovers installed instrumentor packages.

    Endpoint resolution (priority order):

    1. ``endpoint=`` parameter (full URL, used as-is)
    2. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` env var (full URL, used as-is)
    3. ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var (base URL, ``/v1/traces`` appended)
    4. ``http://localhost:21890/opentelemetry/v1/traces`` (Data Prepper default)

    Protocol resolution (priority order):

    1. ``protocol=`` parameter
    2. ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`` env var
    3. ``OTEL_EXPORTER_OTLP_PROTOCOL`` env var
    4. Inferred from URL scheme (``grpc://`` / ``grpcs://`` → gRPC; else HTTP)

    Supported protocols:

        http:// or https:// → HTTP OTLP (protobuf)
        grpc://             → gRPC (insecure)
        grpcs://            → gRPC (TLS)

    For endpoints that require authentication (e.g. AWS OSIS), pass a
    custom exporter via the ``exporter=`` parameter:

        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

        register(
            exporter=AWSSigV4OTLPExporter(
                endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
                service="osis",
            )
        )

    Args:
        endpoint: OTLP endpoint URL (full URL, used as-is). When omitted,
            resolved from OTel env vars or the Data Prepper default.
            Ignored if ``exporter`` is provided.
        protocol: Force ``"http"`` or ``"grpc"``. When omitted, resolved
            from ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`` /
            ``OTEL_EXPORTER_OTLP_PROTOCOL`` env vars, then inferred from
            the URL scheme. Raises ``ValueError`` if it contradicts a
            ``grpc://`` or ``grpcs://`` URL scheme.
            Ignored if ``exporter`` is provided.
        project_name: Project/service name attached to all spans.
            Defaults to ``OTEL_SERVICE_NAME`` or ``OPENSEARCH_PROJECT``
            env var, or ``"default"``.
        service_name: Alias for ``project_name``.
        service_version: Version string for the service (e.g. ``"1.0.0"``).
            Sets the ``service.version`` resource attribute. When omitted,
            resolved from ``OTEL_SERVICE_VERSION`` env var.
        batch: Use ``BatchSpanProcessor`` (``True``) or
            ``SimpleSpanProcessor`` (``False``). Batch is recommended for
            production.
        auto_instrument: Discover and activate installed instrumentor
            packages automatically.
        exporter: Custom ``SpanExporter``. When provided,
            ``endpoint`` / ``protocol`` / ``headers`` are ignored — the
            exporter is used as-is. Use this to plug in authenticated
            exporters such as ``AWSSigV4OTLPExporter``.
        set_global: Set as the global ``TracerProvider`` (default: ``True``).
        headers: Additional HTTP headers for the default OTLP exporter.
            Ignored if ``exporter`` is provided.

    Returns:
        The configured ``TracerProvider``.

    Raises:
        ValueError: If ``protocol`` contradicts the URL scheme, or if an
            unsupported protocol value is specified.

    Examples:
        # Self-hosted Data Prepper — zero config
        register()

        # OTel Collector via env var (standard OTel behaviour)
        # OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
        register()

        # Explicit endpoint
        register(endpoint="http://my-collector:4318/v1/traces")

        # gRPC via URL scheme
        register(endpoint="grpc://localhost:4317")

        # AWS OSIS with SigV4 signing (HTTP only)
        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter
        register(
            exporter=AWSSigV4OTLPExporter(
                endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
                service="osis",
            )
        )
    """
    resolved_endpoint = _resolve_endpoint(endpoint)
    resolved_protocol = _resolve_protocol(protocol)

    name = (
        service_name
        or project_name
        or os.environ.get("OTEL_SERVICE_NAME")
        or os.environ.get("OPENSEARCH_PROJECT", "default")
    )

    # Step 1: Create Resource (identity tag for all spans)
    version = service_version or os.environ.get("OTEL_SERVICE_VERSION")
    resource_attrs: dict[str, str] = {"service.name": name}
    if version:
        resource_attrs["service.version"] = version
    resource = Resource.create(resource_attrs)

    # Step 2: Create TracerProvider
    provider = TracerProvider(resource=resource)

    # Step 3: Create Exporter
    if exporter is None:
        exporter = _create_exporter(
            endpoint=resolved_endpoint,
            protocol=resolved_protocol,
            headers=headers,
        )

    # Step 4: Create Processor and wire up
    processor: SpanProcessor
    if batch:
        processor = BatchSpanProcessor(exporter)
    else:
        processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Step 5: Set as global provider
    if set_global:
        trace.set_tracer_provider(provider)

    # Step 6: Auto-instrument installed libraries
    if auto_instrument:
        _auto_instrument(provider)

    logger.info(
        "OpenSearch AI tracing initialized: endpoint=%s project=%s",
        resolved_endpoint,
        name,
    )

    return provider


def _resolve_endpoint(endpoint: str | None) -> str:
    """Resolve the OTLP endpoint following OTel env var priority.

    Priority:
    1. ``endpoint`` argument (full URL)
    2. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` (full URL, used as-is)
    3. ``OTEL_EXPORTER_OTLP_ENDPOINT`` (base URL, ``/v1/traces`` appended)
    4. Data Prepper default
    """
    if endpoint:
        return endpoint

    traces_ep = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces_ep:
        return traces_ep

    base_ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base_ep:
        return base_ep.rstrip("/") + "/v1/traces"

    return DEFAULT_ENDPOINT


def _resolve_protocol(protocol: str | None) -> str | None:
    """Resolve the OTLP protocol from code param or env vars.

    Returns the raw value (not yet normalised to 'http'/'grpc') so that
    ``_infer_protocol`` can apply it together with the URL scheme.
    Returns ``None`` if nothing is configured (infer from URL scheme).
    """
    if protocol is not None:
        return protocol

    return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.environ.get(
        "OTEL_EXPORTER_OTLP_PROTOCOL"
    )


def _normalize_protocol(value: str) -> str:
    """Map a protocol string to the internal ``'http'`` or ``'grpc'`` token.

    Accepts both shorthand values (``'http'``, ``'grpc'``) and OTel spec
    values (``'http/protobuf'``, ``'grpc'``).

    Raises:
        ValueError: For ``'http/json'`` (unsupported) or unknown values.
    """
    v = value.strip().lower()
    if v == "grpc":
        return "grpc"
    if v in ("http", "http/protobuf"):
        return "http"
    if v == "http/json":
        raise ValueError("Protocol 'http/json' is not supported. Use 'http/protobuf' or 'grpc'.")
    raise ValueError(f"Unknown OTLP protocol '{value}'. Supported values: 'http/protobuf', 'grpc'.")


def _infer_protocol(endpoint: str, protocol: str | None) -> str:
    """Determine the OTLP transport protocol from explicit setting or URL scheme.

    Args:
        endpoint: The full endpoint URL.
        protocol: Explicit protocol override (``'http'``, ``'grpc'``,
            ``'http/protobuf'``, etc.) or ``None`` to infer from URL.

    Returns:
        ``'http'`` or ``'grpc'``.

    Raises:
        ValueError: If ``protocol`` contradicts a ``grpc://`` or ``grpcs://``
            URL scheme, or if the protocol value is unsupported.
    """
    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()
    url_implies_grpc = scheme in ("grpc", "grpcs")

    if protocol is not None:
        resolved = _normalize_protocol(protocol)
        if url_implies_grpc and resolved == "http":
            raise ValueError(
                f"Conflicting protocol: endpoint scheme '{scheme}://' implies gRPC "
                f"but protocol='{protocol}' was specified. "
                f"Either use protocol='grpc' or change the endpoint scheme to "
                f"'http://' or 'https://'."
            )
        return resolved

    if url_implies_grpc:
        return "grpc"

    return "http"


def _create_exporter(
    endpoint: str,
    protocol: str | None,
    headers: dict | None,
) -> SpanExporter:
    """Create a plain OTLP exporter based on protocol and endpoint."""
    resolved_protocol = _infer_protocol(endpoint, protocol)

    if resolved_protocol == "grpc":
        return _create_grpc_exporter(endpoint, headers)

    return _create_http_exporter(endpoint, headers)


def _create_http_exporter(
    endpoint: str,
    headers: dict | None,
) -> SpanExporter:
    """Create a plain HTTP OTLP exporter."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _create_grpc_exporter(
    endpoint: str,
    headers: dict | None,
) -> SpanExporter:
    """Create a gRPC OTLP exporter."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )

    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()

    # gRPC exporter takes host:port, not a full URL
    grpc_endpoint = parsed.netloc or endpoint
    insecure = scheme != "grpcs" and scheme != "https"

    logger.info("Using gRPC exporter: %s (insecure=%s)", grpc_endpoint, insecure)
    return GRPCSpanExporter(
        endpoint=grpc_endpoint,
        insecure=insecure,
        headers=headers,
    )


def _auto_instrument(provider: TracerProvider) -> None:
    """Discover and activate installed instrumentor packages.

    Searches the OpenTelemetry entry point group so instrumentors
    from the OTel ecosystem are discovered automatically.
    """
    discovered = 0
    seen_names = set()

    for group in _INSTRUMENTOR_GROUPS:
        eps = entry_points(group=group)

        for ep in eps:
            # Avoid double-instrumenting if a package registers in both groups
            if ep.name in seen_names:
                continue
            seen_names.add(ep.name)

            try:
                instrumentor_cls = ep.load()
                instrumentor = instrumentor_cls()
                instrumentor.instrument(tracer_provider=provider)
                discovered += 1
                logger.debug("Instrumented: %s (from %s)", ep.name, group)
            except Exception as exc:
                logger.debug("Skipped instrumentor %s: %s", ep.name, exc)

    if discovered == 0:
        logger.warning(
            "No instrumentor packages found. Install instrumentors to auto-trace "
            "LLM calls, e.g.: pip install opentelemetry-instrumentation-openai"
        )
    else:
        logger.info("Auto-instrumented %d libraries", discovered)
