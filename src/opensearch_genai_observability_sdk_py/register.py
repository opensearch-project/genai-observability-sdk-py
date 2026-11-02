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
auto-discovers installed instrumentor packages. It also sets up a
LoggerProvider with an OTLP log exporter, so evaluation events
(``gen_ai.evaluation.result``) are exported over OTLP logs.

Traces and evaluation logs are separate OTLP endpoints. A collector usually
serves both (``/v1/traces`` and ``/v1/logs``); AWS OpenSearch Ingestion
(OSIS) has one pipeline per signal, so it needs a traces pipeline URL and a
logs pipeline URL.

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

Logs endpoint resolution (priority order)
------------------------------------------
1. ``logs_endpoint=`` parameter: full URL, used as-is.
2. ``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`` env var: full URL, used as-is.
3. An explicit traces endpoint (``endpoint=``, or the endpoint of
   ``exporter=AWSSigV4OTLPExporter(...)``): if its path ends with
   ``/v1/traces``, the same base with ``/v1/logs``. This is the only
   derivation. Any other path logs a warning and the logs pipeline is
   skipped.
4. ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var: base URL; ``/v1/logs`` appended
   (per the OTel spec).
5. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` env var: same rule as 3.
6. Data Prepper default: ``http://localhost:21890/opentelemetry/v1/logs``

With gRPC traces only 1 and 2 apply; otherwise a warning is logged and
logs are skipped. The same happens for a custom exporter without
``log_exporter=``.

The logs protocol follows ``protocol=``, then
``OTEL_EXPORTER_OTLP_LOGS_PROTOCOL`` / ``OTEL_EXPORTER_OTLP_PROTOCOL``, then
the URL scheme. With ``exporter=AWSSigV4OTLPExporter(...)`` the log exporter
is an ``AWSSigV4OTLPLogExporter`` with the same service/region on the
resolved logs URL (pass the OSIS logs pipeline URL as ``logs_endpoint=``).
Any other custom ``exporter=`` needs an explicit ``log_exporter=``; otherwise
the logs pipeline is skipped.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from typing import Any, Literal
from urllib.parse import urlparse

from opentelemetry import _logs, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanProcessor,
)

from opensearch_genai_observability_sdk_py._internal import (
    derive_logs_endpoint,
    set_agent_defaults,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:21890/opentelemetry/v1/traces"
DEFAULT_LOGS_ENDPOINT = "http://localhost:21890/opentelemetry/v1/logs"

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
    enable_logs: bool = True,
    logs_endpoint: str | None = None,
    log_exporter: Any | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
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
            Also sent by the default OTLP log exporter.
            Ignored if ``exporter`` is provided.
        enable_logs: Also configure a ``LoggerProvider`` (OTLP logs) so that
            evaluation events are exported (default: ``True``).
            Only done when ``set_global`` is ``True`` and no global
            ``LoggerProvider`` is configured yet; an existing one is reused
            as-is.
        logs_endpoint: OTLP logs endpoint (full URL), for example
            ``http://localhost:4318/v1/logs`` or the OSIS logs pipeline URL.
            When omitted: ``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT``, else derived
            from an explicit traces endpoint ending with ``/v1/traces``
            (same base, ``/v1/logs``), else ``OTEL_EXPORTER_OTLP_ENDPOINT``
            + ``/v1/logs``. If an explicit traces endpoint has another path,
            a warning is logged and logs are skipped. Also used for the
            SigV4 log exporter. Ignored if ``log_exporter`` is provided.
        log_exporter: Custom ``LogRecordExporter``. Required to export logs
            when a custom ``exporter`` other than ``AWSSigV4OTLPExporter``
            is passed.
        agent_name: Name of the agent this process evaluates. Stamped as
            ``gen_ai.agent.name`` on every evaluation result event the SDK
            emits, unless a call names its own agent or the evaluated span
            carries one.
        agent_version: Version of that agent (``gen_ai.agent.version``), stamped the
            same way. Lets OpenSearch Dashboards compare runs version to
            version.

    Returns:
        The configured ``TracerProvider``. Its ``force_flush()`` and
        ``shutdown()`` also flush and shut down the logs pipeline that
        ``register()`` created.

    Raises:
        ValueError: If ``protocol`` contradicts the URL scheme, or if an
            unsupported protocol value is specified.

    Examples:
        # Self-hosted Data Prepper — zero config
        register()

        # OTel Collector via env var (standard OTel behaviour)
        # OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
        register()

        # Explicit endpoints (traces and evaluation logs)
        register(
            endpoint="http://my-collector:4318/v1/traces",
            logs_endpoint="http://my-collector:4318/v1/logs",
        )

        # gRPC via URL scheme
        register(endpoint="grpc://localhost:4317")

        # AWS OSIS with SigV4 signing (HTTP only): one pipeline per signal
        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter
        register(
            exporter=AWSSigV4OTLPExporter(
                endpoint="https://traces-pipe.us-east-1.osis.amazonaws.com/v1/traces",
                service="osis",
            ),
            logs_endpoint="https://logs-pipe.us-east-1.osis.amazonaws.com/v1/logs",
        )
    """
    set_agent_defaults(agent_name, agent_version)
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
    custom_exporter = exporter is not None
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

    # Step 6: Logs pipeline for evaluation events (score())
    if enable_logs and set_global:
        _setup_logs(
            provider=provider,
            resource=resource,
            batch=batch,
            span_exporter=exporter if custom_exporter else None,
            endpoint=endpoint,
            logs_endpoint=logs_endpoint,
            protocol=protocol,
            headers=headers,
            log_exporter=log_exporter,
        )

    # Step 7: Auto-instrument installed libraries
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


def _warn_logs_skipped(reason: str) -> None:
    logger.warning(
        "register(): %s, so evaluation events will not be exported. Pass logs_endpoint= "
        "(for example http://localhost:4318/v1/logs or the OSIS logs pipeline URL) or set "
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT.",
        reason,
    )


def _logs_from_traces(traces_endpoint: str, source: str) -> str | None:
    """The logs URL for an explicit traces URL (the only derivation), or ``None``."""
    derived = derive_logs_endpoint(traces_endpoint)
    if derived is None:
        _warn_logs_skipped(
            f"cannot derive an OTLP logs endpoint from {source} '{traces_endpoint}' "
            "(only a path ending with /v1/traces maps to /v1/logs)"
        )
    return derived


def _resolve_logs_endpoint(
    endpoint: str | None,
    logs_endpoint: str | None,
    protocol: str | None = None,
) -> str | None:
    """Resolve the OTLP logs endpoint (see module docstring for priority).

    ``endpoint`` is the explicit traces endpoint (``endpoint=`` or the SigV4
    exporter's URL). Returns ``None`` (after a warning) when logs are skipped.
    """
    if logs_endpoint:
        return logs_endpoint

    logs_ep = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if logs_ep:
        return logs_ep

    if _infer_protocol(_resolve_endpoint(endpoint), protocol) == "grpc":
        _warn_logs_skipped("traces are exported over gRPC and no logs endpoint is configured")
        return None

    if endpoint:
        return _logs_from_traces(endpoint, "traces endpoint")

    base_ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base_ep:
        return base_ep.rstrip("/") + "/v1/logs"

    traces_ep = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces_ep:
        return _logs_from_traces(traces_ep, "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")

    return DEFAULT_LOGS_ENDPOINT


def _resolve_logs_protocol(protocol: str | None) -> str | None:
    """Resolve the OTLP logs protocol from code param or env vars."""
    if protocol is not None:
        return protocol

    return os.environ.get("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL") or os.environ.get(
        "OTEL_EXPORTER_OTLP_PROTOCOL"
    )


def _global_logger_provider_is_set() -> bool:
    """Return ``True`` if a global LoggerProvider has already been configured."""
    current = _logs.get_logger_provider()
    try:
        from opentelemetry._logs._internal import ProxyLoggerProvider
    except ImportError:  # pragma: no cover - layout of older/newer OTel APIs
        return isinstance(current, LoggerProvider)
    return not isinstance(current, ProxyLoggerProvider)


class _LoggerProviderLifecycle(SpanProcessor):
    """Ties a LoggerProvider's flush/shutdown to the TracerProvider.

    Registered as the last span processor so ``TracerProvider.force_flush()``
    and ``TracerProvider.shutdown()`` (including the atexit shutdown) also
    flush the evaluation events.
    """

    def __init__(self, logger_provider: LoggerProvider) -> None:
        self._logger_provider = logger_provider

    def shutdown(self) -> None:
        self._logger_provider.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._logger_provider.force_flush(timeout_millis)


def _setup_logs(
    *,
    provider: TracerProvider,
    resource: Resource,
    batch: bool,
    span_exporter: SpanExporter | None,
    endpoint: str | None,
    logs_endpoint: str | None,
    protocol: str | None,
    headers: dict | None,
    log_exporter: Any | None,
) -> LoggerProvider | None:
    """Create and install the global LoggerProvider used by ``score()``."""
    if _global_logger_provider_is_set():
        logger.info(
            "A global LoggerProvider is already configured; evaluation events use it as-is."
        )
        return None

    if log_exporter is None:
        logs_protocol = _resolve_logs_protocol(protocol)
        if span_exporter is not None:
            log_exporter = _log_exporter_for_custom(span_exporter, logs_endpoint)
            if log_exporter is None:
                return None
        else:
            resolved = _resolve_logs_endpoint(endpoint, logs_endpoint, logs_protocol)
            if resolved is None:
                return None
            log_exporter = _create_log_exporter(
                endpoint=resolved, protocol=logs_protocol, headers=headers
            )
            logger.info("OpenSearch AI evaluation events: logs endpoint=%s", resolved)

    logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
    if batch:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    else:
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    _logs.set_logger_provider(logger_provider)
    provider.add_span_processor(_LoggerProviderLifecycle(logger_provider))
    return logger_provider


def _log_exporter_for_custom(span_exporter: SpanExporter, logs_endpoint: str | None) -> Any | None:
    """Build a log exporter matching a custom span exporter, if we know how."""
    from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

    if not isinstance(span_exporter, AWSSigV4OTLPExporter):
        logger.warning(
            "register(): custom exporter %s has no matching log exporter; "
            "evaluation events will not be exported. Pass log_exporter= to enable them.",
            type(span_exporter).__name__,
        )
        return None

    # SigV4 exporters are HTTP-only, so no protocol is passed.
    resolved = _resolve_logs_endpoint(span_exporter.endpoint, logs_endpoint)
    if resolved is None:
        return None
    if not logs_endpoint and not os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"):
        logger.info(
            "register(): SigV4 logs endpoint derived from the traces pipeline: %s. "
            "AWS OSIS needs a logs pipeline; pass its URL as logs_endpoint= if the "
            "traces pipeline does not accept /v1/logs.",
            resolved,
        )
    logger.info("OpenSearch AI evaluation events: logs endpoint=%s (SigV4)", resolved)
    return span_exporter.create_log_exporter(endpoint=resolved)


def _create_log_exporter(endpoint: str, protocol: str | None, headers: dict | None) -> Any:
    """Create a plain OTLP log exporter based on protocol and endpoint."""
    if _infer_protocol(endpoint, protocol) == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as GRPCLogExporter,
        )

        parsed = urlparse(endpoint)
        scheme = parsed.scheme.lower()
        insecure = scheme != "grpcs" and scheme != "https"
        return GRPCLogExporter(
            endpoint=parsed.netloc or endpoint, insecure=insecure, headers=headers
        )

    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    return OTLPLogExporter(endpoint=endpoint, headers=headers)


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

    exporter: SpanExporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    return exporter


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
    exporter: SpanExporter = GRPCSpanExporter(
        endpoint=grpc_endpoint,
        insecure=insecure,
        headers=headers,
    )
    return exporter


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
