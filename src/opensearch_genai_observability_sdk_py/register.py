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

Endpoints follow the OTel exporter spec, one rule per signal (traces, logs):

- ``endpoint=`` is always the OTLP base URL for all signals, like
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` (for example ``http://host:4318``).
- ``traces_endpoint=`` / ``logs_endpoint=`` are full URLs, like
  ``OTEL_EXPORTER_OTLP_<SIGNAL>_ENDPOINT``. Pass both or neither
  (``log_exporter=`` counts as the logs side, ``exporter=`` as the traces
  side): nothing is derived from one signal's URL for the other.

There are no port options: the port is part of each URL.

Endpoint per signal (priority order)
------------------------------------
Code arguments always win over env vars, as in the OTel exporters.

1. ``<signal>_endpoint=`` parameter: full URL, used as-is.
2. ``endpoint=`` parameter: base URL (see the base rule below).
3. ``OTEL_EXPORTER_OTLP_<SIGNAL>_ENDPOINT`` env var: full URL, used as-is.
4. ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var: base URL.
5. The default base ``http://localhost:4318`` (a local OTel Collector).

Base rule: over HTTP ``/v1/<signal>`` is appended to the base; over gRPC the
base is used as-is (gRPC has no per-signal path). A base whose path ends with
``/v1/traces`` or ``/v1/logs`` raises ``ValueError``: it is a full signal URL,
not a base. (Up to 0.2.9 ``endpoint=`` was the full traces URL.)

Protocol per signal (priority order)
------------------------------------
1. ``protocol=`` parameter: ``"http"`` or ``"grpc"``.
2. A ``grpc://`` / ``grpcs://`` scheme on that signal's URL passed in code
   (``<signal>_endpoint=`` or ``endpoint=``).
3. ``OTEL_EXPORTER_OTLP_<SIGNAL>_PROTOCOL`` env var.
4. ``OTEL_EXPORTER_OTLP_PROTOCOL`` env var.
5. That signal's URL scheme: ``grpc://`` / ``grpcs://`` is gRPC.
6. Otherwise HTTP.

``protocol="http"`` with a ``grpc://`` or ``grpcs://`` URL raises
``ValueError``, as does an HTTP protocol env var with a ``grpc://`` URL
from an env var.

Supported protocols
--------------------
- ``http://`` / ``https://``: HTTP OTLP (protobuf)
- ``grpc://``: gRPC (insecure)
- ``grpcs://``: gRPC with TLS

``http/json`` is not supported.

Setups
------
(a) Local OTel Collector (for example the observability stack): nothing to
set, or ``register(endpoint="http://localhost:4318")``.

(b) One OTLP source serving both signals under a pipeline path, for example
an OpenSearch Ingestion (OSIS) or Data Prepper ``otlp`` source with
``traces_path: /<pipeline>/v1/traces`` and ``logs_path: /<pipeline>/v1/logs``:

    register(
        endpoint="https://<host>/<pipeline>",
        sigv4={"service": "osis", "region": "us-east-1"},  # OSIS only
    )

(c) Separate trace and log pipelines (``otel_trace_source`` and
``otel_logs_source``): pass both full URLs. On Data Prepper the default ports
are 21890 (traces) and 21892 (logs); set ``path:`` and
``unframed_requests: true`` on each source:

    register(
        traces_endpoint="http://data-prepper:21890/v1/traces",
        logs_endpoint="http://data-prepper:21892/v1/logs",
    )

AWS SigV4
---------
``sigv4={"service": "osis", "region": "us-east-1"}`` (or ``sigv4=True``)
makes register() build signed exporters for both signals. SigV4 is
HTTP-only: a signal resolved to gRPC raises ``ValueError``.

Escape hatches
--------------
``exporter=`` replaces the traces exporter and ``log_exporter=`` the logs
exporter. Nothing is derived from a custom exporter. With ``exporter=`` and
no ``log_exporter=``, a plain (or SigV4) OTLP log exporter is built only if
a logs destination is configured (``logs_endpoint=``, ``endpoint=``,
``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``);
otherwise a warning is logged and evaluation events are not exported.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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

from opensearch_genai_observability_sdk_py._internal import set_agent_defaults

logger = logging.getLogger(__name__)

# OTel standard OTLP/HTTP base (a local OTel Collector): /v1/traces, /v1/logs.
DEFAULT_BASE_ENDPOINT = "http://localhost:4318"
DEFAULT_ENDPOINT = DEFAULT_BASE_ENDPOINT + "/v1/traces"
DEFAULT_LOGS_ENDPOINT = DEFAULT_BASE_ENDPOINT + "/v1/logs"

Signal = Literal["traces", "logs"]

_SIGNAL_PATH_SUFFIXES = ("/v1/traces", "/v1/logs")
_SIGV4_KEYS = frozenset({"service", "region"})

# Entry point group to discover instrumentors from.
# OpenLLMetry + official OTel instrumentors register under this group.
_INSTRUMENTOR_GROUPS = [
    "opentelemetry_instrumentor",
]


@dataclass(frozen=True)
class _SignalTarget:
    """Resolved destination of one OTLP signal."""

    url: str
    protocol: str  # "http" or "grpc"
    # True if the URL came from configuration (a parameter or env var),
    # False if it is the Data Prepper default.
    configured: bool


def register(
    *,
    endpoint: str | None = None,
    traces_endpoint: str | None = None,
    logs_endpoint: str | None = None,
    protocol: Literal["http", "grpc"] | None = None,
    project_name: str | None = None,
    service_name: str | None = None,
    service_version: str | None = None,
    batch: bool = True,
    auto_instrument: bool = True,
    exporter: SpanExporter | None = None,
    set_global: bool = True,
    headers: dict | None = None,
    sigv4: bool | dict[str, str] | None = None,
    enable_logs: bool = True,
    log_exporter: Any | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
) -> TracerProvider:
    """Configure the OTel tracing and evaluation-logs pipeline for OpenSearch.

    One-line setup that creates a TracerProvider, configures an OTLP
    exporter, sets up a LoggerProvider for evaluation events, and
    auto-discovers installed instrumentor packages.

    Endpoint per signal (traces, logs), in priority order. Code arguments
    always win over env vars:

    1. ``traces_endpoint=`` / ``logs_endpoint=`` (full URL, used as-is)
    2. ``endpoint=`` (base URL)
    3. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT``
       (full URL, used as-is)
    4. ``OTEL_EXPORTER_OTLP_ENDPOINT`` (base URL)
    5. ``http://localhost:4318`` (default base, a local OTel Collector)

    A base URL gets ``/v1/traces`` or ``/v1/logs`` appended over HTTP and is
    used as-is over gRPC. There are no port options: the port is part of
    each URL.

    Protocol per signal, in priority order:

    1. ``protocol=``
    2. A ``grpc://`` / ``grpcs://`` scheme on that signal's URL passed in code
    3. ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`` / ``OTEL_EXPORTER_OTLP_LOGS_PROTOCOL``
    4. ``OTEL_EXPORTER_OTLP_PROTOCOL``
    5. That signal's URL scheme (``grpc://`` / ``grpcs://`` is gRPC)
    6. Otherwise HTTP

    Args:
        endpoint: Base OTLP URL for all signals, same meaning as
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` (for example
            ``http://collector:4318``, ``https://<host>/<pipeline>`` or
            ``grpc://collector:4317``). A path ending with ``/v1/traces`` or
            ``/v1/logs`` raises ``ValueError``: pass the base, or
            ``traces_endpoint=`` and ``logs_endpoint=``.
        traces_endpoint: Full OTLP traces URL. Overrides the base for traces.
            Pass it together with ``logs_endpoint=`` (or ``log_exporter=``).
        logs_endpoint: Full OTLP logs URL (evaluation events), for example
            the OSIS logs pipeline. Overrides the base for logs. Pass it
            together with ``traces_endpoint=`` (or ``exporter=``).
        protocol: Force ``"http"`` or ``"grpc"`` for both signals. When
            omitted, resolved per signal from the env vars, then the URL
            scheme. Raises ``ValueError`` if it contradicts a ``grpc://`` or
            ``grpcs://`` URL.
        project_name: Project/service name attached to all spans.
            Defaults to ``OTEL_SERVICE_NAME`` or ``OPENSEARCH_PROJECT``
            env var, or ``"default"``.
        service_name: Alias for ``project_name``.
        service_version: Version string for the service (e.g. ``"1.0.0"``).
            Sets the ``service.version`` resource attribute. When omitted,
            resolved from ``OTEL_SERVICE_VERSION`` env var.
        batch: Use batch processors (``True``) or simple processors
            (``False``). Batch is recommended for production.
        auto_instrument: Discover and activate installed instrumentor
            packages automatically.
        exporter: Custom traces ``SpanExporter``, used as-is. The traces
            endpoint, protocol, headers and ``sigv4`` do not apply to it.
        set_global: Set as the global ``TracerProvider`` (default: ``True``).
        headers: Additional headers for the OTLP exporters register() builds
            (both signals). Over gRPC they are sent as metadata with the names
            lowercased. When omitted, the exporters read
            ``OTEL_EXPORTER_OTLP_HEADERS``.
        sigv4: Sign the exporters register() builds with AWS SigV4.
            ``True`` uses service ``"osis"`` and the botocore region; a dict
            may set ``"service"`` (``"osis"`` or ``"es"``) and ``"region"``.
            HTTP only.
        enable_logs: Also configure a ``LoggerProvider`` (OTLP logs) so that
            evaluation events are exported (default: ``True``).
            Only done when ``set_global`` is ``True`` and no global
            ``LoggerProvider`` is configured yet; an existing one is reused
            as-is.
        log_exporter: Custom ``LogRecordExporter``, used as-is. The logs
            endpoint, protocol, headers and ``sigv4`` do not apply to it.
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
        ValueError: If a base URL ends with ``/v1/traces`` or ``/v1/logs``,
            only one of ``traces_endpoint=`` / ``logs_endpoint=`` is passed,
            a protocol contradicts its URL scheme, a protocol value is
            unsupported, ``sigv4`` is used with gRPC, or ``sigv4`` has
            unknown keys.

    Examples:
        # (a) Local OTel Collector or observability stack: zero config
        register()  # same as register(endpoint="http://localhost:4318")

        # gRPC: the same host:port for both signals
        register(endpoint="grpc://collector:4317")

        # (b) One OSIS / Data Prepper ``otlp`` source for both signals
        register(
            endpoint="https://<host>/<pipeline>",
            sigv4={"service": "osis", "region": "us-east-1"},
        )

        # (c) Separate trace and log pipelines: both full URLs
        register(
            traces_endpoint="https://traces-pipe.us-east-1.osis.amazonaws.com/v1/traces",
            logs_endpoint="https://logs-pipe.us-east-1.osis.amazonaws.com/v1/logs",
            sigv4={"service": "osis", "region": "us-east-1"},
        )
    """
    set_agent_defaults(agent_name, agent_version)
    if endpoint:
        _check_base(endpoint, "endpoint=")
    _check_signal_pair(
        traces_endpoint,
        logs_endpoint,
        has_exporter=exporter is not None,
        logs_needed=enable_logs and set_global and log_exporter is None,
    )
    sigv4_opts = _sigv4_options(sigv4)

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
    traces_url = "custom exporter"
    if exporter is None:
        target = _resolve_signal("traces", traces_endpoint, endpoint, protocol)
        traces_url = target.url
        exporter = _create_exporter(target, headers, sigv4_opts)

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

    # Step 6: Logs pipeline for evaluation events (score(), Benchmark, evaluate())
    if enable_logs and set_global:
        _setup_logs(
            provider=provider,
            resource=resource,
            batch=batch,
            custom_exporter=custom_exporter,
            endpoint=endpoint,
            logs_endpoint=logs_endpoint,
            protocol=protocol,
            headers=headers,
            sigv4=sigv4_opts,
            log_exporter=log_exporter,
        )

    # Step 7: Auto-instrument installed libraries
    if auto_instrument:
        _auto_instrument(provider)

    logger.info(
        "OpenSearch AI tracing initialized: endpoint=%s project=%s",
        traces_url,
        name,
    )

    return provider


def _check_base(base: str, source: str) -> None:
    """Reject a base URL that is really a full signal URL."""
    path = urlparse(base).path.rstrip("/")
    for suffix in _SIGNAL_PATH_SUFFIXES:
        if path.endswith(suffix):
            if source == "endpoint=":
                fix = "pass traces_endpoint= and logs_endpoint="
            else:
                fix = "set OTEL_EXPORTER_OTLP_TRACES_ENDPOINT and OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"
            raise ValueError(
                f"{source} is the OTLP base URL (e.g. http://host:4318). "
                f"Remove '{suffix}', or {fix}. Got {base!r}."
            )


def _check_signal_pair(
    traces_endpoint: str | None,
    logs_endpoint: str | None,
    *,
    has_exporter: bool,
    logs_needed: bool,
) -> None:
    """``traces_endpoint=`` and ``logs_endpoint=`` go together.

    Nothing is derived from one signal's URL for the other, so a lone full
    URL would silently send the other signal to the base. ``exporter=``
    stands in for the traces side and ``log_exporter=`` (or no logs
    pipeline) for the logs side.
    """
    if traces_endpoint and not logs_endpoint and logs_needed:
        missing, given = "logs_endpoint=", "traces_endpoint="
    elif logs_endpoint and not traces_endpoint and not has_exporter:
        missing, given = "traces_endpoint=", "logs_endpoint="
    else:
        return
    raise ValueError(
        f"{given} was passed without {missing}: pass traces_endpoint= and logs_endpoint= "
        "together (full URLs), or endpoint= (the OTLP base URL) for both signals."
    )


def _sigv4_options(sigv4: bool | dict[str, str] | None) -> dict[str, str] | None:
    """Normalize ``sigv4=`` to exporter keyword arguments, or ``None`` for unsigned."""
    if sigv4 is None or sigv4 is False:
        return None
    if sigv4 is True:
        return {}
    if not isinstance(sigv4, dict):
        raise ValueError("sigv4= must be True or a dict with 'service' and/or 'region'.")
    unknown = set(sigv4) - _SIGV4_KEYS
    if unknown:
        raise ValueError(
            f"Unknown sigv4= keys {sorted(unknown)}. Supported keys: 'service', 'region'."
        )
    return {k: str(v) for k, v in sigv4.items() if v}


def _env_protocol(signal: Signal) -> str | None:
    """The protocol env var for one signal, if set."""
    return os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_PROTOCOL") or os.environ.get(
        "OTEL_EXPORTER_OTLP_PROTOCOL"
    )


def _resolve_protocol(signal: Signal, url: str, protocol: str | None, from_code: bool) -> str:
    """Resolve one signal's protocol (see the module docstring)."""
    if protocol is not None:
        return _infer_protocol(url, protocol)
    if from_code and _is_grpc_url(url):
        return "grpc"
    return _infer_protocol(url, _env_protocol(signal))


def _resolve_signal(
    signal: Signal,
    signal_endpoint: str | None,
    endpoint: str | None,
    protocol: str | None,
) -> _SignalTarget:
    """Resolve one signal's URL and protocol (see the module docstring).

    Args:
        signal: ``"traces"`` or ``"logs"``.
        signal_endpoint: The ``<signal>_endpoint=`` parameter (full URL).
        endpoint: The ``endpoint=`` parameter (base URL).
        protocol: The ``protocol=`` parameter.
    """
    # Code arguments first, then env vars (same order as the OTel exporters).
    if signal_endpoint:
        full, base, from_code = signal_endpoint, None, True
    elif endpoint:
        full, base, from_code = None, endpoint, True
    else:
        from_code = False
        full = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT") or None
        base = None if full else (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None)

    if full:
        return _SignalTarget(full, _resolve_protocol(signal, full, protocol, from_code), True)

    if base:
        _check_base(base, "endpoint=" if from_code else "OTEL_EXPORTER_OTLP_ENDPOINT")
    resolved_base = base or DEFAULT_BASE_ENDPOINT
    resolved_protocol = _resolve_protocol(signal, resolved_base, protocol, from_code)
    if resolved_protocol == "grpc":
        url = resolved_base
    else:
        url = resolved_base.rstrip("/") + f"/v1/{signal}"
    return _SignalTarget(url, resolved_protocol, bool(base))


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
    custom_exporter: bool,
    endpoint: str | None,
    logs_endpoint: str | None,
    protocol: str | None,
    headers: dict | None,
    sigv4: dict[str, str] | None,
    log_exporter: Any | None,
) -> LoggerProvider | None:
    """Create and install the global LoggerProvider used for evaluation events."""
    if _global_logger_provider_is_set():
        logger.info(
            "A global LoggerProvider is already configured; evaluation events use it as-is."
        )
        return None

    if log_exporter is None:
        target = _resolve_signal("logs", logs_endpoint, endpoint, protocol)
        if custom_exporter and not target.configured:
            logger.warning(
                "register(): a custom exporter= was passed and no logs destination is "
                "configured, so evaluation events will not be exported. Pass log_exporter=, "
                "logs_endpoint= or endpoint=, or set OTEL_EXPORTER_OTLP_LOGS_ENDPOINT."
            )
            return None
        log_exporter = _create_log_exporter(target, headers, sigv4)
        logger.info(
            "OpenSearch AI evaluation events: logs endpoint=%s%s",
            target.url,
            " (SigV4)" if sigv4 is not None else "",
        )

    logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
    if batch:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    else:
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    _logs.set_logger_provider(logger_provider)
    provider.add_span_processor(_LoggerProviderLifecycle(logger_provider))
    return logger_provider


def _require_http_for_sigv4(target: _SignalTarget, signal: Signal) -> None:
    if target.protocol == "grpc":
        raise ValueError(
            f"sigv4= signs HTTP requests only, but the {signal} endpoint '{target.url}' "
            "resolved to gRPC. Use an https:// endpoint (AWS OSIS has no gRPC endpoint)."
        )


def _grpc_target(url: str) -> tuple[str, bool]:
    """``(host:port, insecure)`` for a gRPC exporter from a URL."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    return parsed.netloc or url, scheme not in ("grpcs", "https")


def _grpc_headers(headers: dict | None) -> dict | None:
    """Headers as gRPC metadata: names lowercased (gRPC rejects uppercase keys)."""
    if not headers:
        return None
    return {str(k).lower(): v for k, v in headers.items()}


def _create_log_exporter(
    target: _SignalTarget, headers: dict | None, sigv4: dict[str, str] | None = None
) -> Any:
    """Create the OTLP log exporter for a resolved logs target."""
    if sigv4 is not None:
        _require_http_for_sigv4(target, "logs")
        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPLogExporter

        return AWSSigV4OTLPLogExporter(endpoint=target.url, headers=headers, **sigv4)

    if target.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as GRPCLogExporter,
        )

        grpc_endpoint, insecure = _grpc_target(target.url)
        return GRPCLogExporter(
            endpoint=grpc_endpoint, insecure=insecure, headers=_grpc_headers(headers)
        )

    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    return OTLPLogExporter(endpoint=target.url, headers=headers)


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


def _is_grpc_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in ("grpc", "grpcs")


def _infer_protocol(endpoint: str, protocol: str | None) -> str:
    """Determine the OTLP transport protocol from explicit setting or URL scheme.

    Args:
        endpoint: The endpoint URL.
        protocol: Explicit protocol override (``'http'``, ``'grpc'``,
            ``'http/protobuf'``, etc.) or ``None`` to infer from URL.

    Returns:
        ``'http'`` or ``'grpc'``.

    Raises:
        ValueError: If ``protocol`` contradicts a ``grpc://`` or ``grpcs://``
            URL scheme, or if the protocol value is unsupported.
    """
    scheme = urlparse(endpoint).scheme.lower()
    url_implies_grpc = _is_grpc_url(endpoint)

    if protocol is not None:
        resolved = _normalize_protocol(protocol)
        if url_implies_grpc and resolved == "http":
            raise ValueError(
                f"Conflicting protocol: endpoint '{endpoint}' (scheme '{scheme}://') implies "
                f"gRPC but protocol '{protocol}' was specified. "
                f"Either use protocol 'grpc' or change the endpoint scheme to "
                f"'http://' or 'https://'."
            )
        return resolved

    if url_implies_grpc:
        return "grpc"

    return "http"


def _create_exporter(
    target: _SignalTarget, headers: dict | None, sigv4: dict[str, str] | None = None
) -> SpanExporter:
    """Create the OTLP span exporter for a resolved traces target."""
    if sigv4 is not None:
        _require_http_for_sigv4(target, "traces")
        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

        signed: SpanExporter = AWSSigV4OTLPExporter(endpoint=target.url, headers=headers, **sigv4)
        return signed

    if target.protocol == "grpc":
        return _create_grpc_exporter(target.url, headers)

    return _create_http_exporter(target.url, headers)


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

    # gRPC exporter takes host:port, not a full URL
    grpc_endpoint, insecure = _grpc_target(endpoint)

    logger.info("Using gRPC exporter: %s (insecure=%s)", grpc_endpoint, insecure)
    exporter: SpanExporter = GRPCSpanExporter(
        endpoint=grpc_endpoint,
        insecure=insecure,
        headers=_grpc_headers(headers),
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
