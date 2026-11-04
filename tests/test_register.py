# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.register."""

import importlib
import warnings
from unittest.mock import MagicMock, patch

import pytest

from opensearch_genai_observability_sdk_py.register import _infer_protocol

# importlib.import_module is needed because opensearch_genai_observability_sdk_py.__init__ imports
# `register` (the function) into the package namespace, shadowing the submodule name.
# Using importlib bypasses that and returns the actual module object.
_register_mod = importlib.import_module("opensearch_genai_observability_sdk_py.register")


# ---------------------------------------------------------------------------
# _infer_protocol
# ---------------------------------------------------------------------------


class TestInferProtocol:
    """Unit tests for protocol inference and validation."""

    def test_http_scheme(self):
        assert _infer_protocol("http://localhost:4318/v1/traces", None) == "http"

    def test_https_scheme(self):
        assert _infer_protocol("https://collector.example.com/v1/traces", None) == "http"

    def test_grpc_scheme(self):
        assert _infer_protocol("grpc://localhost:4317", None) == "grpc"

    def test_grpcs_scheme(self):
        assert _infer_protocol("grpcs://collector.example.com:4317", None) == "grpc"

    def test_explicit_grpc_overrides_http_scheme(self):
        """http:// is ambiguous — explicit grpc wins."""
        assert _infer_protocol("http://localhost:4317", "grpc") == "grpc"

    def test_explicit_http_protobuf_accepted(self):
        assert _infer_protocol("http://localhost:4318/v1/traces", "http/protobuf") == "http"

    def test_contradicting_http_on_grpc_scheme_raises(self):
        """grpc:// unambiguously means gRPC — contradicting it is an error."""
        with pytest.raises(ValueError, match="Conflicting protocol"):
            _infer_protocol("grpc://localhost:4317", "http")

    def test_contradicting_http_protobuf_on_grpcs_scheme_raises(self):
        with pytest.raises(ValueError, match="Conflicting protocol"):
            _infer_protocol("grpcs://localhost:4317", "http/protobuf")

    def test_http_json_raises(self):
        with pytest.raises(ValueError, match="http/json.*not supported"):
            _infer_protocol("http://localhost:4318/v1/traces", "http/json")

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown OTLP protocol"):
            _infer_protocol("http://localhost:4318/v1/traces", "thrift")


# ---------------------------------------------------------------------------
# _resolve_signal: one rule per signal (OTel exporter spec)
# ---------------------------------------------------------------------------

_OTLP_ENV_VARS = (
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
)


@pytest.fixture()
def clean_otlp_env(monkeypatch):
    for var in _OTLP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _resolve(signal, signal_endpoint=None, base=None, protocol=None):
    target = _register_mod._resolve_signal(signal, signal_endpoint, base, protocol)
    return target.url, target.protocol


class TestResolveSignal:
    """Endpoint and protocol resolution per signal."""

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_signal_param_wins(self, signal, clean_otlp_env, monkeypatch):
        monkeypatch.setenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT", "http://env:1/x")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base-env:4318")
        assert _resolve(signal, "http://p:1/full", "http://base:4318") == (
            "http://p:1/full",
            "http",
        )

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_signal_env_beats_base(self, signal, clean_otlp_env, monkeypatch):
        monkeypatch.setenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT", "http://env:1/x")
        assert _resolve(signal, None, "http://base:4318")[0] == "http://env:1/x"

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_base_param_appends_signal_path(self, signal, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base-env:4318")
        assert _resolve(signal, None, "http://base:4318/")[0] == f"http://base:4318/v1/{signal}"

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_base_env_appends_signal_path(self, signal, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert _resolve(signal)[0] == f"http://localhost:4318/v1/{signal}"

    def test_base_keeps_path_prefix(self, clean_otlp_env):
        assert _resolve("logs", None, "https://h/opentelemetry")[0] == (
            "https://h/opentelemetry/v1/logs"
        )

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_data_prepper_default(self, signal, clean_otlp_env):
        target = _register_mod._resolve_signal(signal, None, None, None)
        assert target.url == f"http://localhost:21890/opentelemetry/v1/{signal}"
        assert target.configured is False
        assert _register_mod.DEFAULT_ENDPOINT == _resolve("traces")[0]
        assert _register_mod.DEFAULT_LOGS_ENDPOINT == _resolve("logs")[0]

    def test_configured_flag(self, clean_otlp_env, monkeypatch):
        resolve = _register_mod._resolve_signal
        assert resolve("logs", "http://l/v1/logs", None, None).configured
        assert resolve("logs", None, "http://b:4318", None).configured
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://b:4318")
        assert resolve("logs", None, None, None).configured

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_grpc_base_used_as_is_for_every_signal(self, signal, clean_otlp_env):
        assert _resolve(signal, None, "grpc://collector:4317") == (
            "grpc://collector:4317",
            "grpc",
        )

    @pytest.mark.parametrize("signal", ["traces", "logs"])
    def test_grpc_env_base_with_protocol_env(self, signal, clean_otlp_env, monkeypatch):
        """The standard collector setup: base env + OTEL_EXPORTER_OTLP_PROTOCOL=grpc."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        assert _resolve(signal) == ("http://collector:4317", "grpc")

    def test_protocol_param_beats_env(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf")
        assert _resolve("logs", None, "http://c:4317", "grpc") == ("http://c:4317", "grpc")

    def test_signal_protocol_env_beats_general_env(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        assert _resolve("traces", None, "http://c:4317") == ("http://c:4317", "grpc")
        assert _resolve("logs", None, "http://c:4318") == ("http://c:4318/v1/logs", "http")

    def test_logs_protocol_checked_against_logs_url(self, clean_otlp_env, monkeypatch):
        """OTEL_EXPORTER_OTLP_LOGS_PROTOCOL applies to the logs URL, not the traces URL."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf")
        assert _resolve("traces", None, "grpc://c:4317") == ("grpc://c:4317", "grpc")
        assert _resolve("logs", "http://c:4318/v1/logs", "grpc://c:4317") == (
            "http://c:4318/v1/logs",
            "http",
        )

    def test_logs_protocol_contradicting_logs_url_raises(self, clean_otlp_env, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf")
        with pytest.raises(ValueError, match="grpc://c:4317"):
            _resolve("logs", None, "grpc://c:4317")


class TestLegacyEndpoint:
    """0.2.9 ``endpoint=`` was a full traces URL: deprecated, still accepted."""

    def test_base_passes_through(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _register_mod._base_from_legacy_endpoint("http://c:4318") == "http://c:4318"
            assert _register_mod._base_from_legacy_endpoint(None) is None

    @pytest.mark.parametrize(
        ("endpoint", "base"),
        [
            ("http://c:4318/v1/traces", "http://c:4318"),
            ("http://c:4318/v1/traces/", "http://c:4318"),
            (
                "http://localhost:21890/opentelemetry/v1/traces",
                "http://localhost:21890/opentelemetry",
            ),
        ],
    )
    def test_traces_suffix_is_removed_with_warning(self, endpoint, base):
        with pytest.warns(DeprecationWarning, match="traces_endpoint="):
            assert _register_mod._base_from_legacy_endpoint(endpoint) == base


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestRegister:
    """Tests for the register() function."""

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_returns_tracer_provider(self, mock_create_http):
        from opentelemetry.sdk.trace import TracerProvider

        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        provider = register(set_global=False, auto_instrument=False)
        assert isinstance(provider, TracerProvider)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_default_endpoint(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import (
            DEFAULT_ENDPOINT,
            register,
        )

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_create_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        mock_create_http.assert_called_once_with(DEFAULT_ENDPOINT, None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_provided_endpoint(self, mock_create_http):
        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        register(
            endpoint="http://my-collector:4318",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_http.assert_called_once_with("http://my-collector:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_legacy_traces_url_still_works(self, mock_create_http):
        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        with pytest.warns(DeprecationWarning):
            register(
                endpoint="http://my-collector:4318/v1/traces",
                set_global=False,
                auto_instrument=False,
            )
        mock_create_http.assert_called_once_with("http://my-collector:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_traces_endpoint_param(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://env:4318/v1/traces")
        mock_create_http.return_value = MagicMock()
        register(
            endpoint="http://base:4318",
            traces_endpoint="http://t:4318/custom",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_http.assert_called_once_with("http://t:4318/custom", None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_otel_traces_endpoint_env_var(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://otel:4318/v1/traces")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_create_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        mock_create_http.assert_called_once_with("http://otel:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_uses_otel_base_endpoint_env_var(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        mock_create_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        mock_create_http.assert_called_once_with("http://localhost:4318/v1/traces", None)

    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_register_uses_grpc_for_grpc_scheme(self, mock_create_grpc):
        from opensearch_genai_observability_sdk_py.register import register

        mock_create_grpc.return_value = MagicMock()
        register(
            endpoint="grpc://localhost:4317",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_grpc.assert_called_once()

    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_register_uses_grpc_via_protocol_env_var(self, mock_create_grpc, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)
        mock_create_grpc.return_value = MagicMock()
        register(
            endpoint="http://localhost:4317",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_grpc.assert_called_once()

    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_register_traces_protocol_env_var_takes_priority(self, mock_create_grpc, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        mock_create_grpc.return_value = MagicMock()
        register(
            endpoint="http://localhost:4317",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_grpc.assert_called_once()

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_sets_service_version(self, mock_create_http):
        from opentelemetry.sdk.trace import TracerProvider

        from opensearch_genai_observability_sdk_py.register import register

        mock_create_http.return_value = MagicMock()
        provider = register(service_version="1.0.0", set_global=False, auto_instrument=False)
        assert isinstance(provider, TracerProvider)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.version"] == "1.0.0"

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_service_version_from_env(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_SERVICE_VERSION", "2.0.0")
        mock_create_http.return_value = MagicMock()
        provider = register(set_global=False, auto_instrument=False)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.version"] == "2.0.0"

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_service_version_param_over_env(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_SERVICE_VERSION", "2.0.0")
        mock_create_http.return_value = MagicMock()
        provider = register(service_version="3.0.0", set_global=False, auto_instrument=False)
        resource_attrs = dict(provider.resource.attributes)
        assert resource_attrs["service.version"] == "3.0.0"

    def test_register_uses_custom_exporter(self):
        from opensearch_genai_observability_sdk_py.register import register

        custom_exporter = MagicMock()
        register(exporter=custom_exporter, set_global=False, auto_instrument=False)
        custom_exporter.export.assert_not_called()  # accepted but not called

    @patch.object(_register_mod, "_create_http_exporter")
    def test_register_passes_headers(self, mock_create_http, monkeypatch):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mock_create_http.return_value = MagicMock()
        register(
            headers={"X-Custom": "value"},
            set_global=False,
            auto_instrument=False,
        )
        mock_create_http.assert_called_once_with(
            _register_mod.DEFAULT_ENDPOINT, {"X-Custom": "value"}
        )


# ---------------------------------------------------------------------------
# Logs pipeline for evaluation events
# ---------------------------------------------------------------------------


def _in_memory_log_exporter():
    try:
        from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter as InMemoryLogs
    except ImportError:  # pragma: no cover
        from opentelemetry.sdk._logs.export import InMemoryLogExporter as InMemoryLogs
    return InMemoryLogs()


class _Spy:
    """Wraps a function and records what it returned."""

    def __init__(self, fn):
        self.fn = fn
        self.results = []

    def __call__(self, *args, **kwargs):
        result = self.fn(*args, **kwargs)
        self.results.append(result)
        return result


def _log_target(mock_log):
    """The resolved logs target passed to the patched _create_log_exporter."""
    return mock_log.call_args.args[0]


class TestCreateLogExporter:
    def test_http(self):
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        target = _register_mod._SignalTarget("http://localhost:4318/v1/logs", "http", True)
        exp = _register_mod._create_log_exporter(target, None)
        assert isinstance(exp, OTLPLogExporter)
        assert exp._endpoint == "http://localhost:4318/v1/logs"

    def test_grpc(self):
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        target = _register_mod._SignalTarget("grpc://localhost:4317", "grpc", True)
        exp = _register_mod._create_log_exporter(target, None)
        assert isinstance(exp, OTLPLogExporter)
        exp.shutdown()

    def test_grpc_headers_lowercased(self):
        """gRPC rejects uppercase metadata keys ("Invalid metadata")."""
        assert _register_mod._grpc_headers({"Authorization": "Basic x", "X-Tenant": "t"}) == {
            "authorization": "Basic x",
            "x-tenant": "t",
        }
        assert _register_mod._grpc_headers(None) is None
        assert _register_mod._grpc_headers({}) is None

    def test_grpc_exporters_get_lowercase_headers(self):
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        target = _register_mod._SignalTarget("grpc://localhost:4317", "grpc", True)
        with patch.object(OTLPLogExporter, "__init__", return_value=None) as init:
            _register_mod._create_log_exporter(target, {"Authorization": "x"})
        assert init.call_args.kwargs["headers"] == {"authorization": "x"}
        with patch(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
        ) as span_cls:
            _register_mod._create_grpc_exporter("grpc://localhost:4317", {"Authorization": "x"})
        assert span_cls.call_args.kwargs["headers"] == {"authorization": "x"}

    def test_grpc_target(self):
        assert _register_mod._grpc_target("grpc://h:4317") == ("h:4317", True)
        assert _register_mod._grpc_target("grpcs://h:4317") == ("h:4317", False)
        assert _register_mod._grpc_target("https://h:4317") == ("h:4317", False)
        assert _register_mod._grpc_target("http://h:4317") == ("h:4317", True)


@pytest.fixture()
def fresh_globals():
    """Pretend no global LoggerProvider is set and capture what register() installs."""
    with (
        patch.object(_register_mod, "_global_logger_provider_is_set", return_value=False),
        patch.object(_register_mod._logs, "set_logger_provider") as set_lp,
        patch.object(_register_mod.trace, "set_tracer_provider"),
    ):
        yield set_lp


class TestRegisterLogs:
    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_one_base_for_traces_and_logs(self, mock_http, mock_log, fresh_globals, clean_otlp_env):
        from opentelemetry.sdk._logs import LoggerProvider

        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(
            endpoint="http://my-collector:4318",
            headers={"Authorization": "Basic x"},
            auto_instrument=False,
        )
        mock_http.assert_called_once_with(
            "http://my-collector:4318/v1/traces", {"Authorization": "Basic x"}
        )
        target = _log_target(mock_log)
        assert (target.url, target.protocol) == ("http://my-collector:4318/v1/logs", "http")
        assert mock_log.call_args.args[1] == {"Authorization": "Basic x"}
        fresh_globals.assert_called_once()
        assert isinstance(fresh_globals.call_args.args[0], LoggerProvider)

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_legacy_traces_url_logs_go_next_to_it(
        self, mock_http, mock_log, fresh_globals, clean_otlp_env
    ):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        with pytest.warns(DeprecationWarning):
            register(endpoint="http://c:4318/v1/traces", auto_instrument=False)
        mock_http.assert_called_once_with("http://c:4318/v1/traces", None)
        assert _log_target(mock_log).url == "http://c:4318/v1/logs"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"endpoint": "grpc://collector:4317"},
            {"endpoint": "http://collector:4317", "protocol": "grpc"},
        ],
    )
    def test_grpc_logs_use_the_traces_host_port(self, kwargs, fresh_globals, clean_otlp_env):
        """gRPC has no per-signal path: logs go to the same host:port as traces."""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as GRPCLogExporter,
        )

        from opensearch_genai_observability_sdk_py.register import register

        spy = _Spy(_register_mod._create_log_exporter)
        with (
            patch.object(_register_mod, "_create_grpc_exporter") as mock_grpc,
            patch.object(_register_mod, "_create_log_exporter", side_effect=spy) as mock_log,
        ):
            mock_grpc.return_value = MagicMock()
            register(auto_instrument=False, **kwargs)
        traces_url = mock_grpc.call_args.args[0]
        target = _log_target(mock_log)
        assert (target.url, target.protocol) == (traces_url, "grpc")
        assert isinstance(spy.results[0], GRPCLogExporter)
        fresh_globals.assert_called_once()
        spy.results[0].shutdown()

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_grpc_env_collector_setup(
        self, mock_grpc, mock_log, fresh_globals, clean_otlp_env, monkeypatch
    ):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        mock_grpc.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(auto_instrument=False)
        mock_grpc.assert_called_once_with("http://collector:4317", None)
        target = _log_target(mock_log)
        assert (target.url, target.protocol) == ("http://collector:4317", "grpc")

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_grpc_exporter")
    def test_grpc_traces_http_logs(
        self, mock_grpc, mock_log, fresh_globals, clean_otlp_env, monkeypatch
    ):
        """A logs protocol env var is checked against the logs URL, not the traces URL."""
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf")
        mock_grpc.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(
            endpoint="grpc://collector:4317",
            logs_endpoint="http://collector:4318/v1/logs",
            auto_instrument=False,
        )
        target = _log_target(mock_log)
        assert (target.url, target.protocol) == ("http://collector:4318/v1/logs", "http")

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_logs_endpoint_param(self, mock_http, mock_log, fresh_globals, clean_otlp_env):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(logs_endpoint="http://logs:4318/v1/logs", auto_instrument=False)
        mock_http.assert_called_once_with(_register_mod.DEFAULT_ENDPOINT, None)
        assert _log_target(mock_log).url == "http://logs:4318/v1/logs"

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_separate_traces_and_logs_endpoints(
        self, mock_http, mock_log, fresh_globals, clean_otlp_env
    ):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(
            traces_endpoint="http://traces-host:4318/ingest",
            logs_endpoint="http://logs-host:4318/v1/logs",
            auto_instrument=False,
        )
        mock_http.assert_called_once_with("http://traces-host:4318/ingest", None)
        assert _log_target(mock_log).url == "http://logs-host:4318/v1/logs"

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_signal_env_endpoints(
        self, mock_http, mock_log, fresh_globals, clean_otlp_env, monkeypatch
    ):
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://t:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://l:4318/v1/logs")
        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(auto_instrument=False)
        mock_http.assert_called_once_with("http://t:4318/v1/traces", None)
        assert _log_target(mock_log).url == "http://l:4318/v1/logs"

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_traces_env_does_not_drive_logs(
        self, mock_http, mock_log, fresh_globals, clean_otlp_env, monkeypatch
    ):
        """No derivation from the traces URL: logs fall back to the base."""
        from opensearch_genai_observability_sdk_py.register import register

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://t:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318")
        mock_http.return_value = MagicMock()
        mock_log.return_value = _in_memory_log_exporter()
        register(auto_instrument=False)
        assert _log_target(mock_log).url == "http://base:4318/v1/logs"

    @patch.object(_register_mod, "_create_http_exporter")
    def test_events_flow_and_flush_through_tracer_provider(self, mock_http, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        log_exp = _in_memory_log_exporter()
        provider = register(log_exporter=log_exp, batch=True, auto_instrument=False)
        logger_provider = fresh_globals.call_args.args[0]

        logger_provider.get_logger("t").emit(
            event_name="gen_ai.evaluation.result", attributes={"gen_ai.evaluation.name": "x"}
        )
        assert provider.force_flush()
        records = log_exp.get_finished_logs()
        assert [r.log_record.event_name for r in records] == ["gen_ai.evaluation.result"]
        # Same resource as traces. ReadableLogRecord.resource (>=1.39) or
        # LogData.log_record.resource (1.38).
        resource = getattr(records[0], "resource", None) or records[0].log_record.resource
        assert resource.attributes["service.name"] == provider.resource.attributes["service.name"]

        with patch.object(logger_provider, "shutdown") as lp_shutdown:
            provider.shutdown()
        lp_shutdown.assert_called_once()

    @patch.object(_register_mod, "_create_http_exporter")
    def test_simple_processor_when_batch_false(self, mock_http, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        log_exp = _in_memory_log_exporter()
        with patch.object(
            _register_mod,
            "SimpleLogRecordProcessor",
            wraps=_register_mod.SimpleLogRecordProcessor,
        ) as simple:
            register(log_exporter=log_exp, batch=False, auto_instrument=False)
        simple.assert_called_once_with(log_exp)

    @patch.object(_register_mod, "_create_http_exporter")
    def test_existing_global_logger_provider_untouched(self, mock_http):
        """conftest installs a global LoggerProvider: register() must reuse it."""
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        with (
            patch.object(_register_mod._logs, "set_logger_provider") as set_lp,
            patch.object(_register_mod.trace, "set_tracer_provider"),
            patch.object(_register_mod, "_create_log_exporter") as mock_log,
        ):
            register(auto_instrument=False)
        set_lp.assert_not_called()
        mock_log.assert_not_called()

    def test_global_logger_provider_detection(self):
        # The test session has an SDK LoggerProvider installed (conftest).
        assert _register_mod._global_logger_provider_is_set() is True
        from opentelemetry._logs._internal import ProxyLoggerProvider

        with patch.object(
            _register_mod._logs, "get_logger_provider", return_value=ProxyLoggerProvider()
        ):
            assert _register_mod._global_logger_provider_is_set() is False

    @patch.object(_register_mod, "_create_http_exporter")
    def test_enable_logs_false(self, mock_http, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        register(enable_logs=False, auto_instrument=False)
        fresh_globals.assert_not_called()

    @patch.object(_register_mod, "_create_log_exporter")
    @patch.object(_register_mod, "_create_http_exporter")
    def test_set_global_false_skips_logs(self, mock_http, mock_log, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        mock_http.return_value = MagicMock()
        register(set_global=False, auto_instrument=False)
        fresh_globals.assert_not_called()
        mock_log.assert_not_called()

    def test_custom_exporter_without_logs_destination_warns(
        self, fresh_globals, clean_otlp_env, caplog
    ):
        from opensearch_genai_observability_sdk_py.register import register

        register(exporter=MagicMock(), auto_instrument=False)
        fresh_globals.assert_not_called()
        assert "log_exporter=" in caplog.text

    @patch.object(_register_mod, "_create_log_exporter")
    def test_custom_exporter_with_logs_endpoint(self, mock_log, fresh_globals, clean_otlp_env):
        """No derivation from a custom exporter, but a configured logs URL is used."""
        from opensearch_genai_observability_sdk_py.register import register

        mock_log.return_value = _in_memory_log_exporter()
        register(exporter=MagicMock(), logs_endpoint="http://l:4318/v1/logs", auto_instrument=False)
        assert _log_target(mock_log).url == "http://l:4318/v1/logs"
        fresh_globals.assert_called_once()

    def test_custom_exporter_with_log_exporter(self, fresh_globals):
        from opensearch_genai_observability_sdk_py.register import register

        register(
            exporter=MagicMock(), log_exporter=_in_memory_log_exporter(), auto_instrument=False
        )
        fresh_globals.assert_called_once()


# ---------------------------------------------------------------------------
# sigv4=: register() builds signed exporters for both signals
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_aws():
    creds = MagicMock()
    session = MagicMock()
    session.get_credentials.return_value = creds
    session.get_config_variable.return_value = "us-west-2"
    with patch("botocore.session.get_session", return_value=session):
        yield


class TestSigV4:
    TRACES = "https://traces-pipe.us-east-1.osis.amazonaws.com/v1/traces"
    LOGS = "https://logs-pipe.us-east-1.osis.amazonaws.com/v1/logs"

    @pytest.fixture()
    def built(self):
        """Spy on the exporters register() builds: (span exporters, log exporters)."""
        span_spy = _Spy(_register_mod._create_exporter)
        log_spy = _Spy(_register_mod._create_log_exporter)
        with (
            patch.object(_register_mod, "_create_exporter", side_effect=span_spy),
            patch.object(_register_mod, "_create_log_exporter", side_effect=log_spy),
        ):
            yield span_spy.results, log_spy.results

    def test_osis_one_pipeline_per_signal(self, fake_aws, fresh_globals, clean_otlp_env, built):
        from opensearch_genai_observability_sdk_py.exporters import (
            AWSSigV4OTLPExporter,
            AWSSigV4OTLPLogExporter,
        )
        from opensearch_genai_observability_sdk_py.register import register

        register(
            traces_endpoint=self.TRACES,
            logs_endpoint=self.LOGS,
            sigv4={"service": "osis", "region": "us-east-1"},
            auto_instrument=False,
        )
        (span_exp,), (log_exp,) = built
        assert isinstance(span_exp, AWSSigV4OTLPExporter)
        assert isinstance(log_exp, AWSSigV4OTLPLogExporter)
        assert span_exp._endpoint == self.TRACES
        assert log_exp._endpoint == self.LOGS
        assert span_exp.signing_session._region == "us-east-1"
        assert log_exp.signing_session._service == "osis"

    def test_sigv4_true_uses_defaults_and_base(
        self, fake_aws, fresh_globals, clean_otlp_env, built
    ):
        from opensearch_genai_observability_sdk_py.register import register

        register(
            endpoint="https://pipe.us-west-2.osis.amazonaws.com",
            sigv4=True,
            auto_instrument=False,
        )
        (span_exp,), (log_exp,) = built
        assert span_exp._endpoint == "https://pipe.us-west-2.osis.amazonaws.com/v1/traces"
        assert log_exp._endpoint == "https://pipe.us-west-2.osis.amazonaws.com/v1/logs"
        assert span_exp.signing_session._service == "osis"
        assert log_exp.signing_session._region == "us-west-2"

    def test_sigv4_signs_logs_for_custom_exporter(
        self, fake_aws, fresh_globals, clean_otlp_env, built
    ):
        from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPLogExporter
        from opensearch_genai_observability_sdk_py.register import register

        register(
            exporter=MagicMock(),
            logs_endpoint=self.LOGS,
            sigv4={"service": "es"},
            auto_instrument=False,
        )
        spans, (log_exp,) = built
        assert spans == []
        assert isinstance(log_exp, AWSSigV4OTLPLogExporter)
        assert log_exp.signing_session._service == "es"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"endpoint": "grpc://c:4317"},
            {"traces_endpoint": "grpcs://c:4317"},
        ],
    )
    def test_sigv4_rejects_grpc_traces(self, kwargs, fake_aws, clean_otlp_env):
        from opensearch_genai_observability_sdk_py.register import register

        with pytest.raises(ValueError, match="sigv4= signs HTTP requests only"):
            register(sigv4=True, set_global=False, auto_instrument=False, **kwargs)

    def test_sigv4_rejects_grpc_logs(self, fake_aws, fresh_globals, clean_otlp_env):
        from opensearch_genai_observability_sdk_py.register import register

        with pytest.raises(ValueError, match="logs endpoint"):
            register(
                traces_endpoint=self.TRACES,
                logs_endpoint="grpc://c:4317",
                sigv4=True,
                auto_instrument=False,
            )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (False, None),
            (True, {}),
            ({"service": "es"}, {"service": "es"}),
            (
                {"service": "osis", "region": "eu-west-1"},
                {"service": "osis", "region": "eu-west-1"},
            ),
            ({"region": None}, {}),
        ],
    )
    def test_options(self, value, expected):
        assert _register_mod._sigv4_options(value) == expected

    @pytest.mark.parametrize("value", [{"profile": "x"}, "osis"])
    def test_bad_options(self, value):
        with pytest.raises(ValueError, match="sigv4="):
            _register_mod._sigv4_options(value)
