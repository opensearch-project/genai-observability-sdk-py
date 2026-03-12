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
from unittest.mock import MagicMock, patch

import pytest

from opensearch_genai_observability_sdk_py.register import _infer_protocol, _resolve_endpoint

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
# _resolve_endpoint
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    """Unit tests for endpoint resolution priority."""

    def test_code_endpoint_wins(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318")
        assert (
            _resolve_endpoint("http://explicit:4318/v1/traces") == "http://explicit:4318/v1/traces"
        )

    def test_traces_endpoint_env_var(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
        assert _resolve_endpoint(None) == "http://traces:4318/v1/traces"

    def test_base_endpoint_env_var_appends_path(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert _resolve_endpoint(None) == "http://localhost:4318/v1/traces"

    def test_base_endpoint_trailing_slash_handled(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/")
        assert _resolve_endpoint(None) == "http://localhost:4318/v1/traces"

    def test_traces_endpoint_takes_priority_over_base(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://base:4318")
        assert _resolve_endpoint(None) == "http://traces:4318/v1/traces"

    def test_data_prepper_default(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert _resolve_endpoint(None) == _register_mod.DEFAULT_ENDPOINT


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
            endpoint="http://my-collector:4318/v1/traces",
            set_global=False,
            auto_instrument=False,
        )
        mock_create_http.assert_called_once_with("http://my-collector:4318/v1/traces", None)

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
