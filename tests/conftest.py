# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Shared test fixtures for opensearch-genai-observability-sdk-py tests.

Sets up a TracerProvider with InMemorySpanExporter and a LoggerProvider
with an in-memory log exporter so tests can capture and assert on spans and
evaluation events (``gen_ai.evaluation.result`` log records) without a real
collector.

A single TracerProvider is shared across the entire test session.
The exporter is cleared before and after every test via the autouse
_clear_spans fixture so tests never see each other's spans.

Because the SDK decorators fetch trace.get_tracer() at call time (not at
decoration/import time), the global provider set here is always resolved
correctly when a decorated function is invoked — no private API hacks
required.
"""

import pytest
from opentelemetry import _logs, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Module-level singletons, initialised once per process.
_exporter = InMemorySpanExporter()
_provider = TracerProvider(
    resource=Resource.create({"service.name": "test-service"}),
)
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)

try:  # renamed in opentelemetry-sdk 1.39; the old name is a deprecated alias
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter as _InMemoryLogs
except ImportError:  # pragma: no cover - older opentelemetry-sdk
    from opentelemetry.sdk._logs.export import InMemoryLogExporter as _InMemoryLogs

_log_exporter = _InMemoryLogs()
_logger_provider = LoggerProvider(
    resource=Resource.create({"service.name": "test-service"}),
)
_logger_provider.add_log_record_processor(SimpleLogRecordProcessor(_log_exporter))
_logs.set_logger_provider(_logger_provider)


@pytest.fixture(autouse=True)
def _reset_label_warnings():
    """Label warnings are once per evaluation name per process; reset per test."""
    import importlib

    # The package exports the score() function under the module's name.
    _score_mod = importlib.import_module("opensearch_genai_observability_sdk_py.score")

    _score_mod._label_warned.clear()
    yield
    _score_mod._label_warned.clear()


@pytest.fixture(autouse=True)
def _clear_spans():
    """Clear the shared InMemorySpanExporter before and after every test.

    The ``autouse=True`` ensures this runs for *every* test, even those
    that don't request the ``exporter`` fixture explicitly.
    """
    _exporter.clear()
    _log_exporter.clear()
    yield
    _exporter.clear()
    _log_exporter.clear()


@pytest.fixture()
def exporter():
    """Provide the shared InMemorySpanExporter for tests that need it.

    Call ``exporter.get_finished_spans()`` after exercising the code
    under test to inspect the captured spans.
    """
    return _exporter


@pytest.fixture()
def log_exporter():
    """Provide the shared in-memory log exporter.

    ``log_exporter.get_finished_logs()`` returns the emitted log records
    (``ReadableLogRecord``; the record itself is ``.log_record``).
    """
    return _log_exporter


@pytest.fixture()
def eval_events(log_exporter):
    """Return a callable listing emitted ``gen_ai.evaluation.result`` records."""

    def _get():
        return [
            r
            for r in log_exporter.get_finished_logs()
            if getattr(r, "log_record", r).event_name == "gen_ai.evaluation.result"
        ]

    return _get
