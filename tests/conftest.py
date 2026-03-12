# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Shared test fixtures for opensearch-genai-observability-sdk-py tests.

Sets up a TracerProvider with InMemorySpanExporter so tests can capture
and assert on spans without a real collector.

A single TracerProvider is shared across the entire test session.
The exporter is cleared before and after every test via the autouse
_clear_spans fixture so tests never see each other's spans.

Because the SDK decorators fetch trace.get_tracer() at call time (not at
decoration/import time), the global provider set here is always resolved
correctly when a decorated function is invoked — no private API hacks
required.
"""

import pytest
from opentelemetry import trace
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


@pytest.fixture(autouse=True)
def _clear_spans():
    """Clear the shared InMemorySpanExporter before and after every test.

    The ``autouse=True`` ensures this runs for *every* test, even those
    that don't request the ``exporter`` fixture explicitly.
    """
    _exporter.clear()
    yield
    _exporter.clear()


@pytest.fixture()
def exporter():
    """Provide the shared InMemorySpanExporter for tests that need it.

    Call ``exporter.get_finished_spans()`` after exercising the code
    under test to inspect the captured spans.
    """
    return _exporter
