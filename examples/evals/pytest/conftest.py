# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""pytest fixtures: tracing and one Benchmark run for the whole test session."""

from __future__ import annotations

import os

import pytest
from agent import AGENT_NAME

from opensearch_genai_observability_sdk_py import Benchmark, register


@pytest.fixture(scope="session")
def eval_run():
    """The Benchmark run every test logs its verdicts to."""
    # Strands emits its own chat spans; auto_instrument=False keeps an installed Bedrock
    # (botocore) instrumentor from tracing each model call a second time.
    provider = register(
        auto_instrument=False,
        service_name="travel-assistant-evals-pytest-py",
        agent_name=AGENT_NAME,
        agent_version="v1",
    )
    run = Benchmark(
        "travel-assistant-suite-pytest",
        run_name=os.environ.get("EVAL_RUN_NAME", "pytest"),
    )
    yield run
    run.close()  # prints the local summary
    provider.force_flush()  # export the last spans and results before pytest exits
