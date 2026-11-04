# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Shared internal utilities for the SDK.

Not part of the public API — subject to change without notice.
"""

from __future__ import annotations

import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlparse, urlunparse

from opentelemetry import _logs

logger = logging.getLogger(__name__)

# Instrumentation scope used when the SDK itself emits telemetry (for
# example evaluation events without a named evaluator).
SDK_SCOPE_NAME = "opensearch-genai-observability-sdk-py"

try:
    SDK_VERSION: str = version(SDK_SCOPE_NAME)
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    SDK_VERSION = "0.0.0+unknown"


# OTel CI/CD attributes (semantic-conventions ``cicd.*`` registry) stamped on
# evaluation results and agent root spans when they run inside a CI system.
CICD_PIPELINE_NAME_KEY = "cicd.pipeline.name"
CICD_PIPELINE_RUN_ID_KEY = "cicd.pipeline.run.id"
CICD_PIPELINE_RUN_URL_KEY = "cicd.pipeline.run.url.full"
CICD_KEYS = (CICD_PIPELINE_NAME_KEY, CICD_PIPELINE_RUN_ID_KEY, CICD_PIPELINE_RUN_URL_KEY)


def ci_pipeline_attributes(
    override: dict[str, str] | None = None, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """``cicd.pipeline.*`` for the CI pipeline run this process runs in.

    Read from the CI environment (GitHub Actions, GitLab CI, Jenkins); empty
    outside CI. ``override`` (keys from ``CICD_KEYS``) sets or replaces values,
    for CI systems not detected here.
    """
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    if env.get("GITHUB_ACTIONS") == "true":
        run_id = env.get("GITHUB_RUN_ID", "")
        out[CICD_PIPELINE_NAME_KEY] = env.get("GITHUB_WORKFLOW", "")
        out[CICD_PIPELINE_RUN_ID_KEY] = run_id
        server, repo = env.get("GITHUB_SERVER_URL", ""), env.get("GITHUB_REPOSITORY", "")
        if server and repo and run_id:
            out[CICD_PIPELINE_RUN_URL_KEY] = f"{server}/{repo}/actions/runs/{run_id}"
    elif env.get("GITLAB_CI") == "true":
        out[CICD_PIPELINE_NAME_KEY] = env.get("CI_PIPELINE_NAME") or env.get("CI_PROJECT_PATH", "")
        out[CICD_PIPELINE_RUN_ID_KEY] = env.get("CI_PIPELINE_ID", "")
        out[CICD_PIPELINE_RUN_URL_KEY] = env.get("CI_PIPELINE_URL", "")
    elif env.get("JENKINS_URL") and env.get("BUILD_ID"):
        out[CICD_PIPELINE_NAME_KEY] = env.get("JOB_NAME", "")
        out[CICD_PIPELINE_RUN_ID_KEY] = env.get("BUILD_ID", "")
        out[CICD_PIPELINE_RUN_URL_KEY] = env.get("BUILD_URL", "")
    for key, value in (override or {}).items():
        if key in CICD_KEYS and value:
            out[key] = str(value)
        elif key not in CICD_KEYS:
            logger.warning("ci_pipeline: %r is not one of %s; ignoring it", key, CICD_KEYS)
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Agent identity on evaluation result events
# ---------------------------------------------------------------------------

AGENT_NAME_KEY = "gen_ai.agent.name"
# OTel GenAI attribute for the agent version.
AGENT_VERSION_KEY = "gen_ai.agent.version"

_agent_defaults: dict[str, str | None] = {"name": None, "version": None}


def set_agent_defaults(name: str | None = None, version: str | None = None) -> None:
    """Process-wide agent identity, set by ``register(agent_name=, agent_version=)``."""
    _agent_defaults["name"] = name or None
    _agent_defaults["version"] = version or None


def agent_defaults() -> tuple[str | None, str | None]:
    """The process-wide ``(agent name, agent version)``, or ``(None, None)``."""
    return _agent_defaults["name"], _agent_defaults["version"]


def span_agent(span: object) -> tuple[str | None, str | None]:
    """``(gen_ai.agent.name, gen_ai.agent.version)`` read from a recording SDK span.

    Best effort: only spans that expose their attributes (SDK spans) are read;
    nothing is looked up elsewhere.
    """
    attrs = getattr(span, "attributes", None)
    if not attrs:
        return None, None
    try:
        name = attrs.get(AGENT_NAME_KEY)
        ver = attrs.get(AGENT_VERSION_KEY)
    except Exception:  # pragma: no cover - attribute views that are not mappings
        return None, None
    return (str(name) if name else None), (str(ver) if ver else None)


def resolve_agent(
    name: str | None = None,
    version: str | None = None,
    span: object | None = None,
) -> dict[str, str]:
    """Agent identity attributes for an SDK event (only the keys known).

    Precedence per field: the explicit argument, then the evaluated span's own
    attribute (the agent that actually ran), then the ``register()`` default.
    Names are never invented: an unknown agent is left out.
    """
    span_name, span_version = span_agent(span) if span is not None else (None, None)
    default_name, default_version = agent_defaults()
    resolved_name = name or span_name or default_name
    resolved_version = version or span_version or default_version
    out: dict[str, str] = {}
    if resolved_name:
        out[AGENT_NAME_KEY] = resolved_name
    if resolved_version:
        out[AGENT_VERSION_KEY] = resolved_version
    return out


def event_scope(
    evaluator: str | None = None, evaluator_version: str | None = None
) -> tuple[str, str]:
    """Instrumentation scope ``(name, version)`` for an SDK-emitted event.

    Always the SDK (the instrumentation library, as OTel defines the scope).
    The evaluator identity is carried in the event attributes
    (``opensearch.gen_ai.evaluation.evaluator.*``). Data Prepper and OpenSearch
    Dashboards derive the result source from the scope name. The arguments
    are kept for compatibility and ignored.
    """
    del evaluator, evaluator_version
    return SDK_SCOPE_NAME, SDK_VERSION


def get_event_logger(
    evaluator: str | None = None, evaluator_version: str | None = None
) -> _logs.Logger:
    """OTel Logger for SDK events, with the SDK as the instrumentation scope."""
    name, scope_version = event_scope(evaluator, evaluator_version)
    return _logs.get_logger(name, scope_version)


# ---------------------------------------------------------------------------
# Expected output of an offline test case
# ---------------------------------------------------------------------------

# OpenSearch attributes (no OTel name yet) for the expected output of a test
# case, set on every gen_ai.evaluation.result event of that case.
TEST_CASE_EXPECTED_KEY = "opensearch.test.case.expected"
TEST_CASE_EXPECTED_TRUNCATED_KEY = "opensearch.test.case.expected_truncated"
EXPECTED_MAX_CHARS = 4096


def expected_attributes(expected: Any) -> dict[str, Any]:
    """``opensearch.test.case.expected`` (+ ``_truncated``) for an expected value.

    Strings are used as-is; other values are JSON-encoded (compact, non-ASCII
    kept, unknown types via ``str``). Values longer than
    ``EXPECTED_MAX_CHARS`` (4096) characters are cut to that length and
    ``opensearch.test.case.expected_truncated`` is set to ``True``. ``None``
    means no expected value: nothing is set.
    """
    if expected is None:
        return {}
    if isinstance(expected, str):
        text = expected
    else:
        text = json.dumps(expected, default=str, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= EXPECTED_MAX_CHARS:
        return {TEST_CASE_EXPECTED_KEY: text}
    return {
        TEST_CASE_EXPECTED_KEY: text[:EXPECTED_MAX_CHARS],
        TEST_CASE_EXPECTED_TRUNCATED_KEY: True,
    }


# Exact attribute keys written by the SDK — user metadata must not use these.
_RESERVED_KEYS = frozenset(
    {
        "test.suite.name",
        "test.suite.run.id",
        "test.case.id",
        "test.case.name",
        TEST_CASE_EXPECTED_KEY,
        TEST_CASE_EXPECTED_TRUNCATED_KEY,
        "cicd.pipeline.name",
        "cicd.pipeline.run.id",
        "cicd.pipeline.run.url.full",
    }
)


def parse_hex(value: str) -> int | None:
    """Parse a hex string (with or without 0x prefix) to int.

    Returns ``None`` on failure.
    """
    try:
        return int(value.lstrip("0x").lstrip("0X") or "0", 16)
    except ValueError:
        return None


def validate_metadata_keys(metadata: dict, *, context: str = "") -> None:
    """Warn if metadata keys collide with SDK-reserved attribute keys."""
    for key in metadata:
        if key in _RESERVED_KEYS:
            logger.warning(
                "%sMetadata key '%s' is reserved by the SDK and will be ignored",
                f"{context}: " if context else "",
                key,
            )


def derive_logs_endpoint(traces_endpoint: str) -> str | None:
    """Derive the OTLP logs endpoint from an OTLP traces endpoint, if possible.

    This is the only derivation the SDK does: a traces URL whose path ends
    with ``/v1/traces`` maps to the same base with ``/v1/logs`` (scheme,
    host, port and any path prefix such as ``/opentelemetry`` are kept).
    Any other path returns ``None``: the SDK does not guess.

    Examples::

        http://localhost:4318/v1/traces                  -> http://localhost:4318/v1/logs
        http://localhost:21890/opentelemetry/v1/traces   -> .../opentelemetry/v1/logs
        https://x.us-east-1.osis.amazonaws.com/v1/traces -> .../v1/logs
        http://collector:4318/custom                     -> None
    """
    parsed = urlparse(traces_endpoint)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1/traces"):
        return None
    return urlunparse(parsed._replace(path=path[: -len("/v1/traces")] + "/v1/logs"))
