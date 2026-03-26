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

import logging

logger = logging.getLogger(__name__)

# Exact attribute keys written by the SDK — user metadata must not use these.
_RESERVED_KEYS = frozenset(
    {
        "test.suite.name",
        "test.suite.run.id",
        "test.suite.run.status",
        "test.case.id",
        "test.case.name",
        "test.case.result.status",
        "test.case.error",
        "test.case.input",
        "test.case.output",
        "test.case.expected",
        "gen_ai.operation.name",
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
