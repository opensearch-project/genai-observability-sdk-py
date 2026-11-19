# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""The eval examples keep their own copy of the travel assistant and its checks, so each
folder runs on its own. These tests keep the copies from drifting apart."""

import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EVAL_AGENTS = sorted(EXAMPLES.glob("evals/*/agent.py"))


@pytest.mark.skipif(not EVAL_AGENTS, reason="examples not in this checkout")
def test_eval_examples_share_one_travel_assistant():
    first = EVAL_AGENTS[0].read_text()
    for path in EVAL_AGENTS[1:]:
        assert path.read_text() == first, (
            f"{path.relative_to(EXAMPLES)} differs from {EVAL_AGENTS[0].relative_to(EXAMPLES)}"
        )


def _function_without_docstring(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if ast.get_docstring(fn) is not None:
        fn.body = fn.body[1:]
    return ast.dump(fn, include_attributes=False)


@pytest.mark.skipif(not EVAL_AGENTS, reason="examples not in this checkout")
def test_uses_tool_data_is_the_same_check_in_native_and_pytest():
    native = _function_without_docstring(EXAMPLES / "evals/native/scorers.py", "uses_tool_data")
    in_pytest = _function_without_docstring(
        EXAMPLES / "evals/pytest/test_travel_assistant.py", "uses_tool_data"
    )
    assert native == in_pytest
