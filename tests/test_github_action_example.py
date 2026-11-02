# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for examples/github-action/run_suite.py (no network)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opensearch_genai_observability_sdk_py import EvalScore, evaluate

_ACTION_DIR = Path(__file__).resolve().parent.parent / "examples" / "github-action"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"gha_{name}", _ACTION_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


rs = _load("run_suite")


def _result():
    def exact(i, o, e) -> EvalScore:
        return EvalScore(name="exact", value=1.0 if o == e else 0.0)

    def half(i, o, e) -> float:
        return 0.5

    return evaluate(
        name="gha-unit",
        task=lambda x: x,
        data=[{"input": "a", "expected": "a"}, {"input": "b", "expected": "c"}],
        scores=[exact, half],
    )


class TestParsing:
    def test_thresholds(self):
        assert rs.parse_thresholds("a=0.8, b=0.5\nc=1") == {"a": 0.8, "b": 0.5, "c": 1.0}
        assert rs.parse_thresholds("") == {}

    @pytest.mark.parametrize("bad", ["a", "=0.5", "a=x"])
    def test_bad_thresholds(self, bad):
        with pytest.raises(ValueError):
            rs.parse_thresholds(bad)

    def test_load_suite_from_file(self):
        fn = rs.load_suite(str(_ACTION_DIR / "sample_suite.py") + ":suite")
        assert callable(fn)

    def test_load_suite_bad_spec(self):
        with pytest.raises(ValueError, match="module:function"):
            rs.load_suite("no_colon")


class TestRunUrl:
    def test_deep_link_shape(self):
        url = rs.run_url("https://osd.example.com/", "run_20261028_010101_abcd1234")
        assert url == (
            "https://osd.example.com/app/agentTraces/evaluations#/runs/run_20261028_010101_abcd1234"
        )

    def test_run_id_is_path_encoded(self):
        assert rs.run_url("https://osd", "a/b c").endswith("#/runs/a%2Fb%20c")


class TestReport:
    def test_stats_and_regressions(self):
        result = _result()
        stats = rs.evaluator_stats([result], pass_threshold=0.5)
        assert stats["exact"].mean == 0.5
        assert (stats["exact"].passed, stats["exact"].failed) == (1, 1)
        assert (stats["half"].passed, stats["half"].failed) == (2, 0)
        assert rs.check_regressions(stats, {"half": 0.5}) == []
        problems = rs.check_regressions(stats, {"exact": 0.9, "missing": 0.1})
        assert len(problems) == 2

    def test_summary_markdown(self):
        result = _result()
        stats = rs.evaluator_stats([result], 0.5)
        text = rs.render_summary(
            [result],
            stats,
            dashboards_url="http://osd:5601",
            pass_threshold=0.5,
            thresholds={"exact": 0.9},
            problems=rs.check_regressions(stats, {"exact": 0.9}),
        )
        assert f"#/runs/{result.summary.run_id}" in text
        assert "| `exact` | 0.500 | 1 | 1 | 2 | FAIL (min 0.9) |" in text
        assert "Regression check failed" in text

    def test_case_counts(self):
        def boom(x):
            raise RuntimeError("x")

        ok = evaluate(name="c1", task=lambda x: x, data=[{"input": "a"}], scores=[])
        bad = evaluate(name="c2", task=boom, data=[{"input": "a"}], scores=[])
        assert rs.case_counts(ok) == (1, 0, 0)
        assert rs.case_counts(bad) == (0, 0, 1)


class TestMain:
    def _run(self, tmp_path, *extra):
        summary = tmp_path / "summary.md"
        output = tmp_path / "output.txt"
        provider = MagicMock()
        with patch.object(rs, "setup_telemetry", return_value=provider):
            code = rs.main(
                [
                    "--suite",
                    str(_ACTION_DIR / "sample_suite.py") + ":suite",
                    "--dashboards-url",
                    "http://osd:5601",
                    "--summary-file",
                    str(summary),
                    "--output-file",
                    str(output),
                    "--working-directory",
                    str(tmp_path),
                    *extra,
                ]
            )
        provider.force_flush.assert_called()
        return code, summary.read_text(), output.read_text()

    def test_pass(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, summary, output = self._run(tmp_path, "--fail-on-regression", "answer_length=0.1")
        assert code == 0
        assert "/app/agentTraces/evaluations#/runs/run_" in summary
        assert "Regression check passed." in summary
        assert output.startswith("run-id=run_")
        assert "run-url=http://osd:5601/app/agentTraces/evaluations" in output

    def test_regression_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code, summary, _ = self._run(tmp_path, "--fail-on-regression", "contains_expected=0.99")
        assert code == 1
        assert "contains_expected`: mean 0.750 is below 0.99" in summary

    def test_missing_suite(self):
        assert rs.main(["--suite", ""]) == 2
