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

from opensearch_genai_observability_sdk_py import EvalScore, evaluate, scorer

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
        assert rs.parse_thresholds("a=0.8, b>=0.5\nc <= 12") == {
            "a": rs.Gate(">=", 0.8),
            "b": rs.Gate(">=", 0.5),
            "c": rs.Gate("<=", 12.0),
        }
        assert rs.parse_thresholds("") == {}

    def test_gate(self):
        assert rs.Gate(">=", 0.8).ok(0.8) and not rs.Gate(">=", 0.8).ok(0.7)
        assert rs.Gate("<=", 12).ok(9) and not rs.Gate("<=", 12).ok(13)
        assert rs.Gate("<=", 12).describe() == "max 12"

    @pytest.mark.parametrize("bad", ["a", "=0.5", "a=x", "a<0.5"])
    def test_bad_thresholds(self, bad):
        with pytest.raises(ValueError):
            rs.parse_thresholds(bad)

    def test_load_suite_from_file(self):
        fn = rs.load_suite(str(_ACTION_DIR / "sample_suite.py") + ":suite")
        assert callable(fn)

    def test_load_suite_bad_spec(self):
        with pytest.raises(ValueError, match="module:function"):
            rs.load_suite("no_colon")

    def test_load_suite_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            rs.load_suite(str(tmp_path / "nope.py") + ":suite")


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
        stats = rs.evaluation_stats([result], pass_threshold=0.5)
        assert stats["exact"].mean == 0.5
        assert (stats["exact"].passed, stats["exact"].failed) == (1, 1)
        assert (stats["half"].passed, stats["half"].failed) == (2, 0)
        assert rs.check_regressions(stats, {"half": rs.Gate(">=", 0.5)}) == []
        problems = rs.check_regressions(
            stats, {"exact": rs.Gate(">=", 0.9), "missing": rs.Gate(">=", 0.1)}
        )
        assert len(problems) == 2

    def test_counts_use_verdicts(self):
        """A minimize scorer and a label verdict are counted as the stored label says."""

        @scorer(name="latency_s", direction="minimize", threshold=12)
        def latency(i, o, e):
            return 9.0 if i == "fast" else 20.0

        def judge(i, o, e):
            # Verdict says fail although the value is above --pass-threshold.
            return EvalScore(name="judge", value=0.9, label="fail")

        result = evaluate(
            name="gha-verdicts",
            task=lambda x: x,
            data=[{"input": "fast"}, {"input": "slow"}],
            scores=[latency, judge],
        )
        stats = rs.evaluation_stats([result], pass_threshold=0.5)
        assert (stats["latency_s"].passed, stats["latency_s"].failed) == (1, 1)
        assert (stats["judge"].passed, stats["judge"].failed) == (0, 2)
        assert stats["latency_s"].mean == 14.5
        # Gate a minimize scorer from above.
        assert rs.check_regressions(stats, {"latency_s": rs.Gate("<=", 15)}) == []
        problems = rs.check_regressions(stats, {"latency_s": rs.Gate("<=", 10)})
        assert problems == ["`latency_s`: mean 14.500 is above 10"]

    def test_verdict_only_scores(self):
        result = evaluate(
            name="gha-label-only",
            task=lambda x: x,
            data=[{"input": "a"}],
            scores=[lambda i, o, e: EvalScore(name="tool_ok", value=None, label="pass")],
        )
        stats = rs.evaluation_stats([result], pass_threshold=0.5)
        assert stats["tool_ok"].mean is None
        assert (stats["tool_ok"].passed, stats["tool_ok"].count) == (1, 1)
        assert rs.check_regressions(stats, {"tool_ok": rs.Gate(">=", 0.5)}) == [
            "`tool_ok`: no scores recorded (expected mean >= 0.5)"
        ]
        text = rs.render_summary(
            [result],
            stats,
            dashboards_url="",
            pass_threshold=0.5,
            thresholds={},
            problems=[],
        )
        assert "| `tool_ok` | - | 1 | 0 | 1 | 0 |  |" in text

    def test_summary_markdown(self):
        result = _result()
        stats = rs.evaluation_stats([result], 0.5)
        text = rs.render_summary(
            [result],
            stats,
            dashboards_url="http://osd:5601",
            pass_threshold=0.5,
            thresholds={"exact": rs.Gate(">=", 0.9)},
            problems=rs.check_regressions(stats, {"exact": rs.Gate(">=", 0.9)}),
        )
        assert f"#/runs/{result.summary.run_id}" in text
        assert "| `exact` | 0.500 | 1 | 1 | 2 | 0 | FAIL (min 0.9) |" in text
        assert "Regression check failed" in text

    def test_errored_cases_fail_the_gate(self):
        """19 of 20 cases raise, 1 scores 1.0: the mean is 1.0 but the gate fails."""

        def task(x):
            if x != 0:
                raise RuntimeError("boom")
            return x

        def accuracy(i, o, e) -> float:
            return 1.0

        result = evaluate(
            name="gha-errors",
            task=task,
            data=[{"input": i} for i in range(20)],
            scores=[accuracy],
        )
        stats = rs.evaluation_stats([result], 0.5)
        assert stats["accuracy"].mean == 1.0
        assert stats["accuracy"].errors == 19
        problems = rs.check_regressions(stats, {"accuracy": rs.Gate(">=", 0.8)})
        assert problems == ["`accuracy`: 19 errored case(s) without a score"]
        assert rs.check_errors([result]) == [
            "**gha-errors**: 19 of 20 case(s) errored (task or scorer raised)"
        ]
        text = rs.render_summary(
            [result],
            stats,
            dashboards_url="",
            pass_threshold=0.5,
            thresholds={"accuracy": rs.Gate(">=", 0.8)},
            problems=problems,
            errors=rs.check_errors([result]),
        )
        assert "| `accuracy` | 1.000 | 1 | 0 | 1 | 19 | FAIL (min 0.8) |" in text
        assert "Errored cases (job fails)" in text

    def test_scorer_error_only_fails_the_gate_without_a_value(self):
        """A case where another scorer raised does not fail a gate that has its value."""

        def good(i, o, e) -> float:
            return 1.0

        def bad(i, o, e) -> float:
            raise ValueError("judge down")

        result = evaluate(
            name="gha-scorer-error", task=lambda x: x, data=[{"input": "a"}], scores=[good, bad]
        )
        stats = rs.evaluation_stats([result], 0.5, names=["good", "bad"])
        assert (stats["good"].errors, stats["bad"].errors) == (0, 1)
        assert rs.check_regressions(stats, {"good": rs.Gate(">=", 0.5)}) == []
        assert rs.check_regressions(stats, {"bad": rs.Gate(">=", 0.5)}) == [
            "`bad`: no scores recorded (expected mean >= 0.5)",
            "`bad`: 1 errored case(s) without a score",
        ]

    def test_gate_keys_on_evaluation_name(self):
        """Scorer ``quality`` returning ``answer_quality``: gate and table use the latter."""

        def quality(i, o, e) -> EvalScore:
            return EvalScore(name="answer_quality", value=0.9)

        result = evaluate(
            name="gha-names", task=lambda x: x, data=[{"input": "a"}], scores=[quality]
        )
        stats = rs.evaluation_stats([result], 0.5)
        assert set(stats) == {"answer_quality"}
        assert rs.check_regressions(stats, {"answer_quality": rs.Gate(">=", 0.8)}) == []

    def test_case_counts(self):
        def boom(x):
            raise RuntimeError("x")

        ok = evaluate(name="c1", task=lambda x: x, data=[{"input": "a"}], scores=[])
        bad = evaluate(name="c2", task=boom, data=[{"input": "a"}], scores=[])
        assert rs.case_counts(ok) == (1, 0, 0)
        assert rs.case_counts(bad) == (0, 0, 1)


class TestSetupTelemetry:
    def _args(self, *argv):
        return rs.parse_args(["--suite", "x:y", *argv])

    def test_base_endpoint(self):
        with patch("opensearch_genai_observability_sdk_py.register") as register:
            rs.setup_telemetry(self._args("--endpoint", "http://c:4318"))
        kwargs = register.call_args.kwargs
        assert kwargs["endpoint"] == "http://c:4318"
        assert kwargs["traces_endpoint"] is None and kwargs["logs_endpoint"] is None
        assert "sigv4" not in kwargs

    def test_osis_sigv4(self):
        with patch("opensearch_genai_observability_sdk_py.register") as register:
            rs.setup_telemetry(
                self._args(
                    "--traces-endpoint",
                    "https://t/v1/traces",
                    "--logs-endpoint",
                    "https://l/v1/logs",
                    "--aws-region",
                    "us-east-1",
                )
            )
        kwargs = register.call_args.kwargs
        assert kwargs["traces_endpoint"] == "https://t/v1/traces"
        assert kwargs["logs_endpoint"] == "https://l/v1/logs"
        assert kwargs["sigv4"] == {"service": "osis", "region": "us-east-1"}


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

    def _main(self, tmp_path, suite, *extra):
        provider = MagicMock()
        with patch.object(rs, "setup_telemetry", return_value=provider):
            code = rs.main(
                [
                    "--suite",
                    suite,
                    "--summary-file",
                    str(tmp_path / "summary.md"),
                    "--output-file",
                    str(tmp_path / "output.txt"),
                    "--working-directory",
                    str(tmp_path),
                    *extra,
                ]
            )
        return code, provider

    @pytest.mark.parametrize(
        "suite",
        ["no_colon", "missing_file.py:suite", "bad_suite.py:not_callable", "bad_suite.py:absent"],
    )
    def test_bad_suite_is_a_config_error(self, tmp_path, monkeypatch, capsys, suite):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bad_suite.py").write_text("not_callable = 1\n")
        code, provider = self._main(tmp_path, suite)
        assert code == 2
        assert "error:" in capsys.readouterr().err
        provider.shutdown.assert_called_once()

    def test_errored_cases_fail_without_a_gate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "erroring.py").write_text(
            "from opensearch_genai_observability_sdk_py import evaluate\n"
            "def boom(x):\n"
            "    raise RuntimeError('down')\n"
            "def suite():\n"
            "    return evaluate(name='all-errored', task=boom,\n"
            "                    data=[{'input': 'a'}, {'input': 'b'}],\n"
            "                    scores=[lambda i, o, e: 1.0])\n"
        )
        code, _ = self._main(tmp_path, str(tmp_path / "erroring.py") + ":suite")
        assert code == 1
        summary = (tmp_path / "summary.md").read_text()
        assert "pass 0, fail 0, error 2" in summary
        assert "**all-errored**: 2 of 2 case(s) errored" in summary
