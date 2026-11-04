# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Run an ``evaluate()`` suite in CI and report the result.

The suite is a Python callable (``module:function`` or ``path/to/file.py:function``)
that takes no arguments, calls ``evaluate()`` (or uses ``Benchmark``) and
returns the ``BenchmarkResult`` (or a list of them). This script:

1. calls ``register()`` with the OTLP endpoints (optionally SigV4),
2. runs the suite and flushes traces and evaluation events,
3. prints a summary and appends it to ``$GITHUB_STEP_SUMMARY``: a link to the
   run in OpenSearch Dashboards plus a table of per-evaluator means and
   pass/fail counts,
4. writes ``run-id`` / ``run-url`` to ``$GITHUB_OUTPUT``,
5. exits 1 if a ``--fail-on-regression`` threshold is not met.

Every option can also be set with an environment variable (used by the
composite action), so the script runs the same way locally::

    python examples/github-action/run_suite.py \\
        --suite examples/github-action/sample_suite.py:suite \\
        --endpoint http://localhost:4318 \\
        --service-name my-agent-ci \\
        --dashboards-url http://localhost:5601 \\
        --fail-on-regression "exact_match=0.5"
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2


@dataclass
class EvaluatorStats:
    name: str
    mean: float
    count: int
    passed: int
    failed: int


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suite", default=_env("EVAL_SUITE"), help="module:function or file.py:fn")
    p.add_argument(
        "--endpoint", default=_env("EVAL_ENDPOINT"), help="OTLP base URL for traces and logs"
    )
    p.add_argument("--traces-endpoint", default=_env("EVAL_TRACES_ENDPOINT"))
    p.add_argument("--logs-endpoint", default=_env("EVAL_LOGS_ENDPOINT"))
    p.add_argument("--aws-region", default=_env("EVAL_AWS_REGION"), help="enables SigV4")
    p.add_argument("--aws-service", default=_env("EVAL_AWS_SERVICE", "osis"))
    p.add_argument("--service-name", default=_env("EVAL_SERVICE_NAME", "agent-evals-ci"))
    p.add_argument("--dashboards-url", default=_env("EVAL_DASHBOARDS_URL"))
    p.add_argument(
        "--fail-on-regression",
        default=_env("EVAL_FAIL_ON_REGRESSION"),
        help="evaluator=min_mean pairs, comma or newline separated",
    )
    p.add_argument(
        "--pass-threshold",
        type=float,
        default=float(_env("EVAL_PASS_THRESHOLD", "0.5")),
        help="a score counts as pass when value >= this (default 0.5)",
    )
    p.add_argument("--working-directory", default=_env("EVAL_WORKING_DIRECTORY", "."))
    p.add_argument("--summary-file", default=_env("GITHUB_STEP_SUMMARY"))
    p.add_argument("--output-file", default=_env("GITHUB_OUTPUT"))
    return p.parse_args(argv)


def parse_thresholds(spec: str) -> dict[str, float]:
    """Parse ``"accuracy=0.8, relevance=0.7"`` (commas or newlines)."""
    thresholds: dict[str, float] = {}
    for raw in spec.replace("\n", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"Invalid fail-on-regression entry {item!r}; use evaluator=0.8")
        thresholds[name.strip()] = float(value)
    return thresholds


def load_suite(spec: str) -> Callable[[], Any]:
    """Load ``module:function`` or ``path/to/file.py:function``."""
    target, sep, func_name = spec.rpartition(":")
    if not sep or not target or not func_name:
        raise ValueError(f"--suite must be module:function or file.py:function, got {spec!r}")
    if target.endswith(".py") or os.sep in target:
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ValueError(f"Cannot load suite file {path}")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"{func_name!r} in {target!r} is not callable")
    return fn


# ---------------------------------------------------------------------------
# Telemetry setup
# ---------------------------------------------------------------------------


def setup_telemetry(args: argparse.Namespace) -> Any:
    """Call ``register()`` with the base and per-signal endpoints (optionally SigV4)."""
    from opensearch_genai_observability_sdk_py import register

    kwargs: dict[str, Any] = {
        "service_name": args.service_name,
        "endpoint": args.endpoint or None,
        "traces_endpoint": args.traces_endpoint or None,
        "logs_endpoint": args.logs_endpoint or None,
        "batch": True,
    }
    if args.aws_region:
        kwargs["sigv4"] = {"service": args.aws_service, "region": args.aws_region}
    return register(**kwargs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def run_url(dashboards_url: str, run_id: str) -> str:
    """The run's page in OpenSearch Dashboards: ``#/runs/<test.suite.run.id>``."""
    base = dashboards_url.rstrip("/")
    return f"{base}/app/agentTraces/evaluations#/runs/{quote(run_id, safe='')}"


def evaluator_stats(results: list[Any], pass_threshold: float) -> dict[str, EvaluatorStats]:
    values: dict[str, list[float]] = {}
    for result in results:
        for case in result.cases:
            for name, value in case.scores.items():
                values.setdefault(name, []).append(float(value))
    stats: dict[str, EvaluatorStats] = {}
    for name, vals in values.items():
        passed = sum(1 for v in vals if v >= pass_threshold)
        stats[name] = EvaluatorStats(
            name=name,
            mean=sum(vals) / len(vals),
            count=len(vals),
            passed=passed,
            failed=len(vals) - passed,
        )
    return stats


def case_counts(result: Any) -> tuple[int, int, int]:
    """(pass, fail, error) case counts, matching the local summary."""
    passed = failed = errors = 0
    for c in result.cases:
        if c.error or c.scorer_errors or c.status == "error":
            errors += 1
        elif c.status == "fail":
            failed += 1
        else:
            passed += 1
    return passed, failed, errors


def check_regressions(
    stats: dict[str, EvaluatorStats], thresholds: dict[str, float]
) -> list[str]:
    problems: list[str] = []
    for name, minimum in thresholds.items():
        s = stats.get(name)
        if s is None:
            problems.append(f"`{name}`: no scores recorded (expected mean >= {minimum:g})")
        elif s.mean < minimum:
            problems.append(f"`{name}`: mean {s.mean:.3f} is below {minimum:g}")
    return problems


def render_summary(
    results: list[Any],
    stats: dict[str, EvaluatorStats],
    *,
    dashboards_url: str,
    pass_threshold: float,
    thresholds: dict[str, float],
    problems: list[str],
) -> str:
    lines = ["## Agent evaluation results", ""]
    for result in results:
        s = result.summary
        passed, failed, errors = case_counts(result)
        title = f"**{s.benchmark_name}** run `{s.run_id}`"
        if dashboards_url:
            title += f": [open in OpenSearch Dashboards]({run_url(dashboards_url, s.run_id)})"
        lines.append(f"- {title}")
        lines.append(
            f"  - cases: {s.total_cases} (pass {passed}, fail {failed}, error {errors}), "
            f"duration {s.duration_ms / 1000:.1f}s"
        )
    lines += [
        "",
        f"| Evaluator | Mean | Pass (>= {pass_threshold:g}) | Fail | Scores | Threshold |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in sorted(stats):
        st = stats[name]
        gate = ""
        if name in thresholds:
            ok = st.mean >= thresholds[name]
            gate = f"{'pass' if ok else 'FAIL'} (min {thresholds[name]:g})"
        lines.append(
            f"| `{name}` | {st.mean:.3f} | {st.passed} | {st.failed} | {st.count} | {gate} |"
        )
    if not stats:
        lines.append("| (no scores) | | | | | |")
    if problems:
        lines += ["", "**Regression check failed:**", ""] + [f"- {p}" for p in problems]
    elif thresholds:
        lines += ["", "Regression check passed."]
    return "\n".join(lines) + "\n"


def _append(path: str, text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.suite:
        print("error: --suite (or EVAL_SUITE) is required", file=sys.stderr)
        return EXIT_USAGE
    try:
        thresholds = parse_thresholds(args.fail_on_regression)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    os.chdir(args.working_directory)
    sys.path.insert(0, os.getcwd())

    provider = setup_telemetry(args)
    try:
        suite = load_suite(args.suite)
        returned = suite()
    finally:
        # Export traces and evaluation events before reporting.
        provider.force_flush()

    results = returned if isinstance(returned, list) else [returned]
    if not results or not all(hasattr(r, "summary") and hasattr(r, "cases") for r in results):
        print(
            "error: the suite must return the BenchmarkResult from evaluate() (or a list)",
            file=sys.stderr,
        )
        provider.shutdown()
        return EXIT_USAGE

    stats = evaluator_stats(results, args.pass_threshold)
    problems = check_regressions(stats, thresholds)
    summary = render_summary(
        results,
        stats,
        dashboards_url=args.dashboards_url,
        pass_threshold=args.pass_threshold,
        thresholds=thresholds,
        problems=problems,
    )
    print(summary)
    _append(args.summary_file, summary)

    first = results[0].summary.run_id
    outputs = [f"run-id={first}"]
    if args.dashboards_url:
        outputs.append(f"run-url={run_url(args.dashboards_url, first)}")
    _append(args.output_file, "\n".join(outputs) + "\n")

    provider.shutdown()
    return EXIT_REGRESSION if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
