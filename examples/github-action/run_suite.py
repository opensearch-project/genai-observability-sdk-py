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
   run in OpenSearch Dashboards plus a table of per-evaluation means,
   pass/fail counts (from each score's verdict, the same one stored in
   ``gen_ai.evaluation.score.label``; ``--pass-threshold`` only for scores
   without a verdict) and errored cases,
4. writes ``run-id`` / ``run-url`` to ``$GITHUB_OUTPUT``,
5. exits 1 if any case errored (the task or a scorer raised), or if a
   ``--fail-on-regression`` gate is not met (``name=0.8`` or ``name>=0.8``:
   mean at least; ``name<=12``: mean at most, for ``direction="minimize"``
   scorers such as latency). An errored case without a value for a gated
   evaluation also fails that gate,
6. exits 2 on a configuration error (bad ``--suite``, bad gate, endpoints
   ``register()`` rejects, a suite that does not return a ``BenchmarkResult``).

Gates key on the evaluation name (``gen_ai.evaluation.name``, the
``EvalScore.name`` or ``@scorer(name=...)``, else the scorer function name),
the same name ``TestCaseResult.scores`` uses and OpenSearch Dashboards groups
results by. The evaluator (``opensearch.gen_ai.evaluation.evaluator.name``)
is not used: one evaluator may produce several evaluations.

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
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2


@dataclass
class EvaluationStats:
    name: str  # evaluation name (gen_ai.evaluation.name)
    mean: float | None  # None when the evaluation has verdicts but no numeric values
    count: int
    passed: int
    failed: int
    errors: int = 0  # errored cases with no value for this evaluation


@dataclass(frozen=True)
class Gate:
    """A ``--fail-on-regression`` gate on an evaluation mean."""

    op: str  # ">=" (mean at least) or "<=" (mean at most)
    value: float

    def ok(self, mean: float) -> bool:
        return mean <= self.value if self.op == "<=" else mean >= self.value

    def describe(self) -> str:
        return f"{'max' if self.op == '<=' else 'min'} {self.value:g}"


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
        help="evaluation=min, evaluation>=min or evaluation<=max (mean, by evaluation name), "
        "comma or newline separated",
    )
    p.add_argument(
        "--pass-threshold",
        type=float,
        default=float(_env("EVAL_PASS_THRESHOLD", "0.5")),
        help="for scores without a verdict: pass when value >= this (default 0.5)",
    )
    p.add_argument("--working-directory", default=_env("EVAL_WORKING_DIRECTORY", "."))
    p.add_argument("--summary-file", default=_env("GITHUB_STEP_SUMMARY"))
    p.add_argument("--output-file", default=_env("GITHUB_OUTPUT"))
    return p.parse_args(argv)


_GATE = re.compile(r"^(?P<name>[^<>=\s]+)\s*(?P<op><=|>=|=)\s*(?P<value>\S+)$")


def parse_thresholds(spec: str) -> dict[str, Gate]:
    """Parse ``"accuracy=0.8, latency_s<=12"`` (commas or newlines).

    ``name=0.8`` and ``name>=0.8`` gate the mean from below; ``name<=12``
    from above (for ``direction="minimize"`` scorers).
    """
    thresholds: dict[str, Gate] = {}
    for raw in spec.replace("\n", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        match = _GATE.match(item)
        if match is None:
            raise ValueError(
                f"Invalid fail-on-regression entry {item!r}; use name=0.8, "
                "name>=0.8 or name<=12 (name: the evaluation name)"
            )
        try:
            value = float(match["value"])
        except ValueError:
            raise ValueError(f"Invalid number in fail-on-regression entry {item!r}") from None
        op = "<=" if match["op"] == "<=" else ">="
        thresholds[match["name"]] = Gate(op, value)
    return thresholds


def load_suite(spec: str) -> Callable[[], Any]:
    """Load ``module:function`` or ``path/to/file.py:function``."""
    target, sep, func_name = spec.rpartition(":")
    if not sep or not target or not func_name:
        raise ValueError(f"--suite must be module:function or file.py:function, got {spec!r}")
    if target.endswith(".py") or os.sep in target:
        path = Path(target).resolve()
        if not path.is_file():
            raise ValueError(f"Suite file {path} does not exist")
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


def case_errored(case: Any) -> bool:
    """The task or a scorer raised (or returned no usable result) on this case."""
    return bool(case.error or case.scorer_errors or case.status == "error")


def evaluation_stats(
    results: list[Any], pass_threshold: float, names: Iterable[str] = ()
) -> dict[str, EvaluationStats]:
    """Per-evaluation mean, pass / fail counts and errored cases.

    Keyed by evaluation name (``case.scores`` / ``case.verdicts``). Pass /
    fail comes from ``case.verdicts``: the verdict the SDK resolved (``pass``
    / ``fail`` label, ``passed``, or ``threshold`` + ``direction`` from the
    score or its ``@scorer``), which is also the stored label Dashboards
    shows. Only a score without a verdict falls back to
    ``value >= pass_threshold``. ``errors`` counts errored cases that have
    neither a value nor a verdict for the evaluation. ``names`` (for example the gated names) are
    included even when no case scored them.
    """
    values: dict[str, list[float]] = {}
    passed: dict[str, int] = dict.fromkeys(names, 0)
    failed: dict[str, int] = {}
    errored_cases: list[Any] = []
    for result in results:
        for case in result.cases:
            if case_errored(case):
                errored_cases.append(case)
            verdicts = case.verdicts or {}
            for name in {**case.scores, **verdicts}:
                value = case.scores.get(name)
                if value is not None:
                    values.setdefault(name, []).append(float(value))
                verdict = verdicts.get(name)
                if verdict is None and value is not None:
                    verdict = "pass" if float(value) >= pass_threshold else "fail"
                passed[name] = passed.get(name, 0) + (verdict == "pass")
                failed[name] = failed.get(name, 0) + (verdict == "fail")
    stats: dict[str, EvaluationStats] = {}
    for name in passed:
        vals = values.get(name, [])
        stats[name] = EvaluationStats(
            name=name,
            mean=sum(vals) / len(vals) if vals else None,
            count=passed[name] + failed.get(name, 0),
            passed=passed[name],
            failed=failed.get(name, 0),
            errors=sum(
                case.scores.get(name) is None and name not in (case.verdicts or {})
                for case in errored_cases
            ),
        )
    return stats


def case_counts(result: Any) -> tuple[int, int, int]:
    """(pass, fail, error) case counts, matching the local summary."""
    passed = failed = errors = 0
    for c in result.cases:
        if case_errored(c):
            errors += 1
        elif c.status == "fail":
            failed += 1
        else:
            passed += 1
    return passed, failed, errors


def gate_ok(stats: EvaluationStats | None, gate: Gate) -> bool:
    return stats is not None and stats.mean is not None and not stats.errors and gate.ok(stats.mean)


def check_regressions(stats: dict[str, EvaluationStats], thresholds: dict[str, Gate]) -> list[str]:
    """Problems with the ``--fail-on-regression`` gates (empty when all pass).

    A gate fails when its mean is outside the gate, the evaluation has no
    numeric value at all, or an errored case has no value for it.
    """
    problems: list[str] = []
    for name, gate in thresholds.items():
        s = stats.get(name)
        if s is None or s.mean is None:
            problems.append(
                f"`{name}`: no scores recorded (expected mean {gate.op} {gate.value:g})"
            )
        elif not gate.ok(s.mean):
            side = "above" if gate.op == "<=" else "below"
            problems.append(f"`{name}`: mean {s.mean:.3f} is {side} {gate.value:g}")
        if s is not None and s.errors:
            problems.append(f"`{name}`: {s.errors} errored case(s) without a score")
    return problems


def check_errors(results: list[Any]) -> list[str]:
    """One problem per run with errored cases (these fail the job, gate or not)."""
    problems: list[str] = []
    for result in results:
        errors = case_counts(result)[2]
        if errors:
            problems.append(
                f"**{result.summary.benchmark_name}**: {errors} of "
                f"{len(result.cases)} case(s) errored (task or scorer raised)"
            )
    return problems


def render_summary(
    results: list[Any],
    stats: dict[str, EvaluationStats],
    *,
    dashboards_url: str,
    pass_threshold: float,
    thresholds: dict[str, Gate],
    problems: list[str],
    errors: Sequence[str] = (),
) -> str:
    lines = ["## Agent evaluation results", ""]
    for result in results:
        s = result.summary
        n_pass, n_fail, n_error = case_counts(result)
        title = f"**{s.benchmark_name}** run `{s.run_id}`"
        if dashboards_url:
            title += f": [open in OpenSearch Dashboards]({run_url(dashboards_url, s.run_id)})"
        lines.append(f"- {title}")
        lines.append(
            f"  - cases: {s.total_cases} (pass {n_pass}, fail {n_fail}, error {n_error}), "
            f"duration {s.duration_ms / 1000:.1f}s"
        )
    lines += [
        "",
        "| Evaluation | Mean | Pass | Fail | Scores | Errors | Threshold |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in sorted(stats):
        st = stats[name]
        gate = ""
        if name in thresholds:
            ok = gate_ok(st, thresholds[name])
            gate = f"{'pass' if ok else 'FAIL'} ({thresholds[name].describe()})"
        mean = "-" if st.mean is None else f"{st.mean:.3f}"
        lines.append(
            f"| `{name}` | {mean} | {st.passed} | {st.failed} | {st.count} | {st.errors} | {gate} |"
        )
    if not stats:
        lines.append("| (no scores) | | | | | | |")
    lines += [
        "",
        "Pass / Fail use each score's verdict (the stored label); scores without one "
        f"count as pass when value >= {pass_threshold:g}. Errors: errored cases "
        "with no value for that evaluation.",
    ]
    if errors:
        lines += ["", "**Errored cases (job fails):**", ""] + [f"- {e}" for e in errors]
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

    try:
        provider = setup_telemetry(args)
    except ValueError as exc:  # for example a base URL ending in /v1/traces
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        suite = load_suite(args.suite)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        provider.shutdown()
        return EXIT_USAGE
    try:
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

    stats = evaluation_stats(results, args.pass_threshold, names=thresholds)
    problems = check_regressions(stats, thresholds)
    errors = check_errors(results)
    summary = render_summary(
        results,
        stats,
        dashboards_url=args.dashboards_url,
        pass_threshold=args.pass_threshold,
        thresholds=thresholds,
        problems=problems,
        errors=errors,
    )
    print(summary)
    _append(args.summary_file, summary)

    first = results[0].summary.run_id
    outputs = [f"run-id={first}"]
    if args.dashboards_url:
        outputs.append(f"run-url={run_url(args.dashboards_url, first)}")
    _append(args.output_file, "\n".join(outputs) + "\n")

    provider.shutdown()
    return EXIT_REGRESSION if problems or errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
