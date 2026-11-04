# GitHub Action: agent evaluations in CI

A reusable composite action that runs your `evaluate()` suite on every pull request, sends the agent traces and evaluation events to OpenSearch, and reports the result in the job summary:

- a link to the run in OpenSearch Dashboards: `<dashboards_url>/app/agentTraces/evaluations#/runs/<test.suite.run.id>`
- a markdown table of per-evaluator means and pass/fail counts
- an optional regression gate that fails the job when an evaluator's mean drops below a minimum

| File | Purpose |
|---|---|
| [`action.yml`](action.yml) | The composite action |
| [`run_suite.py`](run_suite.py) | Script the action runs. Also works locally |
| [`example-workflow.yml`](example-workflow.yml) | Sample workflow to copy to `.github/workflows/` |
| [`sample_suite.py`](sample_suite.py) | Sample suite with a mock agent |

## 1. Write a suite

A suite is a function with no arguments that calls `evaluate()` and returns its `BenchmarkResult` (or a list of them). The action calls `register()` before importing it, so `@observe` spans and evaluation events go to the configured endpoints.

```python
# evals/suite.py
from opensearch_genai_observability_sdk_py import EvalScore, evaluate

from my_agent import agent  # decorated with @observe(op=Op.INVOKE_AGENT)


def contains_expected(input, output, expected) -> EvalScore:
    return EvalScore(name="contains_expected", value=float(expected in output))


def suite():
    return evaluate(
        name="my-agent-suite",
        task=agent,
        data=[{"case_id": "paris", "input": "Capital of France?", "expected": "Paris"}],
        scores=[contains_expected],
    )
```

Each named scorer becomes an evaluator (`opensearch.gen_ai.evaluation.evaluator.name` on its results). The run produces the agent's own traces plus one `gen_ai.evaluation.result` log event per score.

## 2. Endpoints

Traces and evaluation logs share one base URL (`endpoint`) or use one full URL each (`traces-endpoint`, `logs-endpoint`); see "Endpoints: traces and evaluation logs" in the [SDK README](../../README.md#endpoints-traces-and-evaluation-logs).

- **OpenSearch Ingestion (OSIS):** one pipeline per signal. Create a traces pipeline and a logs pipeline (OTLP logs source; Data Prepper routes the evaluation events to `otel-v1-genai-evaluation-*`), then pass both URLs as `traces-endpoint` and `logs-endpoint` and set `aws-region` to sign both with SigV4. Give the workflow AWS credentials with `osis:Ingest` on both pipelines, for example with `aws-actions/configure-aws-credentials` and an OIDC role.
- **OTel Collector / Data Prepper:** pass the base URL as `endpoint` (for example `http://collector:4318`; the SDK appends `/v1/traces` and `/v1/logs`) and leave `aws-region` empty. Extra headers (for example auth) can be set with the standard `OTEL_EXPORTER_OTLP_HEADERS` env var on the step.

Store the URLs as secrets or variables.

## 3. Add the workflow

Copy [`example-workflow.yml`](example-workflow.yml) to `.github/workflows/agent-evals.yml` and adjust `suite`, the endpoints and the thresholds. Reference the action by path and a pinned commit SHA of this repository, or copy this directory into your repository and use `uses: ./.github/actions/agent-evals`.

### Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `suite` | yes | | `module:function` or `path/to/file.py:function` |
| `endpoint` | | | OTLP base URL for traces and logs |
| `traces-endpoint` | | | Full OTLP traces URL (overrides `endpoint`) |
| `logs-endpoint` | | | Full OTLP logs URL for evaluation events (overrides `endpoint`) |
| `aws-region` | | | Enables SigV4 signing when set |
| `aws-service` | | `osis` | SigV4 service: `osis` or `es` |
| `service-name` | | `agent-evals-ci` | `service.name` resource attribute |
| `dashboards-url` | | | OpenSearch Dashboards base URL for the run link |
| `fail-on-regression` | | | Gates on evaluator means: `evaluator=0.8` or `evaluator>=0.8` (at least), `evaluator<=12` (at most, for `direction="minimize"` scorers), comma or newline separated |
| `pass-threshold` | | `0.5` | Fallback for scores without a verdict: counts as pass in the table when `value >= threshold`. Scores with a verdict use it |
| `working-directory` | | `.` | Directory to run the suite from (added to `sys.path`) |
| `install` | | `opensearch-genai-observability-sdk-py[aws]` | pip requirement(s) installed first |

### Outputs

| Output | Description |
|---|---|
| `run-id` | `test.suite.run.id` of the (first) run |
| `run-url` | Link to the run in OpenSearch Dashboards |

The table counts pass / fail from each score's verdict (`case.verdicts`: a `pass` / `fail` label, `passed`, or the `@scorer` threshold and direction), the same verdict stored as `gen_ai.evaluation.score.label`, so it matches OpenSearch Dashboards. The job fails (exit code 1) when a `fail-on-regression` evaluator mean is outside its gate or the evaluator produced no scores. Exit code 2 means a configuration error.

### CI/CD attributes

Inside GitHub Actions the SDK stamps the OTel `cicd.pipeline.*` attributes on every evaluation result and the agent root spans, from the job's environment:

| Attribute | From |
|---|---|
| `cicd.pipeline.name` | `GITHUB_WORKFLOW` |
| `cicd.pipeline.run.id` | `GITHUB_RUN_ID` |
| `cicd.pipeline.run.url.full` | `$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID` |

So every evaluation run links back to the workflow run that produced it. Nothing is added when you run the suite locally.

## Run locally

The script reads the same settings from flags (or `EVAL_*` env vars), so you can try a suite against a local collector before wiring CI:

```bash
python examples/github-action/run_suite.py \
  --suite examples/github-action/sample_suite.py:suite \
  --endpoint http://localhost:4318 \
  --service-name my-agent-local \
  --dashboards-url http://localhost:5601 \
  --fail-on-regression "contains_expected=0.75"
```

Set `GITHUB_STEP_SUMMARY` / `GITHUB_OUTPUT` to file paths to see what the action writes.
