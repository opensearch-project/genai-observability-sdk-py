# Agent evaluations with the OpenSearch GenAI SDK

This guide covers the full path of an evaluation result: what the SDK sends, how to score agents online and offline, how to ingest the results into OpenSearch with Data Prepper or Amazon OpenSearch Ingestion, and how to check that they arrived.

- [What gets stored](#what-gets-stored)
- [Setup](#setup)
- [Online evaluation](#online-evaluation)
- [Offline evaluation](#offline-evaluation)
- [Ingestion](#ingestion)
- [Verify](#verify)
- [Troubleshooting](#troubleshooting)

## What gets stored

Each score is one OpenTelemetry log event named `gen_ai.evaluation.result`, as defined by the [OTel GenAI events conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/). The SDK creates no spans for evaluations. The event is a LogRecord sent over OTLP logs (`/v1/logs`) with:

- `event_name` = `gen_ai.evaluation.result`, severity INFO, an empty body, all data in attributes.
- The trace context of the evaluated span (`traceId` + `spanId`), or only the `traceId` for a trace-level result, or no trace context for a session-level result or a case whose task started no span.
- Instrumentation scope `opensearch-genai-observability-sdk-py` with the SDK version.

Attributes (each is set only when it has a value):

| Attribute | Status | Set by | Meaning |
|---|---|---|---|
| `gen_ai.evaluation.name` | OTel | all | Metric name, for example `relevance` |
| `gen_ai.evaluation.score.value` | OTel | all | Numeric score |
| `gen_ai.evaluation.score.label` | OTel | all | Label. `pass` / `fail` (lowercase) is the verdict; when a result has a verdict, it is always the verdict |
| `gen_ai.evaluation.explanation` | OTel | all | Evaluator rationale, up to 8192 characters |
| `error.type` | OTel | all | The evaluator or the task failed; no value or label |
| `gen_ai.agent.name` | OTel | all | Agent under evaluation |
| `gen_ai.agent.version` | OTel | all | Its version |
| `gen_ai.conversation.id` | OTel | `score()` | Evaluated session |
| `gen_ai.response.id` | OTel | `score()` | Evaluated completion |
| `test.suite.name` | OTel | offline | Benchmark name, stable across runs |
| `test.suite.run.id` | proposed OTel | offline | One run of the benchmark |
| `test.case.id` | proposed OTel | offline | One case of the dataset |
| `test.case.name` | OTel | offline | Readable case name |
| `cicd.pipeline.name` | OTel | offline, in CI | CI workflow or job |
| `cicd.pipeline.run.id` | OTel | offline, in CI | CI run id |
| `cicd.pipeline.run.url.full` | OTel | offline, in CI | Link to the CI run |
| `opensearch.gen_ai.evaluation.level` | OpenSearch | all | `span`, `trace`, `session` or `tool_call` |
| `opensearch.gen_ai.evaluation.evaluator.name` | OpenSearch | all | Evaluator name |
| `opensearch.gen_ai.evaluation.evaluator.version` | OpenSearch | `score()` | Evaluator version |
| `opensearch.gen_ai.evaluation.evaluator.type` | OpenSearch | all | `llm_judge`, `code`, `human` or `external` |
| `opensearch.gen_ai.evaluation.score.min` | OpenSearch | all | Lower bound of the scale |
| `opensearch.gen_ai.evaluation.score.max` | OpenSearch | all | Upper bound of the scale |
| `opensearch.gen_ai.evaluation.score.threshold` | OpenSearch | all | Pass threshold |
| `opensearch.gen_ai.evaluation.score.direction` | OpenSearch | all | `maximize` or `minimize` |
| `opensearch.gen_ai.evaluation.judge.trace_id` | OpenSearch | all | Trace of the judge's own run |
| `opensearch.gen_ai.evaluation.judge_label` | OpenSearch | all | The evaluator's own non-verdict label (for example `relevant`) when the result also has a verdict |
| `opensearch.test.case.expected` | OpenSearch | offline | Expected output of the case, up to 4096 characters |
| `opensearch.test.case.expected_truncated` | OpenSearch | offline | `true` when the expected output was cut |

- "OTel" names are in the OTel semantic conventions registry. `test.suite.run.id` and `test.case.id` are proposed in [semantic-conventions#1262](https://github.com/open-telemetry/semantic-conventions/issues/1262) and [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79).
- `opensearch.gen_ai.evaluation.*` names mirror the OTel `gen_ai.evaluation.*` names for fields OTel does not define yet.
- "offline" means `evaluate()` and `Benchmark.log()`. `score()` sets the `test.*` and `opensearch.test.case.*` keys only if you pass them in `attributes`.
- `score(attributes=...)` and run, case and score metadata are added to the events too, under the SDK fields: an SDK field always wins. They cannot set the evaluation fields (`gen_ai.evaluation.*`, `opensearch.gen_ai.evaluation.*`, `error.type`), even ones the SDK leaves unset: those keys are dropped with one warning per call. `score(attributes=...)` also cannot set `gen_ai.response.id`, `gen_ai.conversation.id` or `gen_ai.agent.*` (use the arguments). Metadata also cannot set `test.suite.name`, `test.suite.run.id`, `test.case.id`, `test.case.name`, `cicd.pipeline.*` or `opensearch.test.case.*`.

## Setup

```bash
pip install opensearch-genai-observability-sdk-py
```

Call `register()` once at startup. It sets up the OTLP traces pipeline and the OTLP logs pipeline that carries the evaluation events. `agent_name` and `agent_version` become `gen_ai.agent.name` and `gen_ai.agent.version` on every result, unless a call names another agent or the evaluated span names its own.

```python
from opensearch_genai_observability_sdk_py import register

register(
    endpoint="http://localhost:21890/opentelemetry",  # base URL for traces and logs
    service_name="travel-agent",
    agent_name="travel_planner",
    agent_version="v3",
)
```

Endpoints follow the OTel exporter spec, one rule per signal (traces, logs):

- `endpoint=` is the base URL for all signals (like `OTEL_EXPORTER_OTLP_ENDPOINT`). Over HTTP the SDK appends `/v1/traces` and `/v1/logs`; over gRPC it uses the base as-is for both.
- `traces_endpoint=` / `logs_endpoint=` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`) are full URLs for one signal and win over the base.
- With nothing set, the base is the local Data Prepper `http://localhost:21890/opentelemetry`. See the [README](../README.md#endpoints-traces-and-evaluation-logs) for the full order and the protocol rules.
- Amazon OpenSearch Ingestion has one pipeline per signal, so give both pipeline URLs. `sigv4=` signs both exporters:

```python
from opensearch_genai_observability_sdk_py import register

register(
    traces_endpoint="https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces",
    logs_endpoint="https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs",
    sigv4={"service": "osis", "region": "us-east-1"},
    agent_name="travel_planner",
    agent_version="v3",
)
```

If you configure OpenTelemetry yourself instead of calling `register()`, set a global `LoggerProvider` with an OTLP log exporter, or the events are dropped.

## Online evaluation

Use `score()` to grade live traffic: a judge, a code check or user feedback, run in your service or in a separate job.

### Levels

```python
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import Op, observe, score


@observe(op=Op.INVOKE_AGENT)
def travel_planner(question: str) -> str:
    answer = "Paris"
    # Span level: called while the agent span is active, so the result joins that span.
    score("tool_selection", 1.0, label="pass", evaluator="tool_check", evaluator_type="code")
    return answer


travel_planner("Capital of France?")

# Span level, later: score a stored span by its ids (works after the span ended).
score(
    "relevance",
    0.92,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
    evaluator="relevance_judge",
    evaluator_type="llm_judge",
)

# Trace level: only the trace id. No span id is derived.
score("helpfulness", 4.0, trace_id="6ebb9835f43af1552f2cebb9f5165e39", score_min=1, score_max=5)

# Session level: gen_ai.conversation.id; the level defaults to "session".
score("goal_completed", 1.0, label="pass", conversation_id="session-456", evaluator_type="human")
```

The level defaults to `span` with a span context, `trace` with only `trace_id`, and `session` with only `conversation_id`. Pass `level=` to set it (`span`, `trace`, `session`, `tool_call`).

### Pass rules and who decides

The verdict is `gen_ai.evaluation.score.label` = `pass` or `fail`. Your code decides it; OpenSearch Dashboards shows the stored label and never computes one. Two ways:

```python
from opensearch_genai_observability_sdk_py import score

# 1. A threshold: the SDK writes the label. direction defaults to "maximize".
score("answer_quality", 3.0, threshold=4, score_min=1, score_max=5)  # label = fail
score("latency_s", 9.2, threshold=12, direction="minimize")  # label = pass

# 2. The judge's own verdict. A pass / fail label wins over the threshold.
score("faithfulness", 0.4, label="fail", explanation="Claims a hotel the tool did not return")

# 3. A judge category plus a rule: the verdict takes the label, the category is kept.
score("relevance", 2.0, label="partially_relevant", threshold=4)
# label = fail, opensearch.gen_ai.evaluation.judge_label = partially_relevant
```

The rule, the same for `score()`, `evaluate()` and `Benchmark.log()`:

1. The verdict comes from the first source that gives one: a `pass` / `fail` `label` (any case), then `EvalScore.passed`, then `threshold` + `direction`.
2. The verdict is stored in `gen_ai.evaluation.score.label` as lowercase `pass` / `fail`. `TestCaseResult.status` uses the same verdict, so the stored event and the local result always agree.
3. A label that is not a verdict (for example `relevant`) moves to `opensearch.gen_ai.evaluation.judge_label` when the result has a verdict. Without a verdict it stays in `gen_ai.evaluation.score.label` and the result is neutral: Dashboards shows it as "Scored".
4. If two sources give different verdicts (for example `label="pass"` with `passed=False`), the SDK records the winning one. When the verdict replaces a non-verdict label or overrides another source, the SDK logs a warning, once per evaluation name.

### Scales and evaluators

- `score_min` / `score_max` record the scale (`opensearch.gen_ai.evaluation.score.min` / `.max`), so a 1 to 5 judge and a 0 to 1 check can be read side by side.
- `evaluator`, `evaluator_version` and `evaluator_type` name the evaluator (`opensearch.gen_ai.evaluation.evaluator.*`). The instrumentation scope is always the SDK.

### Judge trace and errors

If the judge itself is traced, pass its trace id so the result links to the judge's run:

```python
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import score

tracer = trace.get_tracer("my-judge")
with tracer.start_as_current_span("judge relevance") as judge_span:
    judge_value = 0.8  # call your judge model here
judge_trace_id = format(judge_span.get_span_context().trace_id, "032x")

score(
    "relevance",
    judge_value,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
    judge_trace_id=judge_trace_id,
    evaluator="relevance_judge",
    evaluator_type="llm_judge",
)
```

If the evaluator fails, record an error result: `error.type` set, no value and no label.

```python
from opensearch_genai_observability_sdk_py import score

score(
    "relevance",
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
    error_type="TimeoutError",
    evaluator="relevance_judge",
)
```

## Offline evaluation

Offline evaluation runs a dataset through the agent (`evaluate()`) or uploads results you already have (`Benchmark`). Each case gets `test.suite.name`, `test.suite.run.id`, `test.case.id`, `test.case.name` and, when given, the expected output.

### evaluate()

`evaluate()` calls your task once per case. Each call is its own trace, like production; the SDK adds no span around it and stamps the `test.*` ids on the root span the task starts. Each scorer result becomes one `gen_ai.evaluation.result` parented to that root span.

```python
import sys

from opensearch_genai_observability_sdk_py import EvalScore, Op, evaluate, observe, register, scorer

register(service_name="travel-agent", agent_name="travel_planner", agent_version="v3")


@observe(op=Op.INVOKE_AGENT)
def travel_planner(question: str) -> str:
    return "Paris" if "France" in question else "I do not know"


@scorer(name="exact_match", threshold=1, score_min=0, score_max=1, evaluator_type="code")
def exact_match(input, output, expected):
    return 1.0 if output == expected else 0.0


@scorer(name="answer_quality", evaluator_type="llm_judge")
def quality(input, output, expected):
    return EvalScore(
        name="answer_quality",
        value=4.0,
        threshold=4,
        score_min=1,
        score_max=5,
        explanation="Correct and concise",
        judge_trace_id=None,  # the judge's trace id, if the judge is traced
    )


result = evaluate(
    name="travel-capitals",
    task=travel_planner,
    data=[
        {
            "input": "Capital of France?",
            "expected": "Paris",
            "case_id": "fr",
            "case_name": "France",
        },
        {"input": "Capital of Peru?", "expected": "Lima", "case_id": "pe", "case_name": "Peru"},
    ],
    scores=[exact_match, quality],
    metadata={"app.variant": "v3-prompt-b"},
)

if result.summary.fail_count or result.summary.error_count:
    sys.exit(1)
```

- A scorer receives `(input, output, expected)` and returns a float, a bool (a verdict), an `EvalScore` or a list of `EvalScore`.
- `@scorer(...)` declares the evaluation name, `threshold`, `direction`, `score_min`, `score_max`, `evaluator` and `evaluator_type`. Fields set on a returned `EvalScore` win.
- `EvalScore(judge_trace_id=...)` sets `opensearch.gen_ai.evaluation.judge.trace_id`, in `evaluate()` and in `Benchmark.log()`.
- Scorers run outside the agent trace, so a traced judge gets its own trace.

### Benchmark

Use `Benchmark` to upload results from another harness (RAGAS, DeepEval, pytest, your own). Pass `trace_id` and `span_id` of the agent run to link each result to its trace.

```python
from opensearch_genai_observability_sdk_py import Benchmark, EvalScore

b = Benchmark(name="travel-capitals", agent_name="travel_planner", agent_version="v3")
b.log(
    input="Capital of France?",
    output="Paris",
    expected="Paris",
    case_id="fr",
    case_name="France",
    scores=[
        EvalScore(name="exact_match", value=1.0, threshold=1, score_min=0, score_max=1),
        EvalScore(name="answer_quality", value=3.0, threshold=4, score_min=1, score_max=5),
    ],
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
)
b.log(input="Capital of Peru?", expected={"city": "Lima"}, scores={"exact_match": 0.0})
summary = b.close()
```

`scores` is `{name: value}` or a list of `EvalScore`. `Benchmark` creates no spans.

### Expected output

The `"expected"` value of an `evaluate()` case, or `Benchmark.log(expected=...)`, is set on every `gen_ai.evaluation.result` of that case (error results included) as `opensearch.test.case.expected`:

- Strings are kept as-is. Other values are JSON-encoded, for example `{"city":"Lima"}`.
- Values longer than 4096 characters are cut to 4096 and `opensearch.test.case.expected_truncated` is set to `true`.
- It is not put on the agent's spans, and online `score()` never sets it.

### Run ids

`run_id` sets `test.suite.run.id` and is used exactly as given. Without it the SDK generates one (`run_<timestamp>_<random>`). Empty or whitespace-only values raise `ValueError`.

- CI shards: pass the same `run_id` from every shard so they report into one run.
- Re-scoring: send new results with an existing run's id to add them to that run. Earlier results stay (see [Re-imports](#d-re-imports-duplicate-results)).

```python
import os

from opensearch_genai_observability_sdk_py import Benchmark

run_id = f"travel-capitals-{os.environ.get('GITHUB_RUN_ID', 'local')}"
b = Benchmark(name="travel-capitals", run_id=run_id)
```

### CI pipeline attributes

Inside CI, results and agent root spans also get `cicd.pipeline.name`, `cicd.pipeline.run.id` and `cicd.pipeline.run.url.full`. Outside CI nothing is added.

| CI system | Detected by | `cicd.pipeline.name` | `cicd.pipeline.run.id` | `cicd.pipeline.run.url.full` |
|---|---|---|---|---|
| GitHub Actions | `GITHUB_ACTIONS=true` | `GITHUB_WORKFLOW` | `GITHUB_RUN_ID` | `GITHUB_SERVER_URL/GITHUB_REPOSITORY/actions/runs/GITHUB_RUN_ID` |
| GitLab CI | `GITLAB_CI=true` | `CI_PIPELINE_NAME`, else `CI_PROJECT_PATH` | `CI_PIPELINE_ID` | `CI_PIPELINE_URL` |
| Jenkins | `JENKINS_URL` and `BUILD_ID` | `JOB_NAME` | `BUILD_ID` | `BUILD_URL` |

For another CI system pass `ci_pipeline={"cicd.pipeline.run.id": "...", ...}` to `evaluate()` or `Benchmark()`.

### Errors

- The task raises (`evaluate()`): the case is not scored. Each scorer emits one error result under its evaluation name with `error.type` = the exception class name, no value, no label and no explanation. The run continues.
- A scorer raises: one error result with `error.type`, no value, no label and no explanation. The exception message is not sent; it stays in `TestCaseResult.scorer_errors`.
- `Benchmark.log(error_type="TimeoutError", scores=[...])`: every listed score is sent as an error result.

### Local summary

The summary is computed in your process and is not sent. `evaluate()` prints it and returns a `BenchmarkResult`; `Benchmark.close()` prints and returns the `BenchmarkSummary`.

- `BenchmarkSummary`: `benchmark_name`, `run_id`, `total_cases`, `pass_count`, `fail_count`, `error_count`, `duration_ms`, and `scores` (per name: `avg`, `min`, `max`, `count`).
- `TestCaseResult` (`result.cases`): `case_id`, `case_name`, `input`, `output`, `expected`, `scores`, `verdicts`, `status`, `error`, `error_type`, `scorer_errors`, `duration_ms`, and `trace_id` / `span_id` of the agent root span.
- Case `status`: `error` if the task or a scorer raised; else `fail` if any score has a `fail` verdict; else `pass`.

## Ingestion

Evaluation events arrive on the OTLP logs path. Route them to their own index so they do not mix with application logs. The examples use the index `otel-v1-genai-evaluation-%{yyyy.MM.dd}`; Agent Traces in OpenSearch Dashboards reads `otel-v1-genai-evaluation*` by default.

### a) Self-managed Data Prepper

Route `gen_ai.evaluation.result` log events from the OTLP source to their own sub-pipeline, copy `time` to `@timestamp`, and write to the evaluation index:

```yaml
otlp-pipeline:
  source:
    otlp:
      port: 21890
      ssl: false
  route:
    - evaluations: 'getEventType() == "LOG" and /eventName == "gen_ai.evaluation.result"'
    - logs: 'getEventType() == "LOG" and /eventName != "gen_ai.evaluation.result"'
    - traces: 'getEventType() == "TRACE"'
  sink:
    - pipeline:
        name: "otel-logs-pipeline"
        routes: ["logs"]
    - pipeline:
        name: "genai-evaluation-pipeline"
        routes: ["evaluations"]
    - pipeline:
        name: "otel-traces-pipeline"
        routes: ["traces"]

genai-evaluation-pipeline:
  source:
    pipeline:
      name: "otlp-pipeline"
  processor:
    - copy_values:
        entries:
          - from_key: "time"
            to_key: "@timestamp"
  sink:
    - opensearch:
        hosts: ["https://opensearch:9200"]
        username: "admin"
        password: "<password>"
        index: "otel-v1-genai-evaluation-%{yyyy.MM.dd}"
        template_type: index-template
        template_file: /usr/share/data-prepper/templates/genai-evaluation-index-template.json
```

`otel-logs-pipeline` and `otel-traces-pipeline` are your existing logs and traces pipelines. Save the [index template JSON](#index-template) as the `template_file`.

**Experimental: `index_type: genai-evaluation-plain`.** The Data Prepper draft [opensearch-project/data-prepper#7243](https://github.com/opensearch-project/data-prepper/pull/7243) adds this index type to the OpenSearch sink. It is not in a Data Prepper release and its mappings may change. The sink manages the template, so drop `index`, `template_type` and `template_file`:

```yaml
  sink:
    - opensearch:
        hosts: ["https://opensearch:9200"]
        username: "admin"
        password: "<password>"
        index_type: genai-evaluation-plain  # experimental
```

`genai-evaluation-plain` stores attributes under their dotted OpenTelemetry names, has no ISM policy, and writes to one index named `otel-v1-genai-evaluation` with the sink-managed template `otel-v1-genai-evaluation-index-template`. Do not combine it with your own template that matches `otel-v1-genai-evaluation`.

### b) Amazon OpenSearch Ingestion (OSI)

OSI does not have the evaluation index type yet. Use the same pipeline shape in an OSI pipeline and create the index template once before ingesting.

1. Create the index template on the domain or collection, for example in Dashboards Dev Tools:

```
PUT _index_template/otel-v1-genai-evaluation
<the index template JSON below>
```

2. Create an OSI pipeline with an OTLP logs source. Point `register(logs_endpoint=...)` at its `/v1/logs` URL.

```yaml
version: "2"
otel-logs-pipeline:
  source:
    otel_logs_source:
      path: "/v1/logs"
  processor:
    - copy_values:
        entries:
          - from_key: "time"
            to_key: "@timestamp"
  route:
    - evaluations: '/eventName == "gen_ai.evaluation.result"'
    - logs: '/eventName != "gen_ai.evaluation.result"'
  sink:
    - opensearch:
        hosts: ["https://search-my-domain-abc123.us-east-1.es.amazonaws.com"]
        aws:
          sts_role_arn: "arn:aws:iam::111122223333:role/osi-pipeline-role"
          region: "us-east-1"
        index: "otel-v1-genai-evaluation-%{yyyy.MM.dd}"
        routes: ["evaluations"]
    - opensearch:
        hosts: ["https://search-my-domain-abc123.us-east-1.es.amazonaws.com"]
        aws:
          sts_role_arn: "arn:aws:iam::111122223333:role/osi-pipeline-role"
          region: "us-east-1"
        index: "logs-otel-v1-%{yyyy.MM.dd}"
        routes: ["logs"]
```

Replace the host, role and region with yours. The pipeline role needs write access to both index patterns.

#### Index template

Used by the `template_file` in (a) and the `PUT _index_template/otel-v1-genai-evaluation` in (b). `opensearch.test.case.expected` is `text` without a `keyword` sub-field: searchable, not aggregatable. Other unknown string attributes map to `keyword`, other numbers to `double`.

<details>
<summary>otel-v1-genai-evaluation index template</summary>

```json
{
  "index_patterns": ["otel-v1-genai-evaluation-*"],
  "priority": 100,
  "version": 1,
  "template": {
    "mappings": {
      "date_detection": false,
      "_source": {"enabled": true},
      "dynamic_templates": [
        {
          "long_attributes": {
            "mapping": {"type": "double"},
            "path_match": "attributes.*",
            "match_mapping_type": "long"
          }
        },
        {
          "double_attributes": {
            "mapping": {"type": "double"},
            "path_match": "attributes.*",
            "match_mapping_type": "double"
          }
        },
        {
          "string_attributes": {
            "mapping": {"type": "keyword", "ignore_above": 1024},
            "path_match": "attributes.*",
            "match_mapping_type": "string"
          }
        },
        {
          "long_resource_attributes": {
            "mapping": {"type": "long"},
            "path_match": "resource.attributes.*",
            "match_mapping_type": "long"
          }
        },
        {
          "double_resource_attributes": {
            "mapping": {"type": "double"},
            "path_match": "resource.attributes.*",
            "match_mapping_type": "double"
          }
        },
        {
          "string_resource_attributes": {
            "mapping": {"type": "keyword", "ignore_above": 256},
            "path_match": "resource.attributes.*",
            "match_mapping_type": "string"
          }
        },
        {
          "long_scope_attributes": {
            "mapping": {"type": "long"},
            "path_match": "instrumentationScope.attributes.*",
            "match_mapping_type": "long"
          }
        },
        {
          "double_scope_attributes": {
            "mapping": {"type": "double"},
            "path_match": "instrumentationScope.attributes.*",
            "match_mapping_type": "double"
          }
        },
        {
          "string_scope_attributes": {
            "mapping": {"type": "keyword", "ignore_above": 256},
            "path_match": "instrumentationScope.attributes.*",
            "match_mapping_type": "string"
          }
        }
      ],
      "properties": {
        "@timestamp": {"type": "date_nanos"},
        "time": {"type": "date_nanos"},
        "observedTimestamp": {"type": "date_nanos"},
        "traceId": {"type": "keyword", "ignore_above": 256},
        "spanId": {"type": "keyword", "ignore_above": 256},
        "flags": {"type": "long"},
        "eventName": {"type": "keyword", "ignore_above": 256},
        "severityNumber": {"type": "integer"},
        "severityText": {"type": "keyword"},
        "body": {"type": "text"},
        "serviceName": {"type": "keyword"},
        "schemaUrl": {"type": "keyword"},
        "droppedAttributesCount": {"type": "integer"},
        "instrumentationScope": {
          "properties": {
            "name": {"type": "keyword"},
            "version": {"type": "keyword"},
            "schemaUrl": {"type": "keyword"},
            "droppedAttributesCount": {"type": "integer"}
          }
        },
        "resource": {
          "properties": {
            "schemaUrl": {"type": "keyword"},
            "droppedAttributesCount": {"type": "integer"}
          }
        },
        "attributes": {
          "properties": {
            "gen_ai": {
              "properties": {
                "evaluation": {
                  "properties": {
                    "name": {"type": "keyword"},
                    "explanation": {
                      "type": "text",
                      "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
                    },
                    "score": {
                      "properties": {
                        "value": {"type": "double"},
                        "label": {"type": "keyword"}
                      }
                    }
                  }
                },
                "conversation": {
                  "properties": {"id": {"type": "keyword"}}
                },
                "agent": {
                  "properties": {"name": {"type": "keyword"}}
                },
                "response": {
                  "properties": {"id": {"type": "keyword"}}
                }
              }
            },
            "opensearch": {
              "properties": {
                "gen_ai": {
                  "properties": {
                    "evaluation": {
                      "properties": {
                        "score": {
                          "properties": {
                            "min": {"type": "double"},
                            "max": {"type": "double"},
                            "threshold": {"type": "double"},
                            "direction": {"type": "keyword"}
                          }
                        },
                        "evaluator": {
                          "properties": {
                            "name": {"type": "keyword"},
                            "version": {"type": "keyword"},
                            "type": {"type": "keyword"}
                          }
                        },
                        "level": {"type": "keyword"},
                        "judge_label": {"type": "keyword"},
                        "judge": {
                          "properties": {"trace_id": {"type": "keyword"}}
                        }
                      }
                    }
                  }
                },
                "test": {
                  "properties": {
                    "case": {
                      "properties": {
                        "expected": {"type": "text"},
                        "expected_truncated": {"type": "boolean"}
                      }
                    }
                  }
                }
              }
            },
            "test": {
              "properties": {
                "suite": {
                  "properties": {
                    "name": {"type": "keyword"},
                    "run": {
                      "properties": {"id": {"type": "keyword"}}
                    }
                  }
                },
                "case": {
                  "properties": {
                    "id": {"type": "keyword"},
                    "name": {"type": "keyword"}
                  }
                }
              }
            },
            "session": {
              "properties": {"id": {"type": "keyword"}}
            },
            "error": {
              "properties": {"type": {"type": "keyword"}}
            },
            "aws": {
              "properties": {
                "bedrock_agentcore": {
                  "properties": {
                    "evaluation_level": {"type": "keyword"},
                    "online_evaluation_config": {
                      "properties": {
                        "name": {"type": "keyword"},
                        "arn": {"type": "keyword"}
                      }
                    },
                    "evaluator": {
                      "properties": {
                        "arn": {"type": "keyword"},
                        "rating_scale": {"type": "keyword"}
                      }
                    }
                  }
                }
              }
            },
            "cicd": {
              "properties": {
                "pipeline": {
                  "properties": {
                    "name": {"type": "keyword"},
                    "run": {
                      "properties": {
                        "id": {"type": "keyword"},
                        "url": {
                          "properties": {"full": {"type": "keyword"}}
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

</details>

### c) Optional mappings you add yourself

These processors are not part of the default pipeline. Add them to `genai-evaluation-pipeline` (or the OSI pipeline) after `copy_values` if you ingest results from other producers.

**AgentCore results.** Copy AgentCore's evaluation level (lowercased) to `opensearch.gen_ai.evaluation.level` and its `session.id` to `gen_ai.conversation.id`, only when those are absent. The original attributes stay.

```yaml
    - add_entries:
        entries:
          - key: "/attributes/evl_agentcore_level"
            value_expression: "/attributes/aws.bedrock_agentcore.evaluation_level"
            add_when: "/attributes/opensearch.gen_ai.evaluation.level == null and /attributes/aws.bedrock_agentcore.evaluation_level != null"
    - lowercase_string:
        with_keys: ["/attributes/evl_agentcore_level"]
    - add_entries:
        entries:
          - key: "/attributes/opensearch.gen_ai.evaluation.level"
            value_expression: "/attributes/evl_agentcore_level"
            add_when: "/attributes/evl_agentcore_level != null"
          - key: "/attributes/gen_ai.conversation.id"
            value_expression: "/attributes/session.id"
            add_when: "/attributes/gen_ai.conversation.id == null and /attributes/session.id != null"
    - delete_entries:
        with_keys: ["/attributes/evl_agentcore_level"]
```

**A pass/fail label for a numeric score.** For a producer that sends only `gen_ai.evaluation.score.value`, write the label from a threshold you choose, only when `gen_ai.evaluation.score.label` is absent. This example passes `helpfulness` at 0.7 or more and records the rule:

```yaml
    - add_entries:
        entries:
          - key: "/attributes/gen_ai.evaluation.score.label"
            value: "pass"
            add_when: '/attributes/gen_ai.evaluation.name == "helpfulness" and /attributes/gen_ai.evaluation.score.label == null and /attributes/gen_ai.evaluation.score.value != null and /attributes/gen_ai.evaluation.score.value >= 0.7'
          - key: "/attributes/gen_ai.evaluation.score.label"
            value: "fail"
            add_when: '/attributes/gen_ai.evaluation.name == "helpfulness" and /attributes/gen_ai.evaluation.score.label == null and /attributes/gen_ai.evaluation.score.value != null and /attributes/gen_ai.evaluation.score.value < 0.7'
          - key: "/attributes/opensearch.gen_ai.evaluation.score.threshold"
            value: 0.7
            add_when: '/attributes/gen_ai.evaluation.name == "helpfulness" and /attributes/opensearch.gen_ai.evaluation.score.threshold == null'
          - key: "/attributes/opensearch.gen_ai.evaluation.score.direction"
            value: "maximize"
            add_when: '/attributes/gen_ai.evaluation.name == "helpfulness" and /attributes/opensearch.gen_ai.evaluation.score.direction == null'
```

### d) Re-imports duplicate results

Documents get ids generated by OpenSearch; there is no dedupe key. Sending the same results twice (a retried upload, a re-run import of the same time range) stores them twice. Import each range once, and use a new `run_id` for a full re-run.

## Verify

Count results per evaluation and verdict (Query Workbench or Discover, PPL):

```
source = otel-v1-genai-evaluation*
| where eventName = 'gen_ai.evaluation.result'
| stats count() by `attributes.gen_ai.evaluation.name`, `attributes.gen_ai.evaluation.score.label`
```

Results of one run:

```
source = otel-v1-genai-evaluation*
| where `attributes.test.suite.run.id` = 'travel-capitals-local'
| fields `@timestamp`, `attributes.test.case.id`, `attributes.gen_ai.evaluation.name`, `attributes.gen_ai.evaluation.score.value`, `attributes.gen_ai.evaluation.score.label`, `attributes.error.type`, `attributes.opensearch.test.case.expected`
```

In OpenSearch Dashboards, open **Agent Traces > Evaluations**. The page is behind feature flags; set both in `opensearch_dashboards.yml`:

```yaml
explore.agentTraces.enabled: true
explore.agentTraces.evaluations.enabled: true
```

## Troubleshooting

- **No events at all.** `register()` was not called, or your own OTel setup has no global `LoggerProvider`. Short scripts: the SDK flushes at exit; call `provider.force_flush()` on the `TracerProvider` that `register()` returns if the process is killed.
- **Traces arrive, results do not.** The logs endpoint is wrong: nothing is derived from the traces URL, so `traces_endpoint=` alone sends logs to the base (`endpoint=` or the local Data Prepper default), and a custom `exporter=` without a logs destination skips logs with a warning. Set `logs_endpoint=` or `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`, or a base `endpoint=` that serves both signals. With OSI, the logs pipeline must be a separate pipeline with an OTLP logs source, and `sigv4=` must be set so the logs are signed.
- **Results land in the logs index.** The route did not match `eventName`. Check that the route uses `/eventName == "gen_ai.evaluation.result"` and that your Data Prepper version maps the OTLP `event_name` field to `eventName` (added in opensearch-project/data-prepper#7148).
- **Field types look wrong** (strings as `text` with `.keyword` sub-fields, `time` not a date). The index was created before the index template. Create the template, then wait for the next daily index or reindex.
- **`opensearch.test.case.expected` cannot be aggregated.** Expected: it is `text`. Filter or search on it instead.
- **A run shows twice the results.** The same results were sent twice; see [Re-imports](#d-re-imports-duplicate-results).
- **No pass/fail shown.** The result has no `pass` / `fail` label. Pass `label=` or `threshold=`, or add the label mapping in [(c)](#c-optional-mappings-you-add-yourself).
- **Evaluations page missing.** Turn on both feature flags in [Verify](#verify).
