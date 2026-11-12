# Agent evaluations with the OpenSearch GenAI SDK

New here? [Getting started with agent evaluations](getting-started-evaluations.md) is the short path: the hierarchy, online and offline examples, Dashboards deep links and copyable LLM prompts.

This guide covers the full path of an evaluation result: what the SDK sends, how to score agents online and offline, how to ingest the results into OpenSearch with Data Prepper or Amazon OpenSearch Ingestion, and how to check that they arrived.

- [Data model](#data-model)
- [What gets stored](#what-gets-stored)
- [Setup](#setup)
- [Online evaluation](#online-evaluation)
- [Offline evaluation](#offline-evaluation)
- [Ingestion](#ingestion)
- [Verify](#verify)
- [Troubleshooting](#troubleshooting)

## Data model

Offline results are organized as suite, run and case. Each evaluator produces one `gen_ai.evaluation.result` event per case; the agent's input and output stay in the agent's own trace.

```
Test suite  "travel-planner-golden"                 test.suite.name
 └─ Run      run_20261105_215438_5df57445           test.suite.run.id  (one execution of the suite)
     └─ Case  "Plan a trip to Lisbon"               test.case.id / test.case.name
         ├─ Input      "Plan 3 days in Lisbon under $1,000"   → the agent's trace
         ├─ Expected   "Itinerary with flights + hotel ≤ $1,000"  → opensearch.test.case.expected
         ├─ Output     the agent's actual answer                  → the agent's trace
         └─ Evaluators run on (input, output, expected, trace):
              ├─ answer_quality   score 4/5, pass   → 1 result event
              ├─ budget_respected pass               → 1 result event
              └─ safety           pass               → 1 result event
```

Where each part is stored:

- Suite, run and case ids, the expected output and the scores are attributes of every result event of the case, in the evaluation index (`otel-v1-genai-evaluation`).
- Input and output are never put on result events. They are in the agent's trace (span index), which the result links to by `traceId` / `spanId`.
- The judge's own LLM call is a separate trace, linked by `opensearch.gen_ai.evaluation.judge.trace_id`. Its full prompt is on that judge span, not on the result (see [Judge prompt](#judge-prompt)).

Case identity:

- Across runs, a case is `test.suite.name` + `test.case.id`. Within a run, it is `test.suite.run.id` + `test.case.id`.
- The default `test.case.id` is the first 16 hex characters of the SHA-256 of the JSON-encoded input. Editing the input therefore makes a new case. Pass `case_id=` (an `evaluate()` data item key or `Benchmark.log(case_id=)`) to keep a case's identity stable when you edit its input.
- Online results (`score()` on live traffic) have no suite, run or case.

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
| `opensearch.gen_ai.evaluation.evaluator.description` | OpenSearch | all | What the evaluator checks (criteria or rubric summary) |
| `gen_ai.request.model` | OTel | all | Model the LLM judge used |
| `gen_ai.prompt.name` | OTel | all | Name of the judge prompt template |
| `gen_ai.prompt.version` | OTel | all | Version of the judge prompt (`name@version` identifies the prompt) |
| `opensearch.test.suite.run.name` | OpenSearch | offline | Human-readable run name. Optional; readers fall back to the short run id |
| `opensearch.test.suite.run.description` | OpenSearch | offline | Free-text run description, up to 1024 characters |
| `opensearch.test.case.expected` | OpenSearch | offline | Expected output of the case, up to 32768 characters |
| `opensearch.test.case.expected_truncated` | OpenSearch | offline | `true` when the expected output was cut |

- Result events never carry the judge's system prompt (`gen_ai.system_instructions`): it goes on the judge's own LLM span. See [Judge prompt](#judge-prompt).
- "OTel" names are in the OTel semantic conventions registry. `test.suite.run.id` and `test.case.id` are proposed in [semantic-conventions#1262](https://github.com/open-telemetry/semantic-conventions/issues/1262) and [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79).
- `opensearch.gen_ai.evaluation.*` names mirror the OTel `gen_ai.evaluation.*` names for fields OTel does not define yet.
- "offline" means `evaluate()` and `Benchmark.log()`. `score()` sets the `test.*` and `opensearch.test.case.*` keys only if you pass them in `attributes`.
- `score(attributes=...)` and run, case and score metadata are added to the events too, under the SDK fields: an SDK field always wins. They cannot set the evaluation fields (`gen_ai.evaluation.*`, `opensearch.gen_ai.evaluation.*`, `error.type`, and the judge keys `gen_ai.request.model`, `gen_ai.prompt.name`, `gen_ai.prompt.version`, `gen_ai.system_instructions`), even ones the SDK leaves unset: those keys are dropped with one warning per call. `score(attributes=...)` also cannot set `gen_ai.response.id`, `gen_ai.conversation.id` or `gen_ai.agent.*` (use the arguments). Metadata also cannot set `test.suite.name`, `test.suite.run.id`, `test.case.id`, `test.case.name`, `cicd.pipeline.*`, `opensearch.test.suite.run.name` / `.description` or `opensearch.test.case.*`.

## Setup

```bash
pip install "opensearch-genai-observability-sdk-py>=0.3.0"
```

Call `register()` once at startup. It sets up the OTLP traces pipeline and the OTLP logs pipeline that carries the evaluation events. `agent_name` and `agent_version` become `gen_ai.agent.name` and `gen_ai.agent.version` on every result, unless a call names another agent or the evaluated span names its own.

```python
from opensearch_genai_observability_sdk_py import register

register(
    endpoint="http://localhost:4318",  # OTLP base URL for traces and logs (the default)
    service_name="travel-agent",
    agent_name="travel_planner",
    agent_version="v3",
)
```

Endpoints follow the OTel exporter spec, one rule per signal (traces, logs):

- `endpoint=` is always the OTLP base URL (like `OTEL_EXPORTER_OTLP_ENDPOINT`). Over HTTP the SDK appends `/v1/traces` and `/v1/logs`; over gRPC it uses the base as-is for both. A base ending with `/v1/traces` or `/v1/logs` raises `ValueError`.
- `traces_endpoint=` / `logs_endpoint=` are full URLs for one signal and win over `endpoint=`. Pass both or neither (`log_exporter=` counts as the logs side).
- Code arguments always win over env vars. Per signal: `<signal>_endpoint=`, then `endpoint=`, then `OTEL_EXPORTER_OTLP_<SIGNAL>_ENDPOINT` (full URL), then `OTEL_EXPORTER_OTLP_ENDPOINT` (base), then the default base `http://localhost:4318`.
- There are no port options: the port is part of each URL. See the [README](../README.md#endpoints-traces-and-evaluation-logs) for the full order and the protocol rules.

Three setups:

(a) Local OTel Collector or the observability stack: nothing to set.

```python
register(service_name="travel-agent", agent_name="travel_planner")  # endpoint http://localhost:4318
```

For gRPC, pass `endpoint="grpc://localhost:4317"`. gRPC is for a local OTel Collector or a self-hosted Data Prepper, whose OTLP sources accept framed gRPC natively; OSI does not support it.

(b) One OTLP source for both signals under a pipeline path: an OpenSearch Ingestion (OSI) or Data Prepper `otlp` source with `traces_path: /<pipeline>/v1/traces` and `logs_path: /<pipeline>/v1/logs`, as in the [Data Prepper pipeline](#a-self-managed-data-prepper) below. Pass the pipeline base. `sigv4=` signs both exporters for OSI:

```python
register(
    endpoint="http://localhost:21890/otlp-pipeline",  # Data Prepper example below
    agent_name="travel_planner",
)

register(
    endpoint="https://<host>/<pipeline>",  # OSI
    sigv4={"service": "osis", "region": "us-east-1"},
    agent_name="travel_planner",
)
```

(c) Separate trace and log pipelines (`otel_trace_source` and `otel_logs_source`): pass both full URLs. On Data Prepper the default ports are 21890 (traces) and 21892 (logs); set `path:` (for example `/v1/traces` and `/v1/logs`) and `unframed_requests: true` on each source. On OSI, one pipeline per signal:

```python
register(
    traces_endpoint="http://data-prepper:21890/v1/traces",
    logs_endpoint="http://data-prepper:21892/v1/logs",
)

register(
    traces_endpoint="https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces",
    logs_endpoint="https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs",
    sigv4={"service": "osis", "region": "us-east-1"},
    agent_name="travel_planner",
    agent_version="v3",
)
```

Pipeline settings the client must match:

- **Compression.** `register(compression="gzip")` (or `OTEL_EXPORTER_OTLP_COMPRESSION=gzip`) must match the pipeline's `compression` setting. OSI accepts only `none` and `gzip`; other values raise `ValueError`. The argument wins over the env vars.
- **Request size.** OSI `otel_*` sources cap requests at 1 MB to 4 MB (1 MB by default with the persistent buffer). For large GenAI spans, lower `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` (and `OTEL_BLRP_MAX_EXPORT_BATCH_SIZE` for evaluation events).
- **Self-hosted Data Prepper.** Set `unframed_requests: true` and the paths (`traces_path` / `logs_path`, or `path:`), and either `ssl: false` with `http://`, or TLS with a CA the client trusts (`OTEL_EXPORTER_OTLP_CERTIFICATE=/path/to/ca.pem`).

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

### Judge context

Record what an LLM judge checked and how, so a reader can tell two judge versions apart:

| `score()` argument | `@scorer(...)` argument | Attribute |
|---|---|---|
| `evaluator_description` | `description` | `opensearch.gen_ai.evaluation.evaluator.description` |
| `judge_model` | `judge_model` | `gen_ai.request.model` |
| `judge_prompt_name` | `judge_prompt_name` | `gen_ai.prompt.name` |
| `judge_prompt_version` | `judge_prompt_version` | `gen_ai.prompt.version` |

`gen_ai.request.model`, `gen_ai.prompt.name` and `gen_ai.prompt.version` are the standard GenAI semantic convention names. There is no argument for the judge's full system prompt: see [Judge prompt](#judge-prompt).

```python
from opensearch_genai_observability_sdk_py import score

score(
    "answer_quality",
    4.0,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    threshold=4,
    score_min=1,
    score_max=5,
    evaluator="answer_quality_judge",
    evaluator_type="llm_judge",
    evaluator_description="Answer is correct and grounded in the tool output",
    judge_model="claude-haiku-4-5",
    judge_prompt_name="answer-quality",
    judge_prompt_version="v2",
)
```

### Judge prompt

Results identify the judge prompt by name and version: `gen_ai.prompt.name` and `gen_ai.prompt.version` (shown as `name@version`). Bump the version whenever the prompt text changes, so two prompt versions are never mixed under one label.

The full prompt text is never put on a result event. It belongs on the judge's own LLM call span as `gen_ai.system_instructions`, its OTel location, so it is stored once per judge call, next to the judge's input and output. The SDK does not create the judge's LLM span: the judge's instrumentation records it there (most GenAI instrumentations do, often behind a content-capture opt-in, because a prompt may hold sensitive data). If you trace the judge yourself, set it with `enrich(system_instructions=...)` inside the judge call, and pass the judge's trace id to the result: The judge span is a `chat` span: it carries no `gen_ai.agent.name`, so Agent Traces does not list the judge as an agent.

```python
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import Op, enrich, observe, score

JUDGE_PROMPT = "You grade answers of a travel agent. Score 1 to 5. Reply with JSON."


@observe(name="answer_quality_judge", op=Op.CHAT)
def judge(question: str, answer: str) -> tuple[float, str]:
    enrich(model="claude-haiku-4-5", system_instructions=JUDGE_PROMPT)  # on the judge span
    value = 4.0  # call your judge model with JUDGE_PROMPT here
    trace_id = format(trace.get_current_span().get_span_context().trace_id, "032x")
    return value, trace_id


value, judge_trace_id = judge("Plan 3 days in Lisbon", "Day 1: ...")
score(
    "answer_quality",
    value,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    threshold=4,
    score_min=1,
    score_max=5,
    evaluator_type="llm_judge",
    judge_model="claude-haiku-4-5",
    judge_prompt_name="answer-quality",
    judge_prompt_version="v2",
    judge_trace_id=judge_trace_id,
)
```

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

- A scorer receives `(input, output, expected)` and returns a number (any `numbers.Real`, for example `numpy.float32`), a bool (a verdict), an `EvalScore` or a list of `EvalScore`. Scorers are synchronous. Any other return (`numpy.bool_`, `Decimal`, `None`, a dict, a coroutine) logs a warning and emits an error result with `error.type` = `TypeError`, and the case status is `error`. An `async def` scorer is not called; wrap the async judge with `asyncio.run()` in a sync scorer.
- `@scorer(...)` declares the evaluation name, `threshold`, `direction`, `score_min`, `score_max`, `evaluator` and `evaluator_type`. Fields set on a returned `EvalScore` win.
- `EvalScore(judge_trace_id=...)` sets `opensearch.gen_ai.evaluation.judge.trace_id`, in `evaluate()` and in `Benchmark.log()`.
- `EvalScore(evaluator=..., evaluator_type=...)` names the evaluator of that score and wins over the `@scorer(...)` declaration.
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

`scores` is `{name: value}` or a list of `EvalScore`. `Benchmark` creates no spans, and it stamps the `test.*` and `cicd.pipeline.*` ids only on the results, not on the agent's spans (only `evaluate()` stamps the spans it starts).

Name the evaluator of each score with `EvalScore(evaluator=..., evaluator_type=...)` (`opensearch.gen_ai.evaluation.evaluator.name` / `.type`, the same keys `evaluate()` and `@scorer` write), or pass `Benchmark.log(evaluator=..., evaluator_type=...)` as the default for every score of that call; an `EvalScore` that sets its own wins. Without either, the result has no evaluator name, as before.

```python
b.log(
    input=question,
    output=answer,
    scores=[EvalScore(name="answer_relevancy", value=0.9, threshold=0.7)],
    evaluator="deepeval.answer_relevancy",
    evaluator_type="llm_judge",
    trace_id=trace_id,  # from ask() below
    span_id=span_id,
)
```

To link each result to its agent run, run the agent inside an `@observe(op=Op.INVOKE_AGENT)` function and return the span ids with the answer. Without `trace_id` and `span_id` the results are not linked to a trace, so the run page has no trace link and the result shows no agent input or output.

```python
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import Op, observe


@observe(name="my_agent", op=Op.INVOKE_AGENT)
def ask(question: str) -> tuple[str, str, str]:
    answer = my_agent(question)  # your agent call
    ctx = trace.get_current_span().get_span_context()
    return answer, format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


answer, trace_id, span_id = ask("Capital of France?")
```

### Expected output

The `"expected"` value of an `evaluate()` case, or `Benchmark.log(expected=...)`, is set on every `gen_ai.evaluation.result` of that case (error results included) as `opensearch.test.case.expected`:

- Strings are kept as-is. Other values are JSON-encoded, for example `{"city":"Lima"}`.
- Values longer than 32768 characters are cut to 32768 and `opensearch.test.case.expected_truncated` is set to `true`. The index stores the whole value and shows it on the result; only values up to 256 characters are indexed for filters (see [Evaluation index](#evaluation-index)).
- It is not put on the agent's spans, and online `score()` never sets it.
- It is exported by default. If your test sets hold sensitive data, pass `record_expected=False` to `evaluate()` or `Benchmark(...)`: neither attribute is written. Scorers still receive the value, and it stays on the local result. Inputs and outputs are never exported.

### Run ids

`run_name` and `run_description` on `evaluate()` and `Benchmark(...)` give a run a human-readable name (`opensearch.test.suite.run.name`, for example `"v5 shorter prompt"`) and a free-text description (`opensearch.test.suite.run.description`, cut to 1024 characters). Both are optional and go on every result of the run; OpenSearch Dashboards shows the name in place of the short run id.

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

Inside CI, every result gets `cicd.pipeline.name`, `cicd.pipeline.run.id` and `cicd.pipeline.run.url.full`, and `evaluate()` also stamps them on the agent root spans it starts (`Benchmark` only on results). Outside CI nothing is added.

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

Evaluation events arrive on the OTLP logs path, mixed with your application logs. Route by the OTLP event name: only events with `eventName == "gen_ai.evaluation.result"` go to the evaluation index, and only there (the logs route excludes them, so a result is never stored twice). Routing matches the event name field, not an `event.name` attribute; matching that attribute is a documented follow-up. OpenSearch Dashboards finds the results through the index pattern `otel-v1-genai-evaluation*`.

### Evaluation index

Results are written to the alias `otel-v1-genai-evaluation`. Create the index template, the ISM policy and the first index once per cluster, for example in Dashboards Dev Tools, before the first result arrives. The automatic option, the experimental Data Prepper [`index_type: genai-evaluation-plain`](#automatic-option-index_type-genai-evaluation-plain), creates all of this itself; the manual steps here are for Data Prepper versions without it and for other pipelines.

1. The index template. A composable template, `"version": 1`, priority 100, for the indices `otel-v1-genai-evaluation-*`. Installs from this guide and from Data Prepper use the same name and version, so they converge: Data Prepper does not overwrite an existing template with the same name and version.

<details>
<summary>otel-v1-genai-evaluation-index-template</summary>

```json
PUT _index_template/otel-v1-genai-evaluation-index-template
{
  "version": 1,
  "index_patterns": [
    "otel-v1-genai-evaluation-*"
  ],
  "priority": 100,
  "template": {
    "settings": {
      "plugins.index_state_management.rollover_alias": "otel-v1-genai-evaluation"
    },
    "mappings": {
      "date_detection": false,
      "_source": {
        "enabled": true
      },
      "dynamic_templates": [
        {
          "long_resource_attributes": {
            "mapping": {
              "type": "long"
            },
            "path_match": "resource.attributes.*",
            "match_mapping_type": "long"
          }
        },
        {
          "double_resource_attributes": {
            "mapping": {
              "type": "double"
            },
            "path_match": "resource.attributes.*",
            "match_mapping_type": "double"
          }
        },
        {
          "string_resource_attributes": {
            "mapping": {
              "type": "keyword",
              "ignore_above": 256
            },
            "path_match": "resource.attributes.*",
            "match_mapping_type": "string"
          }
        },
        {
          "long_scope_attributes": {
            "mapping": {
              "type": "long"
            },
            "path_match": "instrumentationScope.attributes.*",
            "match_mapping_type": "long"
          }
        },
        {
          "double_scope_attributes": {
            "mapping": {
              "type": "double"
            },
            "path_match": "instrumentationScope.attributes.*",
            "match_mapping_type": "double"
          }
        },
        {
          "string_scope_attributes": {
            "mapping": {
              "type": "keyword",
              "ignore_above": 256
            },
            "path_match": "instrumentationScope.attributes.*",
            "match_mapping_type": "string"
          }
        },
        {
          "long_attributes": {
            "mapping": {
              "type": "long"
            },
            "path_match": "attributes.*",
            "match_mapping_type": "long"
          }
        },
        {
          "double_attributes": {
            "mapping": {
              "type": "double"
            },
            "path_match": "attributes.*",
            "match_mapping_type": "double"
          }
        },
        {
          "string_attributes": {
            "mapping": {
              "type": "keyword",
              "ignore_above": 256
            },
            "path_match": "attributes.*",
            "match_mapping_type": "string"
          }
        }
      ],
      "properties": {
        "droppedAttributesCount": {
          "type": "integer"
        },
        "instrumentationScope": {
          "properties": {
            "droppedAttributesCount": {
              "type": "integer"
            },
            "schemaUrl": {
              "type": "keyword",
              "ignore_above": 256
            },
            "name": {
              "type": "keyword",
              "ignore_above": 128
            },
            "version": {
              "type": "keyword",
              "ignore_above": 64
            }
          }
        },
        "resource": {
          "properties": {
            "droppedAttributesCount": {
              "type": "integer"
            },
            "schemaUrl": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        "severity": {
          "properties": {
            "number": {
              "type": "integer"
            },
            "text": {
              "type": "keyword",
              "ignore_above": 32
            }
          }
        },
        "body": {
          "type": "text"
        },
        "@timestamp": {
          "type": "date_nanos"
        },
        "time": {
          "type": "date_nanos"
        },
        "observedTime": {
          "type": "date_nanos"
        },
        "traceId": {
          "type": "keyword",
          "ignore_above": 32
        },
        "spanId": {
          "type": "keyword",
          "ignore_above": 16
        },
        "flags": {
          "type": "long"
        },
        "attributes": {
          "properties": {
            "gen_ai": {
              "properties": {
                "evaluation": {
                  "properties": {
                    "score": {
                      "properties": {
                        "value": {
                          "type": "double"
                        }
                      }
                    },
                    "explanation": {
                      "type": "text"
                    }
                  }
                },
                "agent": {
                  "properties": {
                    "version": {
                      "type": "keyword",
                      "ignore_above": 256
                    }
                  }
                },
                "prompt": {
                  "properties": {
                    "version": {
                      "type": "keyword",
                      "ignore_above": 256
                    }
                  }
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
                            "min": {
                              "type": "double"
                            },
                            "max": {
                              "type": "double"
                            },
                            "threshold": {
                              "type": "double"
                            }
                          }
                        },
                        "evaluator": {
                          "properties": {
                            "version": {
                              "type": "keyword",
                              "ignore_above": 256
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            },
            "test": {
              "properties": {
                "case": {
                  "properties": {
                    "id": {
                      "type": "keyword",
                      "ignore_above": 256
                    }
                  }
                },
                "suite": {
                  "properties": {
                    "run": {
                      "properties": {
                        "id": {
                          "type": "keyword",
                          "ignore_above": 256
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

2. The ISM policy: roll over at 50 GB or after 24 hours, never delete (add a delete state if you want retention). Its `ism_template` attaches it to every new `otel-v1-genai-evaluation-*` index.

```json
PUT _plugins/_ism/policies/otel-v1-genai-evaluation-policy
{
  "policy": {
    "description": "Rollover for GenAI evaluation results",
    "default_state": "current_write_index",
    "states": [
      {
        "name": "current_write_index",
        "actions": [
          {
            "rollover": {
              "min_size": "50gb",
              "min_index_age": "24h"
            }
          }
        ]
      }
    ],
    "ism_template": {
      "index_patterns": [
        "otel-v1-genai-evaluation-*"
      ]
    }
  }
}
```

3. The first index, with the write alias:

```json
PUT otel-v1-genai-evaluation-000001
{
  "aliases": {
    "otel-v1-genai-evaluation": {"is_write_index": true}
  }
}
```

On a backend without ISM or aliases (for example OpenSearch Serverless), skip steps 2 and 3. Install a variant of the template with the same mappings, `"index_patterns": ["otel-v1-genai-evaluation"]` and no `plugins.index_state_management.rollover_alias` setting, then create the single index `otel-v1-genai-evaluation` (`PUT otel-v1-genai-evaluation`). A manual install on serverless must use this `["otel-v1-genai-evaluation"]` pattern variant: Data Prepper skips a template with the same name and version, so it would keep a `-*` pattern template that never matches the serverless index.

How the template maps the result attributes:

- `double`: `gen_ai.evaluation.score.value`, `opensearch.gen_ai.evaluation.score.min`, `.max` and `.threshold`. Other numbers map dynamically (`long` or `double`).
- `text`: `gen_ai.evaluation.explanation` (searchable, not aggregatable).
- `keyword` (`ignore_above` 256), pinned so the first value cannot pick a numeric type: `test.case.id`, `test.suite.run.id`, `gen_ai.agent.version`, `opensearch.gen_ai.evaluation.evaluator.version`, `gen_ai.prompt.version`.
- Every other string attribute, including `opensearch.test.case.expected`, the evaluator description, the case name and the CI URL, maps to `keyword` with `ignore_above` 256: the full value is stored and shown, never rejected, and only values up to 256 characters are indexed for filters and aggregations.
- `date_detection` is off, so a date-like string stays a string.

The index keeps the default limit of 1000 fields. If your own `attributes` and metadata add many keys, raise it in the template (`"index.mapping.total_fields.limit": 2000` under `template.settings`) and on the current write index (`PUT otel-v1-genai-evaluation-000001/_settings` with the same setting).

### a) Self-managed Data Prepper

Route `gen_ai.evaluation.result` log events from the OTLP source to their own sub-pipeline, copy `time` to `@timestamp`, and write to the evaluation index:

```yaml
otlp-pipeline:
  source:
    otlp:
      port: 21890
      ssl: false
      unframed_requests: true  # accept plain OTLP/HTTP from the SDK
      traces_path: "/otlp-pipeline/v1/traces"
      logs_path: "/otlp-pipeline/v1/logs"
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
        index: "otel-v1-genai-evaluation"  # the write alias (see Evaluation index)
```

`ssl: false` serves plain HTTP; with TLS, the client must trust the certificate (`OTEL_EXPORTER_OTLP_CERTIFICATE` for a private CA). If you set `compression: gzip` on the source, pass `register(compression="gzip")` too. Point the SDK at the pipeline base: `register(endpoint="http://<data-prepper-host>:21890/otlp-pipeline")` sends to `/otlp-pipeline/v1/traces` and `/otlp-pipeline/v1/logs`, the paths set on the source. `otel-logs-pipeline` and `otel-traces-pipeline` are your existing logs and traces pipelines. The sink writes to the alias, so create the [evaluation index](#evaluation-index) first; do not set `template_file` or an ISM policy on the sink.

#### Automatic option: `index_type: genai-evaluation-plain`

Experimental, not yet in a Data Prepper release: the index type is added by [opensearch-project/data-prepper#7243](https://github.com/opensearch-project/data-prepper/pull/7243) and its mappings may change. Until you run a release with it, use the manual [Evaluation index](#evaluation-index) setup above.

With `index_type: genai-evaluation-plain` the sink creates everything itself (`template_type` defaults to `index-template`):

- On backends with ISM: the composable index template `otel-v1-genai-evaluation-index-template` (version 1, priority 100, pattern `otel-v1-genai-evaluation-*`, with the rollover alias setting), the ISM policy `otel-v1-genai-evaluation-policy` (rollover at 50 GB or 24 hours), and the first index `otel-v1-genai-evaluation-000001` as the write index of the alias `otel-v1-genai-evaluation`.
- On serverless (`aws.serverless: true`): the template with the pattern `["otel-v1-genai-evaluation"]` and no ISM setting, and a plain index `otel-v1-genai-evaluation`, with no policy and no alias.
- It does not overwrite an existing template with the same name and version, so it keeps a template you installed from this guide.

The source must be `otel_logs_source` with `output_format: otel`. With the default `opensearch` output format, attributes are stored under `log.attributes.*` and the evaluation mappings do not apply. A tested pipeline:

```yaml
genai-evaluation-otlp-pipeline:
  source:
    otel_logs_source:
      port: 21892
      ssl: false
      http_path: /v1/logs
      output_format: otel
  route:
    - genai-evaluation: '/eventName == "gen_ai.evaluation.result"'
    - logs: '/eventName != "gen_ai.evaluation.result"'
  sink:
    - opensearch:
        hosts: ["https://OPENSEARCH_HOST:9200"]
        username: OPENSEARCH_USERNAME
        password: OPENSEARCH_PASSWORD
        index_type: genai-evaluation-plain
        routes: [genai-evaluation]
    - opensearch:
        hosts: ["https://OPENSEARCH_HOST:9200"]
        username: OPENSEARCH_USERNAME
        password: OPENSEARCH_PASSWORD
        index_type: log-analytics-plain
        routes: [logs]
```

This is a logs-only pipeline, so send traces to your traces pipeline and the logs here, with one full URL per signal (setup (c)): `register(traces_endpoint="http://<traces-pipeline-host>:21890/v1/traces", logs_endpoint="http://<data-prepper-host>:21892/v1/logs")`.

### b) Amazon OpenSearch Ingestion (OSI)

OSI does not have the evaluation index type yet. Use the same routing in an OSI pipeline and create the [evaluation index](#evaluation-index) (template, ISM policy, first index with the write alias) once before ingesting.

OSI accepts OTLP/HTTP only (protobuf), signed with SigV4 service `osis`; gRPC is not supported. Use `https://` endpoints with `sigv4=`. `register()` raises `ValueError` if `sigv4=` is set and a signal resolves to gRPC (a `grpc://` / `grpcs://` URL, `protocol="grpc"`, or a gRPC protocol env var).

1. Create the [evaluation index](#evaluation-index) on the domain: steps 1 to 3.

2. Create an OSI pipeline with an OTLP logs source. Pass its `/v1/logs` URL as `register(logs_endpoint=...)`, together with your traces pipeline URL as `traces_endpoint=` (setup (c)). With a single OSI `otlp` source for both signals, use setup (b) instead.

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
        index: "otel-v1-genai-evaluation"  # the write alias
        routes: ["evaluations"]
    - opensearch:
        hosts: ["https://search-my-domain-abc123.us-east-1.es.amazonaws.com"]
        aws:
          sts_role_arn: "arn:aws:iam::111122223333:role/osi-pipeline-role"
          region: "us-east-1"
        index: "logs-otel-v1-%{yyyy.MM.dd}"
        routes: ["logs"]
```

Replace the host, role and region with yours. The pipeline role needs write access to the `otel-v1-genai-evaluation` alias and the logs index pattern.

### c) Optional mappings you add yourself

These processors are not part of the default pipeline. Add them to `genai-evaluation-pipeline` (or the OSI pipeline) after `copy_values` if you ingest results from other producers.

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
- **Traces arrive, results do not.** The logs endpoint is wrong: nothing is derived from the traces URL, and a custom `exporter=` without a logs destination skips logs with a warning. Set both `traces_endpoint=` and `logs_endpoint=`, or a base `endpoint=` that serves both signals. `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` only applies when neither `logs_endpoint=` nor `endpoint=` is passed. With OSI, the logs pipeline must be a separate pipeline with an OTLP logs source, and `sigv4=` must be set so the logs are signed.
- **Results land in the logs index.** The route did not match `eventName`. Check that the route uses `/eventName == "gen_ai.evaluation.result"` and that your Data Prepper version maps the OTLP `event_name` field to `eventName` (added in opensearch-project/data-prepper#7148).
- **Field types look wrong** (strings as `text` with `.keyword` sub-fields, `time` not a date). The index was created before the index template, or the sink wrote to a plain index named `otel-v1-genai-evaluation` because the alias did not exist. Create the [evaluation index](#evaluation-index) (template, policy, first index with the alias); for an existing write index, roll it over (`POST otel-v1-genai-evaluation/_rollover`) or reindex.
- **A long `opensearch.test.case.expected` does not match a filter.** Expected: string attributes are `keyword` with `ignore_above` 256, so longer values are stored and shown but not indexed. Filter on the case id instead.
- **Mapping fails with "Limit of total fields [1000] has been exceeded".** Raise `index.mapping.total_fields.limit` as shown in [Evaluation index](#evaluation-index).
- **A run shows twice the results.** The same results were sent twice; see [Re-imports](#d-re-imports-duplicate-results).
- **No pass/fail shown.** The result has no `pass` / `fail` label. Pass `label=` or `threshold=`, or add the label mapping in [(c)](#c-optional-mappings-you-add-yourself).
- **Evaluations page missing.** Turn on both feature flags in [Verify](#verify).
