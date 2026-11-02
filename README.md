# OpenSearch GenAI SDK

OTel-native tracing and scoring for LLM applications. Instrument your AI workflows with standard OpenTelemetry spans and submit evaluation scores — all routed to OpenSearch through a single OTLP pipeline.

## Features

- **One-line setup** — `register()` configures the full OTel pipeline (TracerProvider, exporter, auto-instrumentation)
- **`observe()`** — single decorator / context manager that creates OTel spans with [GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes
- **`enrich()`** — add model, token usage, and other GenAI attributes to the active span from anywhere in your code
- **`mark_agent_root()`** — mark the agent's span as the trace's agent root in OpenSearch Agent Traces
- **Auto-instrumentation** — automatically discovers and activates installed instrumentor packages (OpenAI, Anthropic, Bedrock, LangChain, etc.)
- **Scoring**: `score()` emits each evaluation result as an OTel `gen_ai.evaluation.result` log event, parented to the evaluated span, at span, trace, session, or tool-call level
- **Benchmarks**: `evaluate()` runs your agent against a dataset with scorers; `Benchmark` uploads results from any eval framework (RAGAS, DeepEval, pytest). Each case is its own agent trace plus one `gen_ai.evaluation.result` log event per score
- **AWS SigV4** — built-in SigV4 signing for AWS-hosted OpenSearch and Data Prepper endpoints
- **Zero lock-in** — remove a decorator and your code still works; everything is standard OTel

## Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OpenTelemetry SDK**: >=1.38.0, <2 (Logs API with `event_name`, used for evaluation events)

## Installation

```bash
pip install opensearch-genai-observability-sdk-py
```

The core package includes the OTel SDK and exporters. Auto-instrumentation of LLM libraries is opt-in — install only the providers you use:

```bash
# Single provider
pip install opensearch-genai-observability-sdk-py[openai]
pip install opensearch-genai-observability-sdk-py[anthropic]
pip install opensearch-genai-observability-sdk-py[bedrock]
pip install opensearch-genai-observability-sdk-py[langchain]

# Multiple providers
pip install "opensearch-genai-observability-sdk-py[openai,anthropic]"

# All instrumentors at once
pip install opensearch-genai-observability-sdk-py[otel-instrumentors]

# Everything
pip install opensearch-genai-observability-sdk-py[all]
```

**Available extras:** `openai`, `anthropic`, `bedrock`, `google`, `langchain`, `llamaindex`, `otel-instrumentors` (all instrumentors), `all`

## Quick Start

```python
from opensearch_genai_observability_sdk_py import register, observe, Op, enrich, score

# 1. Initialize tracing (one line)
register(endpoint="http://localhost:21890/opentelemetry/v1/traces")


# 2. Trace your functions
@observe(name="web_search", op=Op.EXECUTE_TOOL)
def search(query: str) -> list[dict]:
    return [{"title": f"Result for: {query}"}]


@observe(name="research_agent", op=Op.INVOKE_AGENT)
def research(query: str) -> str:
    results = search(query)
    enrich(model="gpt-4.1", provider="openai", input_tokens=150, output_tokens=50)
    return f"Summary of: {results}"


# 3. Use context managers for inline blocks
@observe(name="qa_pipeline", op=Op.INVOKE_AGENT)
def run(question: str) -> str:
    answer = research(question)
    with observe("safety_check", op="guardrail"):
        enrich(safe=True)
    return answer


result = run("What is OpenSearch?")

# 4. Record evaluation results (gen_ai.evaluation.result events, see score())
score(name="relevance", value=0.95, trace_id="...", span_id="...")
```

This produces the following span tree:

```
invoke_agent qa_pipeline
├── invoke_agent research_agent
│   └── execute_tool web_search
└── safety_check
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Your Application                    │
│                                                       │
│  @observe(op=Op.INVOKE_AGENT)   enrich()   score()   │
│  with observe("step", op=...)                         │
│                     │                                 │
│         opensearch-genai-observability-sdk-py         │
├──────────────────────────────────────────────────────┤
│  register()                                           │
│  ┌──────────────────────────────────────────────┐    │
│  │  TracerProvider                               │    │
│  │  ├── Resource (service.name)                  │    │
│  │  ├── BatchSpanProcessor                       │    │
│  │  │   └── OTLPSpanExporter (HTTP or gRPC)      │    │
│  │  │       └── SigV4 signing (AWS endpoints)    │    │
│  │  └── Auto-instrumentation                     │    │
│  │      ├── openai, anthropic, bedrock, ...      │    │
│  │      ├── langchain, llamaindex, haystack      │    │
│  │      └── chromadb, pinecone, qdrant, ...      │    │
│  │  LoggerProvider (evaluation events)           │    │
│  │  └── BatchLogRecordProcessor                  │    │
│  │      └── OTLPLogExporter (logs endpoint)   │    │
│  └──────────────────────────────────────────────┘    │
└───────────────────────┬──────────────────────────────┘
                        │ OTLP (HTTP/gRPC): /v1/traces + /v1/logs
                        ▼
               ┌─────────────────┐
               │  Data Prepper /  │
               │  OTel Collector  │
               └────────┬────────┘
                        │
                        ▼
               ┌──────────────────────────────┐
               │   OpenSearch                  │
               │  ├── otel-v1-apm-span-*       │
               │  └── otel-v1-genai-evaluation │
               └──────────────────────────────┘
```

## API Reference

### `register()`

Configures the OTel tracing pipeline, plus an OTLP logs pipeline for the evaluation events emitted by `score()`. Call once at startup.

```python
register(
    endpoint="http://my-collector:4318/v1/traces",  # or use env vars
    service_name="my-app",
    batch=True,  # BatchSpanProcessor (True) or Simple (False)
    auto_instrument=True,  # discover installed instrumentor packages
    agent_name="travel_planner",  # optional: the agent this process evaluates
    agent_version="v3",  # optional: its version
)
```

#### Agent identity

`agent_name` / `agent_version` are stamped as `gen_ai.agent.name` and `gen_ai.agent.version` on every `gen_ai.evaluation.result` event the SDK emits. OpenSearch Dashboards uses them to scope the Evaluations page to one agent and to compare runs version to version. Both are OpenTelemetry GenAI attributes (`gen_ai.agent.name`, `gen_ai.agent.version`).

Per field, the SDK uses the first of:

1. the explicit argument (`score(agent_name=, agent_version=)`, `evaluate(...)`, `Benchmark(...)`);
2. the evaluated span's own attribute: the active span for inline `score()`, the agent root span for `evaluate()`;
3. the `register()` default.

An unknown agent is left out, never invented. `evaluate()` also stamps the agent on the agent root spans where the span names none itself, so traces can be filtered by agent version.

**Endpoint resolution (priority order):**

1. `endpoint=` parameter — full URL, used as-is
2. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` env var — full URL, used as-is
3. `OTEL_EXPORTER_OTLP_ENDPOINT` env var — base URL, `/v1/traces` appended automatically
4. `http://localhost:21890/opentelemetry/v1/traces` — Data Prepper default

**Protocol resolution (priority order):**

1. `protocol=` parameter — `"http"` or `"grpc"`
2. `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` env var
3. `OTEL_EXPORTER_OTLP_PROTOCOL` env var
4. Inferred from URL scheme

**URL schemes:**

| URL scheme | Transport |
|---|---|
| `http://` / `https://` | HTTP OTLP (protobuf) |
| `grpc://` | gRPC (insecure) |
| `grpcs://` | gRPC (TLS) |

`http/json` is not supported. A `ValueError` is raised if the protocol contradicts a `grpc://` or `grpcs://` URL scheme.

**Authenticated endpoints (e.g. AWS OSIS):** pass a custom exporter via `exporter=`:

```python
from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

register(
    exporter=AWSSigV4OTLPExporter(
        endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces",
        service="osis",
    )
)
```

`AWSSigV4OTLPExporter` is HTTP-only. AWS OSIS does not expose a gRPC endpoint.

#### Endpoints: traces and evaluation logs

Traces and evaluation logs are separate OTLP endpoints. `register()` creates a `TracerProvider` for traces and a `LoggerProvider` (with a `BatchLogRecordProcessor`, or `SimpleLogRecordProcessor` with `batch=False`) for the evaluation events (`gen_ai.evaluation.result`). The log exporter uses the same protocol, headers and SigV4 settings as traces. An OTel Collector or Data Prepper serves both paths on one host; AWS OpenSearch Ingestion (OSIS) has one pipeline per signal, so you need a traces pipeline and a logs pipeline.

**Logs endpoint resolution (priority order):**

1. `logs_endpoint=` parameter: full URL, used as-is
2. `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` env var: full URL, used as-is
3. An explicit traces endpoint (`endpoint=`, or the URL of `exporter=AWSSigV4OTLPExporter(...)`): if its path ends with `/v1/traces`, the same base with `/v1/logs`. This is the only derivation. For any other path the SDK logs a warning and skips logs (traces still work).
4. `OTEL_EXPORTER_OTLP_ENDPOINT` env var: base URL, `/v1/logs` appended (per the OTel spec)
5. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` env var: same rule as 3
6. `http://localhost:21890/opentelemetry/v1/logs`: Data Prepper default

With gRPC traces only 1 and 2 apply; without them the SDK warns and skips logs. A custom `exporter=` other than `AWSSigV4OTLPExporter` also needs `log_exporter=`, else logs are skipped with a warning. The logs protocol honors `protocol=`, then `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` / `OTEL_EXPORTER_OTLP_PROTOCOL`, then the URL scheme.

**Local OTel Collector** (one host, both paths):

```python
register(
    endpoint="http://localhost:4318/v1/traces",
    logs_endpoint="http://localhost:4318/v1/logs",  # optional here: derived from /v1/traces
    service_name="my-agent",
)
```

**AWS OSIS** (two pipeline URLs, SigV4):

```python
from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter

register(
    exporter=AWSSigV4OTLPExporter(
        endpoint="https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces",
        service="osis",
    ),
    logs_endpoint="https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs",
)
```

The logs pipeline needs an OTLP logs source; Data Prepper routes the evaluation events to `otel-v1-genai-evaluation-*`. Without `logs_endpoint=` the SDK derives `/v1/logs` on the traces pipeline host, which only works if that pipeline also accepts logs. You can also pass `log_exporter=AWSSigV4OTLPLogExporter(endpoint=..., service="osis")` directly.

**Environment variables only:**

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs
# or, for a collector serving both: export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

| Parameter | Default | Description |
|---|---|---|
| `enable_logs` | `True` | Set up the logs pipeline for evaluation events |
| `logs_endpoint` | resolved | Full OTLP logs URL, for example the OSIS logs pipeline |
| `log_exporter` | `None` | Custom `LogRecordExporter`. Needed when `exporter=` is a custom exporter other than `AWSSigV4OTLPExporter` |

Existing setups are left alone: if a global `LoggerProvider` is already configured, `register()` does not replace it and `score()` uses yours. With `set_global=False` no logs pipeline is created. The returned `TracerProvider`'s `force_flush()` and `shutdown()` also flush and shut down the logs pipeline, so `provider.force_flush()` before exit covers both signals.

### `observe()`

Single tracing primitive — works as both a **decorator** and a **context manager**. Creates an OTel span with GenAI semantic convention attributes.

**As a decorator:**

```python
@observe(name="planner", op=Op.INVOKE_AGENT)
def plan(query: str) -> str:
    enrich(model="gpt-4.1")
    return call_llm(query)


# Without parentheses (uses function name, no op)
@observe
def my_function(): ...
```

**As a context manager:**

```python
with observe("thinking", op=Op.CHAT) as span:
    enrich(model="gpt-4.1", input_tokens=1500)
    result = call_llm(prompt)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | Function `__qualname__` (decorator) or `"unnamed"` (context manager) | Span name |
| `op` | `str` | `None` | `gen_ai.operation.name` value. Use `Op` constants or any custom string |
| `kind` | `SpanKind` | `INTERNAL` | OTel span kind. Use `SpanKind.CLIENT` for external service calls |

**Span naming:** When `op` is a well-known value, the span name is `"{op} {name}"` (e.g. `"invoke_agent planner"`). Custom ops follow the same pattern.

**Attributes set automatically:**

| Attribute | When set |
|---|---|
| `gen_ai.operation.name` | When `op` is provided |
| `gen_ai.agent.name` | All ops except `execute_tool` |
| `gen_ai.tool.name` | When `op=Op.EXECUTE_TOOL` |
| `gen_ai.input.messages` / `gen_ai.output.messages` | All ops except `execute_tool` (decorator only) |
| `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` | When `op=Op.EXECUTE_TOOL` (decorator only) |

**Supported function types:** sync, async, generators, async generators. Errors are captured as span status + exception events.

### `Op`

Constants for well-known `gen_ai.operation.name` values. Any custom string is also accepted.

| Constant | Value | Use for |
|---|---|---|
| `Op.CHAT` | `"chat"` | LLM chat completions |
| `Op.INVOKE_AGENT` | `"invoke_agent"` | Agent invocations |
| `Op.CREATE_AGENT` | `"create_agent"` | Agent creation/setup |
| `Op.EXECUTE_TOOL` | `"execute_tool"` | Tool/function calls |
| `Op.RETRIEVAL` | `"retrieval"` | RAG retrieval steps |
| `Op.EMBEDDINGS` | `"embeddings"` | Embedding generation |
| `Op.GENERATE_CONTENT` | `"generate_content"` | Content generation |
| `Op.TEXT_COMPLETION` | `"text_completion"` | Text completions |

Custom strings work too: `@observe(name="check", op="guardrail")`.

### `enrich()`

Add GenAI semantic convention attributes to the currently active span. Call from inside an `@observe`-decorated function or a `with observe(...)` block.

```python
@observe(name="chat", op=Op.CHAT)
def chat(prompt: str) -> str:
    result = call_llm(prompt)
    enrich(
        model="gpt-4.1",
        provider="openai",
        input_tokens=150,
        output_tokens=50,
        temperature=0.7,
    )
    return result
```

**Parameters:**

| Parameter | Attribute | Description |
|---|---|---|
| `model` | `gen_ai.request.model` | Model name |
| `provider` | `gen_ai.provider.name` | Provider name (openai, anthropic, etc.) |
| `input_tokens` | `gen_ai.usage.input_tokens` | Input token count |
| `output_tokens` | `gen_ai.usage.output_tokens` | Output token count |
| `total_tokens` | `gen_ai.usage.total_tokens` | Total token count |
| `response_id` | `gen_ai.response.id` | Response/completion ID |
| `finish_reason` | `gen_ai.response.finish_reasons` | Finish reason(s) |
| `temperature` | `gen_ai.request.temperature` | Temperature setting |
| `max_tokens` | `gen_ai.request.max_tokens` | Max tokens setting |
| `session_id` | `gen_ai.session.id` | Session/conversation ID |
| `**extra` | As provided | Any additional key-value attributes |

### `mark_agent_root()`

Mark the current span as its trace's agent root. OpenSearch Agent Traces lists a trace by the span marked with `derived.gen_ai.is_root`. Data Prepper marks the outermost GenAI span of each trace for you (the root when it is a GenAI span, or the agent span below an HTTP handler), so you only need this to choose the span yourself. A span marked at the source wins.

```python
@observe(name="planner", op=Op.INVOKE_AGENT)
def plan(query: str) -> str:
    mark_agent_root()
    ...
```

Without the SDK, set the span attribute `derived.gen_ai.is_root` to `true` on the agent's span.

### `score()`

Records one evaluation result as an OTel **log-based event** named `gen_ai.evaluation.result`, per the [OTel GenAI event spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult). Use any evaluation framework you prefer (autoevals, RAGAS, custom) and submit the results through `score()`.

The event is a LogRecord with `event_name="gen_ai.evaluation.result"`, a timestamp, severity INFO, and the trace context of the **evaluated** span, so it joins that trace without adding a span. It is exported over OTLP logs (`/v1/logs`). Data Prepper routes these events to the **`otel-v1-genai-evaluation`** index (instead of `logs-otel-v1-*`), where they join spans on `traceId` / `spanId`.

```python
from opensearch_genai_observability_sdk_py import Op, observe, score

# Inline: score the active span (the event is parented to it)
with observe("chat", op=Op.CHAT):
    answer = call_llm(question)
    score("relevance", 0.92, label="relevant", evaluator="relevance_judge")

# Offline: score a stored span by id (works after the span has ended)
score(
    name="accuracy",
    value=0.95,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
    explanation="Weather data matches ground truth",
    evaluator="accuracy_judge",
    evaluator_version="2.1.0",
    evaluator_type="llm_judge",
    score_min=0.0,
    score_max=1.0,
    direction="maximize",
)

# Trace-level: only the trace id is known. No span id is derived; the event
# carries the trace id as its own traceId, with no spanId.
score(
    name="relevance",
    value=0.92,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    attributes={"test.suite.name": "nightly_eval", "test.case.id": "case_001"},
)

# Session-level and human feedback
score(
    "user_satisfaction",
    1.0,
    label="thumbs_up",
    conversation_id="session-456",
    evaluator_type="human",
)
```

**Parameters** (all keyword-only except `name` and `value`):

| Parameter | Type | Attribute | Description |
|---|---|---|---|
| `name` | `str` | `gen_ai.evaluation.name` | Metric name (e.g., `"relevance"`) |
| `value` | `float` | `gen_ai.evaluation.score.value` | Numeric score |
| `trace_id` | `str` | event `traceId` | Hex trace ID of the evaluated trace |
| `span_id` | `str` | event `spanId` | Hex span ID of the evaluated span. Without it the result is trace-level |
| `label` | `str` | `gen_ai.evaluation.score.label` | Low-cardinality label (`"relevant"`). `"pass"` / `"fail"` (any case) is the verdict |
| `explanation` | `str` | `gen_ai.evaluation.explanation` | Evaluator rationale. Kept whole up to 8192 characters (safety cap, ends with `...[truncated]` beyond) |
| `response_id` | `str` | `gen_ai.response.id` | Completion ID of the evaluated response |
| `level` | `str` | `opensearch.evaluation.level` | `span`, `trace`, `session`, or `tool_call`. Default: `span` with a span context, `trace` with only `trace_id`, `session` with only `conversation_id` |
| `evaluator` | `str` | `opensearch.evaluation.evaluator.name` | Evaluator name |
| `evaluator_version` | `str` | `opensearch.evaluation.evaluator.version` | Evaluator version |
| `evaluator_type` | `str` | `opensearch.evaluation.evaluator.type` | `llm_judge`, `code`, `human`, or `external` |
| `direction` | `str` | `opensearch.evaluation.score.direction` | `maximize` or `minimize` |
| `score_min` / `score_max` | `float` | `opensearch.evaluation.score.min` / `.max` | Scale bounds |
| `threshold` | `float` | `opensearch.evaluation.score.threshold` | Pass threshold. `direction` defaults to `maximize` (recorded) |
| `conversation_id` | `str` | `gen_ai.conversation.id` | Evaluated session |
| `agent_name` | `str` | `gen_ai.agent.name` | Evaluated agent. Default: the active span's, else `register(agent_name=)` |
| `agent_version` | `str` | `gen_ai.agent.version` | Its version. Default: the active span's, else `register(agent_version=)` |
| `error_type` | `str` | `error.type` | Set when the evaluator itself failed |
| `judge_trace_id` | `str` | `opensearch.evaluation.judge.trace_id` | Trace of the judge's own run |
| `attributes` | `dict` | as-is | Extra attributes, e.g. `test.*` grouping keys from [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79) |

`opensearch.evaluation.*` names are extensions for proposals that have not merged upstream yet; they will move to `gen_ai.*` names when they do.

**Verdicts.** OpenSearch Dashboards shows pass/fail only for a declared rule; a result without one is neutral ("Scored"). Declare the rule with `label="pass"` / `"fail"` (explicit verdict) or `threshold` (+ `direction`). The verdict is `gen_ai.evaluation.score.label` = `pass` / `fail`, as in OTel's own examples. When `value` and `threshold` are given without a `label`, the SDK computes it: `pass` if `value >= threshold` (`maximize`) or `value <= threshold` (`minimize`), else `fail`. A `label` you set is kept over a computed verdict (readers recompute it from `threshold` + `direction`).

```python
score("answer_quality", 3.0, threshold=4, score_min=1, score_max=5)  # label=fail
score("latency_s", 9.2, threshold=12, direction="minimize")          # label=pass
score("tool_selection", 1.0, label="pass")
```

If you configure OTel yourself instead of calling `register()`, install a global `LoggerProvider` with an OTLP log exporter, otherwise the events are dropped.

**Instrumentation scope.** The instrumentation scope is the library that emitted the event, so every SDK event has scope `opensearch-genai-observability-sdk-py` with the SDK version, and no scope attributes. The evaluator is data about the result and goes in `opensearch.evaluation.evaluator.*`. Data Prepper and OpenSearch Dashboards derive the grade source (`attributes.derived.evaluation.source = opensearch-sdk`) from the scope name.

Trace-level scores never derive a span id from the trace id.

### `evaluate()`

Run a task against a dataset, score outputs, and record each score as an OTel `gen_ai.evaluation.result` log event. The SDK creates no spans of its own: each case's agent run is its own trace, exactly like production.

```python
from opensearch_genai_observability_sdk_py import evaluate, EvalScore, observe, Op


@observe(op=Op.INVOKE_AGENT)
def my_agent(question: str) -> str:
    return call_llm(question)


def accuracy(input, output, expected) -> EvalScore:
    return EvalScore(name="accuracy", value=1.0 if expected in output else 0.0)


result = evaluate(
    name="rag-agent",
    task=my_agent,
    data=[
        {"input": "What is Python?", "expected": "programming language"},
        {"input": "What causes rain?", "expected": "water vapor"},
    ],
    scores=[accuracy],
    metadata={"app.variant": "v2"},
)
```

Produces, per case:
```
trace (the agent run)
 └── span invoke_agent my_agent        root span, stamped with test.suite.name, test.suite.run.id,
      └── ... normal GenAI spans        test.case.id, test.case.name

event gen_ai.evaluation.result          one per score, parented to the agent root span,
                                        with the test.* ids copied
```

Each `gen_ai.evaluation.result` carries `test.suite.name`, `test.suite.run.id`, `test.case.id`, `test.case.name`, `gen_ai.agent.name` / `gen_ai.agent.version` when known, and the run `metadata` (reserved keys are dropped; the score's own `metadata` and the SDK attributes win). Pass/fail/error counts are not sent: readers derive them from the results of a run. The summary `evaluate()` returns is computed locally. Data Prepper routes the events to `otel-v1-genai-evaluation-*`; filter on `eventName`.

- The `test.*` ids are stamped on every root span the task starts (a span with no local parent) by a span processor that `evaluate()` installs once on the global SDK `TracerProvider`. Spans started outside `evaluate()` are not touched. Instrument the agent with the global provider (for example `@observe()` after `register()`).
- `test.suite.run.id` and `test.case.id` are proposed OTel attributes ([semantic-conventions#1262](https://github.com/open-telemetry/semantic-conventions/issues/1262), [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79)); the other `test.*` names are OTel's.
- **CI/CD.** Inside CI, every evaluation result and the agent root spans also carry the OTel `cicd.pipeline.name`, `cicd.pipeline.run.id` and `cicd.pipeline.run.url.full`, read from GitHub Actions (`GITHUB_WORKFLOW`, `GITHUB_RUN_ID`, run URL), GitLab CI (`CI_PIPELINE_NAME`, `CI_PIPELINE_ID`, `CI_PIPELINE_URL`) or Jenkins (`JOB_NAME`, `BUILD_ID`, `BUILD_URL`). Pass `ci_pipeline={"cicd.pipeline.run.id": ...}` to `evaluate()` / `Benchmark()` to set or override them. Outside CI nothing is added.
- `agent_name=` / `agent_version=` name the agent under test (see [Agent identity](#agent-identity)). They go on the root spans that don't name an agent, and on every evaluation result.
- If the task starts no span, the score events carry no trace context (the `test.*` attributes are still set).
- A case whose task raises is not scored: each declared scorer emits one error result instead, named after the scorer (its `@scorer(name=...)`, else the function name), with `error.type` set to the exception class name, no score value, no label and no explanation (the exception message is not sent), and the scorer's evaluator attributes. The run continues. A scorer that raises emits an error result with `error.type`, no score value and the exception message as the explanation.
- Scorers run outside the agent trace, so a traced LLM judge gets its own trace.
- Each `TestCaseResult` has `status` (see below), `error_type`, `duration_ms`, `verdicts` (evaluation name to verdict), and `trace_id` / `span_id` of the agent root span when the task created one.

Each scorer is the evaluator: a named scorer function sets the grade event's `opensearch.evaluation.evaluator.name`. All events use the SDK instrumentation scope. `EvalScore.label`, `.explanation`, and `.metadata` go on the event.

**Declaring rules.** Decorate a scorer with `@scorer(...)` to declare its evaluation name and pass rule up front. A scorer may also return a `bool` (a verdict) or an `EvalScore` with `label` (`"pass"` / `"fail"`), `passed`, `threshold`, `direction`, `score_min`, `score_max`; fields set on the `EvalScore` win over the declaration.

```python
from opensearch_genai_observability_sdk_py import scorer

@scorer(name="answer_quality", threshold=4, score_min=1, score_max=5, evaluator_type="llm_judge")
def quality(input, output, expected):
    return judge(input, output)          # float: named "answer_quality", verdict from threshold

@scorer(name="latency_budget", direction="minimize", threshold=12)
def latency(input, output, expected):
    return seconds_taken                 # pass when <= 12

@scorer(name="tool_selection")
def tools(input, output, expected):
    return EvalScore(name="tool_selection", value=hit_rate, passed=hit_rate == 1)
```

`@scorer` arguments: `name` (evaluation name of float/bool results and of the error grade; default the function name), `threshold`, `direction` (default `maximize` with a threshold), `score_min` / `score_max`, `evaluator` (`evaluator.name`; default the function name), `evaluator_type`. A scorer that raises emits its error grade under the declared `name`, so it lands on the same evaluation as its successful results.

**Case status** (local `TestCaseResult.status`, not sent): `error` if the task or any scorer raised; else `fail` if any score of the case has a `fail` verdict (a `pass` / `fail` `label`, `passed`, or `threshold` + `direction`); else `pass`. Scores with no declared rule never fail a case. The summary keeps `pass_count`, `fail_count` and `error_count`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Benchmark name (`test.suite.name`). Stable across runs |
| `task` | `Callable` | Function that takes input and returns output. Decorate with `@observe()` for tracing |
| `data` | `list[dict]` | Dicts with `"input"` and optionally `"expected"`, `"case_id"`, `"case_name"` |
| `scores` | `list[Callable]` | Scorer functions. Each receives `(input, output, expected)` and returns `EvalScore`, `list[EvalScore]`, `float`, or `bool`. Optionally decorated with `@scorer(...)` |
| `metadata` | `dict` | Copied to every `gen_ai.evaluation.result` of the run. SDK-reserved keys are filtered |
| `run_id` | `str` | `test.suite.run.id`, used as given (default: generated). Use the same id from every CI shard so they report into one run, or an existing run's id to re-score it (send new results with the same run id). Empty or whitespace-only raises `ValueError` |
| `agent_name` / `agent_version` | `str` | Agent under test (see [Agent identity](#agent-identity)) |
| `ci_pipeline` | `dict` | `cicd.pipeline.*` overrides |

Returns `BenchmarkResult` with `.summary` (aggregate stats) and `.cases` (per-case results).

### `Benchmark`

Upload pre-computed evaluation results from any framework (RAGAS, DeepEval, pytest, custom). Each `log()` emits one `gen_ai.evaluation.result` event per score, parented to the evaluated agent span when `trace_id` and `span_id` are given; otherwise the events have no trace context. Each result carries the `test.*` ids, the agent, `cicd.pipeline.*` in CI, and the run `metadata` plus the case `metadata` passed to `log()` (reserved keys are dropped; the score's own `metadata` and the SDK attributes win). `scores` is `{name: value}` or a list of `EvalScore` (to declare `label` / `threshold` / bounds). Use when you already have results and want to visualize them in agent-health or OpenSearch Dashboards.

`error`, `error_type` and `status` set the local result: `error` or `error_type` marks the case `error`; otherwise a `fail` verdict marks it `fail`; `status` sets `pass`, `fail` or `error` explicitly. With `error_type`, every score in `scores` is emitted as an evaluator error result: `error.type` = `error_type`, no score value and no label. `close()` returns the local summary and sends nothing. No spans are created.

Pass `run_id` to set `test.suite.run.id` (default: generated): use the same id from every CI shard so they report into one run, or an existing run's id to re-score it (send the new results with the same run id). An empty or whitespace-only `run_id` raises `ValueError`.

```python
from opensearch_genai_observability_sdk_py import Benchmark, EvalScore

# Upload results from your eval pipeline
with Benchmark(
    name="nightly-eval",
    metadata={"model": "gpt-4o"},
    agent_name="travel_planner",
    agent_version="v3",
) as b:
    b.log(input="What is Python?", output="A language", scores={"accuracy": 1.0})
    b.log(input="Capital of France?", output="Paris", scores={"accuracy": 1.0}, status="pass")

# Link to existing agent traces (click through from a failed case to the agent trace)
with Benchmark(name="ci-eval") as b:
    b.log(
        input="query",
        output="answer",
        scores={"accuracy": 0.9},
        trace_id="6ebb9835f43af1552f2cebb9f5165e39",
        span_id="89829115c2128845",
    )

# Declare verdicts: the case fails because answer_quality is below its threshold
with Benchmark(name="judged-eval") as b:
    b.log(
        input="query",
        output="answer",
        scores=[
            EvalScore(name="answer_quality", value=3.0, threshold=4, score_min=1, score_max=5),
            EvalScore(name="tool_selection", value=1.0, label="pass"),
        ],
    )

# The evaluator failed: each score is sent as an error result (error.type, no value)
with Benchmark(name="judged-eval", run_id="nightly-2026-11-01") as b:
    b.log(
        input="query",
        scores=[EvalScore(name="answer_quality", value=None)],
        error_type="TimeoutError",
    )
```

### `OpenSearchTraceRetriever`

Retrieves GenAI trace spans from OpenSearch. Works with any agent library that emits OTel GenAI semantic convention spans indexed by Data Prepper into `otel-v1-apm-span-*`.

```python
from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever

# Option 1: Basic auth (local / docker-compose)
retriever = OpenSearchTraceRetriever(
    host="https://localhost:9200",
    auth=("admin", "admin"),
    verify_certs=False,
)

# Option 2: AWS OpenSearch Service (SigV4) — use this OR Option 1, not both
import boto3
from opensearchpy import RequestsAWSV4SignerAuth

credentials = boto3.Session().get_credentials()
auth = RequestsAWSV4SignerAuth(credentials, "us-west-2", "es")
retriever = OpenSearchTraceRetriever(
    host="https://search-my-domain.us-west-2.es.amazonaws.com",
    auth=auth,
)

# Retrieve all spans for a session or trace
session = retriever.get_traces("my-conversation-id")
for trace in session.traces:
    for span in trace.spans:
        print(f"{span.operation_name}: {span.name} ({span.model})")

# List recent root spans (for discovering traces to evaluate)
roots = retriever.list_root_spans(services=["my-agent"], max_results=10)

# Filter by time
from datetime import datetime

roots = retriever.list_root_spans(services=["my-agent"], since=datetime(2026, 3, 16))

# Check which traces already have evaluation results (gen_ai.evaluation.result events)
evaluated = retriever.find_evaluated_trace_ids(["trace-id-1", "trace-id-2"])
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"https://localhost:9200"` | OpenSearch endpoint |
| `index` | `str` | `"otel-v1-apm-span-*"` | Index pattern for span data |
| `evaluation_index` | `str` | `"otel-v1-genai-evaluation*"` | Index pattern for evaluation events |
| `auth` | `tuple \| RequestsAWSV4SignerAuth` | `None` | Basic auth tuple or SigV4 auth |
| `verify_certs` | `bool` | `True` | Verify TLS certificates |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `get_traces(identifier, max_spans=10000)` | `SessionRecord` | Fetch spans by conversation ID or trace ID |
| `list_root_spans(services=None, since=None, max_results=50)` | `list[SpanRecord]` | List recent root spans, optionally filtered by service |
| `find_evaluated_trace_ids(trace_ids)` | `set[str]` | Return subset of trace IDs that already have evaluation results (`gen_ai.evaluation.result` events in the evaluation index) |

Requires the `[opensearch]` extra: `pip install opensearch-genai-observability-sdk-py[opensearch]`

## Auto-Instrumented Libraries

`register()` automatically discovers and activates any installed instrumentor packages via OTel entry points. No code changes needed — install the extras for the providers you use and their calls are traced automatically.

| Category | Extras / packages |
|---|---|
| LLM providers | `[openai]`, `[anthropic]` |
| Cloud AI | `[bedrock]`, `[google]` (Vertex AI + Generative AI) |
| Frameworks | `[langchain]`, `[llamaindex]` |
| All of the above + more | `[otel-instrumentors]` |

`[otel-instrumentors]` includes all of the above plus Cohere, Mistral, Groq, Ollama, Together, Replicate, Writer, Voyage AI, Aleph Alpha, SageMaker, watsonx, Haystack, CrewAI, Agno, MCP, Transformers, ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, LanceDB, and Marqo.

## Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Full OTLP traces endpoint URL | — |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base OTLP endpoint URL (`/v1/traces` and `/v1/logs` appended) | — |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for traces (`http/protobuf`, `grpc`) | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Protocol for all signals (`http/protobuf`, `grpc`) | — |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Full OTLP logs endpoint URL (evaluation events), e.g. the OSIS logs pipeline | see [Endpoints](#endpoints-traces-and-evaluation-logs) |
| `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` | Protocol for logs (`http/protobuf`, `grpc`) | as traces |
| `OTEL_SERVICE_NAME` | Service name for spans | `"default"` |
| `OPENSEARCH_PROJECT` | Project/service name (fallback) | `"default"` |
| `AWS_DEFAULT_REGION` | AWS region for SigV4 signing | auto-detected |

When no endpoint env var is set, `register()` defaults to the Data Prepper endpoint: `http://localhost:21890/opentelemetry/v1/traces` (evaluation events: `http://localhost:21890/opentelemetry/v1/logs`).

## Examples

See the [`examples/`](examples/) directory:

| Example | Description |
|---|---|
| [`01_tracing_basics.py`](examples/01_tracing_basics.py) | `@observe` decorator, context manager, `enrich()` |
| [`02_scoring.py`](examples/02_scoring.py) | Evaluation events: inline, offline, trace-level, session-level, evaluator failures |
| [`03_aws_sigv4.py`](examples/03_aws_sigv4.py) | AWS SigV4 authentication with `AWSSigV4OTLPExporter` |
| [`04_async_tracing.py`](examples/04_async_tracing.py) | Async function tracing with `@observe` |
| [`05_openai_auto_instrument.py`](examples/05_openai_auto_instrument.py) | OpenAI auto-instrumentation via `register()` |
| [`06_retrieval_and_eval.py`](examples/06_retrieval_and_eval.py) | Retrieve traces from OpenSearch, evaluate, write scores back |
| [`07_benchmarks.py`](examples/07_benchmarks.py) | `evaluate()` with scorers, compare agent versions |
| [`08_upload_benchmark_results.py`](examples/08_upload_benchmark_results.py) | `Benchmark.log()` — upload results from RAGAS, DeepEval, custom, with trace links |
| [`github-action/`](examples/github-action/) | Reusable GitHub Action: run an `evaluate()` suite in CI, link the run in OpenSearch Dashboards, fail on score regressions |

## License

Apache-2.0
