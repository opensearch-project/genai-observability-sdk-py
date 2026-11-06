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
- **Agent evaluations guide**: [docs/agent-evaluations.md](docs/agent-evaluations.md) covers what gets stored, online and offline evaluation, ingestion with Data Prepper or Amazon OpenSearch Ingestion, and how to verify results
- **Getting started with evaluations**: [docs/getting-started-evaluations.md](docs/getting-started-evaluations.md) is the short path from an instrumented agent to online and offline results in OpenSearch Dashboards, with deep links and copyable LLM prompts
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
register()  # local OTel Collector at http://localhost:4318


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
    endpoint="http://my-collector:4318",  # base URL for traces and logs, or use env vars
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

#### Endpoints: traces and evaluation logs

`register()` creates a `TracerProvider` for traces and a `LoggerProvider` (with a `BatchLogRecordProcessor`, or `SimpleLogRecordProcessor` with `batch=False`) for the evaluation events (`gen_ai.evaluation.result`). Endpoints follow the [OTel exporter spec](https://opentelemetry.io/docs/specs/otel/protocol/exporter/), one rule per signal:

- `endpoint=` is always the OTLP base URL for all signals, same as `OTEL_EXPORTER_OTLP_ENDPOINT`: `/v1/traces` and `/v1/logs` are appended over HTTP, and it is used as-is over gRPC. A base whose path ends with `/v1/traces` or `/v1/logs` raises `ValueError`.
- `traces_endpoint=` / `logs_endpoint=` are full URLs, same as `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`. Pass both or neither: one without the other raises `ValueError` (`log_exporter=` counts as the logs side, `exporter=` as the traces side). Nothing is derived from one signal's URL for the other.
- There are no port options: the port is part of each URL.

**Three setups:**

(a) **Local OTel Collector or the observability stack.** Nothing to set: the default base is `http://localhost:4318`.

```python
register()  # same as register(endpoint="http://localhost:4318")
register(endpoint="grpc://localhost:4317")  # gRPC: same host:port for both signals
```

(b) **One OTLP source for both signals** under a pipeline path: an OpenSearch Ingestion (OSIS) or Data Prepper `otlp` source with `traces_path: /<pipeline>/v1/traces` and `logs_path: /<pipeline>/v1/logs`. Pass the pipeline base:

```python
register(
    endpoint="https://<host>/<pipeline>",  # -> /<pipeline>/v1/traces and /<pipeline>/v1/logs
    sigv4={"service": "osis", "region": "us-east-1"},  # OSIS only
)
```

(c) **Separate trace and log pipelines** (`otel_trace_source` and `otel_logs_source`). Pass both full URLs. On Data Prepper the default ports are 21890 (traces) and 21892 (logs); set `path:` (for example `/v1/traces` and `/v1/logs`) and `unframed_requests: true` on each source:

```python
register(
    traces_endpoint="http://data-prepper:21890/v1/traces",
    logs_endpoint="http://data-prepper:21892/v1/logs",
)

# OSIS, one pipeline per signal, SigV4:
register(
    traces_endpoint="https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces",
    logs_endpoint="https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs",
    sigv4={"service": "osis", "region": "us-east-1"},  # or sigv4=True
)
```

**Endpoint resolution (per signal, priority order).** Code arguments always win over env vars, as in the OTel exporters:

1. `traces_endpoint=` / `logs_endpoint=`: full URL, used as-is
2. `endpoint=`: base URL
3. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`: full URL, used as-is
4. `OTEL_EXPORTER_OTLP_ENDPOINT`: base URL
5. The default base `http://localhost:4318` (a local OTel Collector)

For example, with `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://env:4318/v1/traces` set in the environment:

```python
register(endpoint="http://collector:4318")
# traces -> http://collector:4318/v1/traces  (endpoint= beats the env var)
# logs   -> http://collector:4318/v1/logs

register()
# traces -> http://env:4318/v1/traces
# logs   -> OTEL_EXPORTER_OTLP_LOGS_ENDPOINT, else OTEL_EXPORTER_OTLP_ENDPOINT + /v1/logs,
#           else http://localhost:4318/v1/logs
```

**Protocol per signal (priority order):**

1. `protocol=` parameter: `"http"` or `"grpc"`
2. A `grpc://` / `grpcs://` scheme on that signal's URL passed in code (`<signal>_endpoint=` or `endpoint=`)
3. `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` / `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL`
4. `OTEL_EXPORTER_OTLP_PROTOCOL`
5. That signal's URL scheme
6. Otherwise HTTP

**URL schemes:**

| URL scheme | Transport |
|---|---|
| `http://` / `https://` | HTTP OTLP (protobuf) |
| `grpc://` | gRPC (insecure) |
| `grpcs://` | gRPC (TLS) |

`http/json` is not supported. `protocol="http"` with a `grpc://` or `grpcs://` URL raises `ValueError`, as does an HTTP protocol env var with a `grpc://` URL from an env var.

Use gRPC with a local OTel Collector (default `localhost:4317`) or a self-hosted Data Prepper, whose OTLP sources accept framed gRPC natively. OpenSearch Ingestion (OSIS) does not support gRPC: use `https://` with `sigv4=`.

**Compression.** `compression="none"` or `"gzip"` applies to both exporters `register()` builds (HTTP, gRPC and SigV4). When omitted it comes from `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION` / `OTEL_EXPORTER_OTLP_LOGS_COMPRESSION`, then `OTEL_EXPORTER_OTLP_COMPRESSION`, else `none`; the argument wins over the env vars. It must match the receiving pipeline's `compression` setting: OpenSearch Ingestion accepts only `none` and `gzip`, so other values (for example `deflate`) raise `ValueError`. With SigV4 the signature covers the gzip bytes that are sent.

**Request size.** OpenSearch Ingestion `otel_*` sources cap each request at 1 MB to 4 MB (1 MB by default with the persistent buffer). GenAI spans with prompts and completions are large, so lower the batch size if exports are rejected: `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` (spans, default 512) and `OTEL_BLRP_MAX_EXPORT_BATCH_SIZE` (evaluation events). `gzip` also helps when the pipeline is set to it.

**Self-hosted Data Prepper.** For OTLP/HTTP from the SDK, set `unframed_requests: true` on the source and the paths the SDK sends to: `traces_path` / `logs_path` on an `otlp` source (setup (b)), or `path:` on `otel_trace_source` / `otel_logs_source` (setup (c)). Use `ssl: false` on the source for plain `http://`, or serve TLS with a certificate the client trusts; for a private CA, point the exporters at it with `OTEL_EXPORTER_OTLP_CERTIFICATE` (or the per-signal `OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE` / `OTEL_EXPORTER_OTLP_LOGS_CERTIFICATE`).

Headers (`headers=`) and SigV4 (`sigv4=`) apply to both exporters `register()` builds. Over gRPC, headers are sent as gRPC metadata with lowercase names (gRPC rejects uppercase keys). Without `headers=`, the exporters read the standard `OTEL_EXPORTER_OTLP_HEADERS`.

`sigv4=` makes `register()` build an `AWSSigV4OTLPExporter` and an `AWSSigV4OTLPLogExporter`, signed with botocore credentials. `service` defaults to `"osis"` (use `"es"` for OpenSearch Service direct) and `region` to the botocore region. OpenSearch Ingestion (OSIS) accepts OTLP/HTTP only (protobuf), signed with SigV4 service `osis`; gRPC is not supported. `sigv4=` therefore signs HTTP only: if a signal resolves to gRPC (a `grpc://` / `grpcs://` URL, `protocol="grpc"`, or a gRPC protocol env var), `register()` raises `ValueError` when it builds that signal's exporter. The logs pipeline needs an OTLP logs source; Data Prepper routes the evaluation events to `otel-v1-genai-evaluation-*`.

**Environment variables only:**

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<host>/<pipeline>   # base for both signals
# or one full URL per signal:
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://traces-pipe-abc.us-east-1.osis.amazonaws.com/v1/traces
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=https://logs-pipe-def.us-east-1.osis.amazonaws.com/v1/logs
```

**Custom exporters:** `exporter=` (traces) and `log_exporter=` (logs) are used as-is; nothing is derived from them. With `exporter=` and no `log_exporter=`, `register()` builds the log exporter only if a logs destination is configured (`logs_endpoint=`, `endpoint=`, `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT`); otherwise it logs a warning and evaluation events are not exported.

**Upgrading from 0.2.9:** `endpoint=` used to be the full traces URL; it is now the OTLP base URL. A value ending with `/v1/traces` (or `/v1/logs`) raises `ValueError`: remove the suffix (`endpoint="http://host:4318"`), or pass `traces_endpoint=` and `logs_endpoint=`. The default moved from `http://localhost:21890/opentelemetry` to `http://localhost:4318`. `register(exporter=AWSSigV4OTLPExporter(...))` still works for traces; use `sigv4=` to also sign the evaluation events.

| Parameter | Default | Description |
|---|---|---|
| `endpoint` | resolved | OTLP base URL for all signals (no `/v1/<signal>` suffix) |
| `traces_endpoint` | resolved | Full OTLP traces URL; pass with `logs_endpoint` |
| `logs_endpoint` | resolved | Full OTLP logs URL, for example the OSIS logs pipeline; pass with `traces_endpoint` |
| `protocol` | resolved | `"http"` or `"grpc"` for both signals |
| `compression` | resolved | `"none"` or `"gzip"` for both signals; must match the pipeline |
| `headers` | `None` | Extra headers for both exporters (lowercased gRPC metadata over gRPC); default `OTEL_EXPORTER_OTLP_HEADERS` |
| `sigv4` | `None` | `True` or `{"service": ..., "region": ...}`: sign both exporters with AWS SigV4 |
| `enable_logs` | `True` | Set up the logs pipeline for evaluation events |
| `exporter` | `None` | Custom traces `SpanExporter`, used as-is |
| `log_exporter` | `None` | Custom `LogRecordExporter`, used as-is |

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
| `label` | `str` | `gen_ai.evaluation.score.label` | Low-cardinality label (`"relevant"`). `"pass"` / `"fail"` (any case) is the verdict, stored lowercase. A non-verdict label moves to `opensearch.gen_ai.evaluation.judge_label` when `threshold` gives a verdict |
| `explanation` | `str` | `gen_ai.evaluation.explanation` | Evaluator rationale. Kept whole up to 8192 characters (safety cap, ends with `...[truncated]` beyond) |
| `response_id` | `str` | `gen_ai.response.id` | Completion ID of the evaluated response |
| `level` | `str` | `opensearch.gen_ai.evaluation.level` | `span`, `trace`, `session`, or `tool_call`. Default: `span` with a span context, `trace` with only `trace_id`, `session` with only `conversation_id` |
| `evaluator` | `str` | `opensearch.gen_ai.evaluation.evaluator.name` | Evaluator name |
| `evaluator_version` | `str` | `opensearch.gen_ai.evaluation.evaluator.version` | Evaluator version |
| `evaluator_type` | `str` | `opensearch.gen_ai.evaluation.evaluator.type` | `llm_judge`, `code`, `human`, or `external` |
| `direction` | `str` | `opensearch.gen_ai.evaluation.score.direction` | `maximize` or `minimize` |
| `score_min` / `score_max` | `float` | `opensearch.gen_ai.evaluation.score.min` / `.max` | Scale bounds |
| `threshold` | `float` | `opensearch.gen_ai.evaluation.score.threshold` | Pass threshold. `direction` defaults to `maximize` (recorded) |
| `conversation_id` | `str` | `gen_ai.conversation.id` | Evaluated session |
| `agent_name` | `str` | `gen_ai.agent.name` | Evaluated agent. Default: the active span's, else `register(agent_name=)` |
| `agent_version` | `str` | `gen_ai.agent.version` | Its version. Default: the active span's, else `register(agent_version=)` |
| `error_type` | `str` | `error.type` | Set when the evaluator itself failed |
| `judge_trace_id` | `str` | `opensearch.gen_ai.evaluation.judge.trace_id` | Trace of the judge's own run |
| `evaluator_description` | `str` | `opensearch.gen_ai.evaluation.evaluator.description` | What the evaluator checks (criteria or rubric summary) |
| `judge_model` | `str` | `gen_ai.request.model` | Model the LLM judge used |
| `judge_prompt_name` | `str` | `gen_ai.prompt.name` | Name of the judge prompt template |
| `judge_prompt_version` | `str` | `gen_ai.prompt.version` | Version of the judge prompt |
| `judge_system_instructions` | `str` | `gen_ai.system_instructions` | The judge's system prompt. Opt-in (may hold sensitive data): recorded only when passed. Cut to 8192 characters with `opensearch.gen_ai.evaluation.system_instructions_truncated` = `true` |
| `attributes` | `dict` | | Extra attributes, e.g. `test.*` grouping keys from [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79). The SDK fields win over them; keys the SDK owns (`gen_ai.evaluation.*`, `opensearch.gen_ai.evaluation.*`, `error.type`, `gen_ai.response.id`, `gen_ai.conversation.id`, `gen_ai.agent.*`, and the judge keys `gen_ai.request.model`, `gen_ai.prompt.*`, `gen_ai.system_instructions`) are dropped with one warning per call (use the matching arguments). `test.*` keys pass through |

`opensearch.gen_ai.evaluation.*` names are OpenSearch extensions that mirror the OTel `gen_ai.evaluation.*` names for fields OTel does not define yet. `score()` never sets `opensearch.test.case.expected`; pass it in `attributes` if you want it on an online result.

**Verdicts.** OpenSearch Dashboards shows pass/fail only for a declared rule; a result without one is neutral ("Scored"). Declare the rule with `label="pass"` / `"fail"` (explicit verdict) or `threshold` (+ `direction`). The verdict is `gen_ai.evaluation.score.label` = `pass` / `fail`, as in OTel's own examples. When `value` and `threshold` are given, the SDK computes it: `pass` if `value >= threshold` (`maximize`) or `value <= threshold` (`minimize`), else `fail`. A `pass` / `fail` label (any case, stored lowercase) wins over the threshold. When the result has a verdict, the label is always the verdict: a non-verdict label such as `relevant` is kept in `opensearch.gen_ai.evaluation.judge_label`. When the verdict replaces a label or overrides another source, the SDK logs one warning per evaluation name. Without a verdict, the label is stored as-is. The verdict is decided in your code; OpenSearch Dashboards shows the stored label and does not compute one.

```python
score("answer_quality", 3.0, threshold=4, score_min=1, score_max=5)  # label=fail
score("latency_s", 9.2, threshold=12, direction="minimize")  # label=pass
score("tool_selection", 1.0, label="pass")
score("relevance", 2.0, label="partially_relevant", threshold=4)  # label=fail + judge_label
```

If you configure OTel yourself instead of calling `register()`, install a global `LoggerProvider` with an OTLP log exporter, otherwise the events are dropped.

**Instrumentation scope.** The instrumentation scope is the library that emitted the event, so every SDK event has scope `opensearch-genai-observability-sdk-py` with the SDK version, and no scope attributes. The evaluator is data about the result and goes in `opensearch.gen_ai.evaluation.evaluator.*`.

Trace-level scores never derive a span id from the trace id.

### `evaluate()`

New to evaluations? Start with [Getting started with agent evaluations](docs/getting-started-evaluations.md): the agent, test suite, test case and eval run hierarchy, the online and offline paths, Dashboards deep links and copyable LLM prompts. The full reference is [docs/agent-evaluations.md](docs/agent-evaluations.md).

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
    run_name="v2 shorter prompt",  # optional, shown in place of the run id
    run_description="Shorter system prompt, same tools",
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

Each `gen_ai.evaluation.result` carries `test.suite.name`, `test.suite.run.id`, `test.case.id`, `test.case.name`, `opensearch.test.case.expected` when the case has an `"expected"` value, `gen_ai.agent.name` / `gen_ai.agent.version` when known, and the run `metadata` (reserved keys are dropped; the score's own `metadata` and the SDK attributes win). Pass/fail/error counts are not sent: readers derive them from the results of a run. The summary `evaluate()` returns is computed locally. Data Prepper routes the events to `otel-v1-genai-evaluation-*`; filter on `eventName`.

- The `test.*` ids are stamped on every root span the task starts (a span with no local parent) by a span processor that `evaluate()` installs once on the global SDK `TracerProvider`. Spans started outside `evaluate()` are not touched. Instrument the agent with the global provider (for example `@observe()` after `register()`).
- `test.suite.run.id` and `test.case.id` are proposed OTel attributes ([semantic-conventions#1262](https://github.com/open-telemetry/semantic-conventions/issues/1262), [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79)); the other `test.*` names are OTel's.
- **CI/CD.** Inside CI, every evaluation result and the agent root spans also carry the OTel `cicd.pipeline.name`, `cicd.pipeline.run.id` and `cicd.pipeline.run.url.full`, read from GitHub Actions (`GITHUB_WORKFLOW`, `GITHUB_RUN_ID`, run URL), GitLab CI (`CI_PIPELINE_NAME`, `CI_PIPELINE_ID`, `CI_PIPELINE_URL`) or Jenkins (`JOB_NAME`, `BUILD_ID`, `BUILD_URL`). Pass `ci_pipeline={"cicd.pipeline.run.id": ...}` to `evaluate()` / `Benchmark()` to set or override them. Outside CI nothing is added.
- `agent_name=` / `agent_version=` name the agent under test (see [Agent identity](#agent-identity)). They go on the root spans that don't name an agent, and on every evaluation result.
- If the task starts no span, the score events carry no trace context (the `test.*` attributes are still set).
- A case whose task raises is not scored: each declared scorer emits one error result instead, named after the scorer (its `@scorer(name=...)`, else the function name), with `error.type` set to the exception class name, no score value, no label and no explanation (the exception message is not sent), and the scorer's evaluator attributes. The run continues. A scorer that raises emits an error result the same way: `error.type`, no score value, no label and no explanation. The exception message stays local in `TestCaseResult.scorer_errors` (judge errors can carry URLs, keys or request bodies).
- Scorers run outside the agent trace, so a traced LLM judge gets its own trace.
- Each `TestCaseResult` has `status` (see below), `error_type`, `duration_ms`, `verdicts` (evaluation name to verdict), and `trace_id` / `span_id` of the agent root span when the task created one.

Each scorer is the evaluator: a named scorer function sets the grade event's `opensearch.gen_ai.evaluation.evaluator.name`. All events use the SDK instrumentation scope. `EvalScore.label`, `.explanation`, `.metadata` and `.judge_trace_id` (`opensearch.gen_ai.evaluation.judge.trace_id`) go on the event.

**Expected output.** A case's `"expected"` value (`evaluate()` data, `Benchmark.log(expected=)`) is set on every `gen_ai.evaluation.result` of the case, including error results, as `opensearch.test.case.expected`. Strings are kept as-is; other values are JSON-encoded. Values longer than 4096 characters are cut to 4096 and `opensearch.test.case.expected_truncated` is set to `true`. Both keys are reserved: metadata cannot set them. The expected value is not put on the agent's spans. `expected` is exported by default; if your test sets hold sensitive data, pass `record_expected=False` to `evaluate()` or `Benchmark(...)` and neither key is written (scorers still receive it). Inputs and outputs are never exported.

**Declaring rules.** Decorate a scorer with `@scorer(...)` to declare its evaluation name and pass rule up front. A scorer may also return a `bool` (a verdict) or an `EvalScore` with `label` (`"pass"` / `"fail"`), `passed`, `threshold`, `direction`, `score_min`, `score_max`; fields set on the `EvalScore` win over the declaration.

```python
from opensearch_genai_observability_sdk_py import scorer


@scorer(
    name="answer_quality",
    threshold=4,
    score_min=1,
    score_max=5,
    evaluator_type="llm_judge",
    description="Answer is correct and grounded in the tool output",
    judge_model="claude-haiku-4-5",
    judge_prompt_name="answer-quality",
    judge_prompt_version="v2",
)
def quality(input, output, expected):
    return judge(input, output)  # float: named "answer_quality", verdict from threshold


@scorer(name="latency_budget", direction="minimize", threshold=12)
def latency(input, output, expected):
    return seconds_taken  # pass when <= 12


@scorer(name="tool_selection")
def tools(input, output, expected):
    return EvalScore(name="tool_selection", value=hit_rate, passed=hit_rate == 1)
```

`@scorer` arguments: `name` (evaluation name of float/bool results and of the error grade; default the function name), `threshold`, `direction` (default `maximize` with a threshold), `score_min` / `score_max`, `evaluator` (`evaluator.name`; default the function name), `evaluator_type`, `description` (`opensearch.gen_ai.evaluation.evaluator.description`), `judge_model` (`gen_ai.request.model`), `judge_prompt_name` / `judge_prompt_version` (`gen_ai.prompt.name` / `.version`), and the opt-in `judge_system_instructions` (`gen_ai.system_instructions`, cut to 8192 characters). The description and judge context go on every result of the scorer, error results included. A scorer that raises emits its error grade under the declared `name`, so it lands on the same evaluation as its successful results.

**Case status** (local `TestCaseResult.status`, not sent): `error` if the task or any scorer raised; else `fail` if any score of the case has a `fail` verdict; else `pass`. A score's verdict comes from the first of a `pass` / `fail` `label`, `passed`, then `threshold` + `direction`; the same verdict is stored as the event's label, so the event and the local status always agree. Scores with no declared rule never fail a case. The summary keeps `pass_count`, `fail_count` and `error_count`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Benchmark name (`test.suite.name`). Stable across runs |
| `task` | `Callable` | Function that takes input and returns output. Decorate with `@observe()` for tracing |
| `data` | `list[dict]` | Dicts with `"input"` and optionally `"expected"` (passed to scorers and sent as `opensearch.test.case.expected`), `"case_id"`, `"case_name"` |
| `scores` | `list[Callable]` | Synchronous scorer functions. Each receives `(input, output, expected)` and returns `EvalScore`, `list[EvalScore]`, a number (any `numbers.Real`, for example `numpy.float32`), or `bool`. Any other return (`numpy.bool_`, `Decimal`, `None`, a dict, a coroutine) logs a warning and emits an error result with `error.type` = `TypeError`; `async def` scorers are not called. Optionally decorated with `@scorer(...)` |
| `metadata` | `dict` | Copied to every `gen_ai.evaluation.result` of the run. SDK-reserved keys (`test.*` ids, `cicd.pipeline.*`, `opensearch.test.case.*`, evaluation fields) are filtered |
| `run_id` | `str` | `test.suite.run.id`, used as given (default: generated). Use the same id from every CI shard so they report into one run, or an existing run's id to re-score it (send new results with the same run id). Empty or whitespace-only raises `ValueError` |
| `agent_name` / `agent_version` | `str` | Agent under test (see [Agent identity](#agent-identity)) |
| `ci_pipeline` | `dict` | `cicd.pipeline.*` overrides |
| `record_expected` | `bool` | Export `"expected"` as `opensearch.test.case.expected` (default `True`). Pass `False` if test sets hold sensitive data |
| `run_name` | `str` | Human-readable run name (`opensearch.test.suite.run.name`), for example `"v5 shorter prompt"`. Optional; OpenSearch Dashboards falls back to the short run id |
| `run_description` | `str` | Free-text run description (`opensearch.test.suite.run.description`), cut to 1024 characters. Optional |

Returns `BenchmarkResult` with `.summary` (aggregate stats) and `.cases` (per-case results).

### `Benchmark`

Upload pre-computed evaluation results from any framework (RAGAS, DeepEval, pytest, custom). Each `log()` emits one `gen_ai.evaluation.result` event per score, parented to the evaluated agent span when `trace_id` and `span_id` are given; otherwise the events have no trace context. Each result carries the `test.*` ids, `opensearch.test.case.expected` when `expected` is given (unless `Benchmark(record_expected=False)`), the agent, `cicd.pipeline.*` in CI, and the run `metadata` plus the case `metadata` passed to `log()` (reserved keys are dropped; the score's own `metadata` and the SDK attributes win). `scores` is `{name: value}` or a list of `EvalScore` (to declare `label` / `threshold` / bounds). Use when you already have results and want to visualize them in agent-health or OpenSearch Dashboards.

`error`, `error_type` and `status` set the local result: `error` or `error_type` marks the case `error`; otherwise a `fail` verdict marks it `fail`; `status` sets `pass`, `fail` or `error` explicitly. With `error_type`, every score in `scores` is emitted as an evaluator error result: `error.type` = `error_type`, no score value and no label. `close()` returns the local summary and sends nothing. No spans are created.

`Benchmark(...)` also takes `run_name` and `run_description` (`opensearch.test.suite.run.name` / `.description`, the description cut to 1024 characters), set on every result of the run.

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
| `find_evaluated_trace_ids(trace_ids)` | `set[str]` | Return subset of trace IDs that already have evaluation results (`gen_ai.evaluation.result` events in the evaluation index). A missing index returns an empty set; other search errors raise |

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
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base OTLP endpoint URL (`/v1/traces` and `/v1/logs` appended over HTTP, used as-is over gRPC) | — |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for traces (`http/protobuf`, `grpc`) | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Protocol for all signals (`http/protobuf`, `grpc`) | — |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | Full OTLP logs endpoint URL (evaluation events), e.g. the OSIS logs pipeline | see [Endpoints](#endpoints-traces-and-evaluation-logs) |
| `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` | Protocol for logs (`http/protobuf`, `grpc`) | `OTEL_EXPORTER_OTLP_PROTOCOL` |
| `OTEL_EXPORTER_OTLP_COMPRESSION` | Compression for all signals (`none`, `gzip`); must match the pipeline | `none` |
| `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION` / `OTEL_EXPORTER_OTLP_LOGS_COMPRESSION` | Compression for one signal | `OTEL_EXPORTER_OTLP_COMPRESSION` |
| `OTEL_EXPORTER_OTLP_CERTIFICATE` | CA file to trust for an `https://` / `grpcs://` endpoint with a private CA | system CAs |
| `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | Spans per export request; lower it for OSIS request size limits | `512` |
| `OTEL_SERVICE_NAME` | Service name for spans | `"default"` |
| `OPENSEARCH_PROJECT` | Project/service name (fallback) | `"default"` |
| `AWS_DEFAULT_REGION` | AWS region for SigV4 signing | auto-detected |

When no endpoint argument or env var is set, `register()` uses the OTel default base `http://localhost:4318`: traces to `http://localhost:4318/v1/traces`, evaluation events to `http://localhost:4318/v1/logs`.

## Examples

For an end-to-end walkthrough of agent evaluations (SDK, ingestion, verification), see [docs/agent-evaluations.md](docs/agent-evaluations.md).

See the [`examples/`](examples/) directory:

| Example | Description |
|---|---|
| [`01_tracing_basics.py`](examples/01_tracing_basics.py) | `@observe` decorator, context manager, `enrich()` |
| [`02_scoring.py`](examples/02_scoring.py) | Evaluation events: inline, offline, trace-level, session-level, evaluator failures |
| [`03_aws_sigv4.py`](examples/03_aws_sigv4.py) | AWS SigV4 for traces and evaluation events with `register(sigv4=...)` |
| [`04_async_tracing.py`](examples/04_async_tracing.py) | Async function tracing with `@observe` |
| [`05_openai_auto_instrument.py`](examples/05_openai_auto_instrument.py) | OpenAI auto-instrumentation via `register()` |
| [`06_retrieval_and_eval.py`](examples/06_retrieval_and_eval.py) | Retrieve traces from OpenSearch, evaluate, write scores back |
| [`07_benchmarks.py`](examples/07_benchmarks.py) | `evaluate()` with scorers, compare agent versions |
| [`08_upload_benchmark_results.py`](examples/08_upload_benchmark_results.py) | `Benchmark.log()` — upload results from RAGAS, DeepEval, custom, with trace links |
| [`github-action/`](examples/github-action/) | Reusable GitHub Action: run an `evaluate()` suite in CI, link the run in OpenSearch Dashboards, fail on score regressions |

## License

Apache-2.0
