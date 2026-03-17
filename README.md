# OpenSearch GenAI SDK

OTel-native tracing and scoring for LLM applications. Instrument your AI workflows with standard OpenTelemetry spans and submit evaluation scores — all routed to OpenSearch through a single OTLP pipeline.

## Features

- **One-line setup** — `register()` configures the full OTel pipeline (TracerProvider, exporter, auto-instrumentation)
- **`observe()`** — single decorator / context manager that creates OTel spans with [GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes
- **`enrich()`** — add model, token usage, and other GenAI attributes to the active span from anywhere in your code
- **Auto-instrumentation** — automatically discovers and activates installed instrumentor packages (OpenAI, Anthropic, Bedrock, LangChain, etc.)
- **Scoring** — `score()` emits evaluation metrics as OTel spans at span, trace, or session level
- **AWS SigV4** — built-in SigV4 signing for AWS-hosted OpenSearch and Data Prepper endpoints
- **Zero lock-in** — remove a decorator and your code still works; everything is standard OTel

## Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OpenTelemetry SDK**: >=1.20.0, <2

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

# 4. Submit scores (after workflow completes)
score(name="relevance", value=0.95, trace_id="...")
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
│  └──────────────────────────────────────────────┘    │
└───────────────────────┬──────────────────────────────┘
                        │ OTLP (HTTP/gRPC)
                        ▼
               ┌─────────────────┐
               │  Data Prepper /  │
               │  OTel Collector  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   OpenSearch     │
               │  ├── traces      │
               │  └── scores      │
               └─────────────────┘
```

## API Reference

### `register()`

Configures the OTel tracing pipeline. Call once at startup.

```python
register(
    endpoint="http://my-collector:4318/v1/traces",  # or use env vars
    service_name="my-app",
    batch=True,            # BatchSpanProcessor (True) or Simple (False)
    auto_instrument=True,  # discover installed instrumentor packages
)
```

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
def my_function():
    ...
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
| `provider` | `gen_ai.system` | Provider name (openai, anthropic, etc.) |
| `input_tokens` | `gen_ai.usage.input_tokens` | Input token count |
| `output_tokens` | `gen_ai.usage.output_tokens` | Output token count |
| `total_tokens` | `gen_ai.usage.total_tokens` | Total token count |
| `response_id` | `gen_ai.response.id` | Response/completion ID |
| `finish_reason` | `gen_ai.response.finish_reasons` | Finish reason(s) |
| `temperature` | `gen_ai.request.temperature` | Temperature setting |
| `max_tokens` | `gen_ai.request.max_tokens` | Max tokens setting |
| `session_id` | `gen_ai.session.id` | Session/conversation ID |
| `**extra` | As provided | Any additional key-value attributes |

### `score()`

Submits evaluation scores as OTel spans. Use any evaluation framework you prefer (autoevals, RAGAS, custom) and submit the results through `score()`.

The score span is attached to the evaluated trace so it appears in the same trace waterfall as the spans it evaluates.

**Two scoring levels:**

```python
# Span-level: score a specific span (score becomes a child of that span)
score(
    name="accuracy",
    value=0.95,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
    explanation="Weather data matches ground truth",
)

# Trace-level: score the entire trace (score attaches to the root span)
score(
    name="relevance",
    value=0.92,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    explanation="Response addresses the user's query",
    attributes={
        "test.suite.name": "nightly_eval",
        "test.case.result.status": "pass",
    },
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Metric name (e.g., `"relevance"`, `"factuality"`) |
| `value` | `float` | Numeric score |
| `trace_id` | `str` | Hex trace ID of the trace being scored |
| `span_id` | `str` | Hex span ID for span-level scoring. When omitted, attaches to root span |
| `label` | `str` | Human-readable label (`"pass"`, `"relevant"`, `"correct"`) |
| `explanation` | `str` | Evaluator justification (truncated to 500 chars) |
| `response_id` | `str` | LLM completion ID for correlation |
| `attributes` | `dict` | Additional span attributes (keys used as-is, e.g. `test.*` from [semantic-conventions#3398](https://github.com/open-telemetry/semantic-conventions/issues/3398)) |

Scores follow the OTel GenAI semantic conventions with `gen_ai.evaluation.*` attributes. Each score span also emits a `gen_ai.evaluation.result` event per the [OTel GenAI event spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult).

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

# Check which traces already have evaluation spans
evaluated = retriever.find_evaluated_trace_ids(["trace-id-1", "trace-id-2"])
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"https://localhost:9200"` | OpenSearch endpoint |
| `index` | `str` | `"otel-v1-apm-span-*"` | Index pattern for span data |
| `auth` | `tuple \| RequestsAWSV4SignerAuth` | `None` | Basic auth tuple or SigV4 auth |
| `verify_certs` | `bool` | `True` | Verify TLS certificates |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `get_traces(identifier, max_spans=10000)` | `SessionRecord` | Fetch spans by conversation ID or trace ID |
| `list_root_spans(services=None, since=None, max_results=50)` | `list[SpanRecord]` | List recent root spans, optionally filtered by service |
| `find_evaluated_trace_ids(trace_ids)` | `set[str]` | Return subset of trace IDs that already have evaluation spans |

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
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base OTLP endpoint URL (`/v1/traces` appended) | — |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for traces (`http/protobuf`, `grpc`) | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Protocol for all signals (`http/protobuf`, `grpc`) | — |
| `OTEL_SERVICE_NAME` | Service name for spans | `"default"` |
| `OPENSEARCH_PROJECT` | Project/service name (fallback) | `"default"` |
| `AWS_DEFAULT_REGION` | AWS region for SigV4 signing | auto-detected |

When no endpoint env var is set, `register()` defaults to the Data Prepper endpoint: `http://localhost:21890/opentelemetry/v1/traces`.

## Examples

See the [`examples/`](examples/) directory:

| Example | Description |
|---|---|
| [`01_tracing_basics.py`](examples/01_tracing_basics.py) | `@observe` decorator, context manager, `enrich()` |
| [`02_scoring.py`](examples/02_scoring.py) | Span-level, trace-level, and session-level scoring |
| [`03_aws_sigv4.py`](examples/03_aws_sigv4.py) | AWS SigV4 authentication with `AWSSigV4OTLPExporter` |
| [`04_async_tracing.py`](examples/04_async_tracing.py) | Async function tracing with `@observe` |
| [`05_openai_auto_instrument.py`](examples/05_openai_auto_instrument.py) | OpenAI auto-instrumentation via `register()` |
| [`06_retrieval_and_eval.py`](examples/06_retrieval_and_eval.py) | Retrieve traces from OpenSearch, evaluate, write scores back |

## License

Apache-2.0
