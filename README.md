# OpenSearch GenAI SDK

OTel-native tracing and scoring for LLM applications. Instrument your AI workflows with standard OpenTelemetry spans and submit evaluation scores — all routed to OpenSearch through a single OTLP pipeline.

## Features

- **One-line setup** — `register()` configures the full OTel pipeline (TracerProvider, exporter, auto-instrumentation)
- **Decorators** — `@workflow`, `@task`, `@agent`, `@tool` wrap functions as OTel spans with [GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes
- **Auto-instrumentation** — automatically discovers and activates installed instrumentor packages (OpenAI, Anthropic, Bedrock, LangChain, etc.)
- **Scoring** — `score()` emits evaluation metrics as OTel spans at span, trace, or session level
- **AWS SigV4** — built-in SigV4 signing for AWS-hosted OpenSearch and Data Prepper endpoints
- **Zero lock-in** — remove a decorator and your code still works; everything is standard OTel

## Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OpenTelemetry SDK**: ≥1.20.0, <2

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
from opensearch_genai_observability_sdk_py import register, workflow, agent, tool, score

# 1. Initialize tracing (one line)
# Defaults to Data Prepper at http://localhost:21890/opentelemetry/v1/traces.
# Set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 for an OTel Collector.
register()

# 2. Decorate your functions
@tool("get_weather")
def get_weather(city: str) -> dict:
    """Fetch weather data for a city."""
    return {"city": city, "temp": 22, "condition": "sunny"}

@agent("weather_assistant")
def assistant(query: str) -> str:
    data = get_weather("Paris")
    return f"{data['condition']}, {data['temp']}C"

@workflow("weather_query")
def run(query: str) -> str:
    return assistant(query)

result = run("What's the weather?")

# 3. Submit scores (after workflow completes)
score(name="relevance", value=0.95, trace_id="...", source="llm-judge")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                    │
│                                                      │
│  @workflow ─→ @agent ─→ @tool    score()            │
│     │            │         │        │                │
│     └────────────┴─────────┴────────┘                │
│                     │                                │
│            opensearch-genai-observability-sdk-py                    │
├─────────────────────────────────────────────────────┤
│  register()                                          │
│  ┌─────────────────────────────────────────────┐    │
│  │  TracerProvider                              │    │
│  │  ├── Resource (service.name)                 │    │
│  │  ├── BatchSpanProcessor                      │    │
│  │  │   └── OTLPSpanExporter (HTTP or gRPC)     │    │
│  │  │       └── SigV4 signing (AWS endpoints)   │    │
│  │  └── Auto-instrumentation                    │    │
│  │      ├── openai, anthropic, bedrock, ...     │    │
│  │      ├── langchain, llamaindex, haystack     │    │
│  │      └── chromadb, pinecone, qdrant, ...     │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────┘
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

### Decorators

Four decorators for tracing application logic. Each creates an OTel span with `gen_ai.*` semantic convention attributes.

| Decorator | Use for | Operation name | Span name format |
|---|---|---|---|
| `@workflow("name")` | Top-level orchestration | `invoke_agent` | `name` |
| `@task("name")` | Units of work | `invoke_agent` | `name` |
| `@agent("name")` | Autonomous agent logic | `invoke_agent` | `invoke_agent name` |
| `@tool("name")` | Tool/function calls | `execute_tool` | `execute_tool name` |

All decorators accept `name` (defaults to function's `__qualname__`) and `version`.

**Attributes set automatically:**

| Attribute | Set by |
|---|---|
| `gen_ai.operation.name` | All decorators |
| `gen_ai.agent.name` / `gen_ai.tool.name` | All decorators |
| `gen_ai.input.messages` / `gen_ai.output.messages` | `@workflow`, `@task`, `@agent` |
| `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` | `@tool` |
| `gen_ai.tool.type` | `@tool` (always `"function"`) |
| `gen_ai.tool.description` | `@tool` (from docstring, if present) |
| `gen_ai.agent.version` | All decorators (when `version` is set) |

**Supported function types:** sync, async, generators, async generators. Errors are captured as span status + exception events.

```python
@agent("research_agent", version=2)
async def research(query: str) -> str:
    """Agents create invoke_agent spans with gen_ai.agent.* attributes."""
    result = await search_tool(query)
    return summarize(result)

@tool("search")
def search_tool(query: str) -> list:
    """Docstring becomes gen_ai.tool.description. Input/output use gen_ai.tool.call.* attributes."""
    return api.search(query)
```

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
    source="heuristic",
)

# Trace-level: score the entire trace (score attaches to the root span)
score(
    name="relevance",
    value=0.92,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    explanation="Response addresses the user's query",
    source="llm-judge",
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
| `source` | `str` | Score origin: `"sdk"`, `"human"`, `"llm-judge"`, `"heuristic"` |
| `metadata` | `dict` | Arbitrary key-value metadata |

Scores follow the OTel GenAI semantic conventions with `gen_ai.evaluation.*` attributes.

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

## License

Apache-2.0
