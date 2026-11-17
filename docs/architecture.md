# Architecture

How spans and evaluation events flow from your application through the SDK and an OTLP pipeline into OpenSearch.

```
┌──────────────────────────────────────────────────────┐
│                   Your Application                   │
│                                                      │
│  @observe(op=Op.INVOKE_AGENT)   enrich()   score()   │
│  with observe("step", op=...)   evaluate() Benchmark │
│                         │                            │
│         opensearch-genai-observability-sdk-py        │
├──────────────────────────────────────────────────────┤
│  register()                                          │
│  ┌────────────────────────────────────────────────┐  │
│  │  TracerProvider                                │  │
│  │  ├── Resource (service.name)                   │  │
│  │  ├── BatchSpanProcessor                        │  │
│  │  │   └── OTLP span exporter (HTTP or gRPC,     │  │
│  │  │       optional SigV4 over HTTP)             │  │
│  │  └── Auto-instrumentation                      │  │
│  │      ├── openai, anthropic, bedrock, ...       │  │
│  │      ├── langchain, llamaindex, haystack       │  │
│  │      └── chromadb, pinecone, qdrant, ...       │  │
│  │  LoggerProvider (evaluation events)            │  │
│  │  └── BatchLogRecordProcessor                   │  │
│  │      └── OTLP log exporter (HTTP or gRPC,      │  │
│  │          optional SigV4 over HTTP)             │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────┘
                           │ OTLP (HTTP/gRPC): /v1/traces + /v1/logs
                           ▼
                 ┌──────────────────┐
                 │  Data Prepper    │
                 │  (or OSI)        │
                 └────────┬─────────┘
                          │
                          ▼
            ┌───────────────────────────────┐
            │  OpenSearch                   │
            │  ├── otel-v1-apm-span-*       │
            │  └── otel-v1-genai-evaluation │
            └───────────────────────────────┘
```

The pipeline must be Data Prepper or Amazon OpenSearch Ingestion (OSI): it routes evaluation events by `eventName` to `otel-v1-genai-evaluation` and marks agent roots with `derived.gen_ai.is_root`. A plain OTel Collector exporting straight to OpenSearch does neither, so evaluation events land in the logs index and OpenSearch Agent Traces does not list the traces. An OTel Collector in front of Data Prepper, as in the observability stack, is fine.

## Spans

`register()` sets up a `TracerProvider` with a `BatchSpanProcessor` (`SimpleSpanProcessor` with `batch=False`) and an OTLP span exporter (HTTP or gRPC, optionally SigV4-signed), and turns on auto-instrumentation for the installed instrumentor packages. `observe()` and the instrumentors create standard OTel spans with GenAI semantic convention attributes. They are exported to `/v1/traces` and Data Prepper writes them to `otel-v1-apm-span-*`.

For example, this code:

```python
from opensearch_genai_observability_sdk_py import register, observe, Op, enrich, score

# 1. Initialize tracing (one line)
register(service_name="research-app", agent_name="qa_pipeline")  # OTel Collector, localhost:4318


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
    # 4. Record an evaluation result on the active span (a gen_ai.evaluation.result event, see score())
    score("relevance", 0.95)
    return answer


result = run("What is OpenSearch?")
```

produces this span tree:

```
invoke_agent qa_pipeline
├── invoke_agent research_agent
│   └── execute_tool web_search
└── safety_check
```

The `relevance` result is not a span: it is one `gen_ai.evaluation.result` event parented to `invoke_agent qa_pipeline`. To score a span after it has ended, pass its `trace_id=` and `span_id=` (32 and 16 hex characters) to `score()`.

Data Prepper marks the outermost GenAI span of each trace with `derived.gen_ai.is_root`, which OpenSearch Agent Traces uses to list the trace. `mark_agent_root()` marks a span at the source instead (see [`mark_agent_root()`](api-reference.md#mark_agent_root)).

## Evaluation events

`score()`, `evaluate()` and `Benchmark` emit each result as an OTel log event named `gen_ai.evaluation.result`. `register()` sets up a `LoggerProvider` with a `BatchLogRecordProcessor` and an OTLP log exporter for them. The event carries the trace context of the evaluated span, so it joins that trace without adding a span. It is exported to `/v1/logs`, and Data Prepper routes it (`eventName == "gen_ai.evaluation.result"`) to the `otel-v1-genai-evaluation` alias instead of `logs-otel-v1-*`, where it joins spans on `traceId` / `spanId`.

`evaluate()` creates no spans of its own: each case's agent run is its own trace, and the SDK stamps the `test.*` ids on its root span. What each event holds is in [agent-evaluations.md, What gets stored](agent-evaluations.md#what-gets-stored); the pipelines are in [Ingestion](agent-evaluations.md#ingestion).

## Endpoints

Both signals go to one OTLP base URL by default (`http://localhost:4318`), or to one full URL each. The rules are in [configuration.md](configuration.md#endpoints-traces-and-evaluation-logs).
