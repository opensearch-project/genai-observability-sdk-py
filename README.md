# OpenSearch GenAI SDK

OTel-native tracing and scoring for LLM applications. Instrument your AI workflows with standard OpenTelemetry spans and submit evaluation scores, all routed to OpenSearch through a single OTLP pipeline.

## Features

- **One-line setup**: `register()` configures the full OTel pipeline (TracerProvider, exporter, auto-instrumentation)
- **`observe()`**: one decorator and context manager that creates OTel spans with [GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes
- **`enrich()`**: add model, token usage and other GenAI attributes to the active span from anywhere in your code
- **`mark_agent_root()`**: mark the agent's span as the trace's agent root in OpenSearch Agent Traces
- **Auto-instrumentation**: discovers and activates installed instrumentor packages (OpenAI, Anthropic, Bedrock, LangChain and more)
- **Scoring**: `score()` emits each evaluation result as an OTel `gen_ai.evaluation.result` log event, parented to the evaluated span, at span, trace, session or tool-call level
- **Benchmarks**: `evaluate()` runs your agent against a dataset with scorers; `Benchmark` uploads results from any eval framework (RAGAS, DeepEval, pytest). Each case is its own agent trace plus one `gen_ai.evaluation.result` log event per score
- **AWS SigV4**: built-in SigV4 signing for AWS-hosted OpenSearch and Data Prepper endpoints
- **Zero lock-in**: remove a decorator and your code still works; everything is standard OTel

## Requirements

- **Python**: 3.10, 3.11, 3.12 or 3.13
- **OpenTelemetry SDK**: >=1.38.0, <2 (Logs API with `event_name`, used for evaluation events)

## Installation

```bash
pip install opensearch-genai-observability-sdk-py
```

Upgrading from 0.2.x? `endpoint=` is now the OTLP base URL and the default moved to `http://localhost:4318`: see [Upgrading to 0.3.0](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/configuration.md#upgrading-to-030).

The core package includes the OTel SDK and exporters. Auto-instrumentation of LLM libraries is opt-in: install the extras for the providers you use, for example `pip install "opensearch-genai-observability-sdk-py[openai,anthropic]"`.

| Extra | Installs |
|---|---|
| `openai`, `anthropic`, `bedrock`, `google`, `langchain`, `llamaindex` | The instrumentor for that provider or framework |
| `otel-instrumentors` | All instrumentors, including Cohere, Mistral, CrewAI, MCP and vector stores |
| `opensearch` | `opensearch-py`, for `OpenSearchTraceRetriever` |
| `all` | Everything, for development and integration testing |

The full list is in [Auto-instrumentation](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/configuration.md#auto-instrumentation).

## Quick start

### Tracing

```python
from opensearch_genai_observability_sdk_py import Op, enrich, observe, register

register(service_name="research-app")  # OTLP to http://localhost:4318


@observe(name="research_agent", op=Op.INVOKE_AGENT)
def research(query: str) -> str:
    with observe("web_search", op=Op.EXECUTE_TOOL):
        results = [f"Result for: {query}"]
    enrich(model="gpt-4.1", provider="openai", input_tokens=150, output_tokens=50)
    return f"Summary of: {results}"


research("What is OpenSearch?")
```

This produces the span `invoke_agent research_agent` with a child `execute_tool web_search`. A longer example and the span tree are in [Architecture](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/architecture.md#spans).

### Evaluations

This continues the tracing example: it scores `research`, and the evaluation events need the `LoggerProvider` that `register()` installs.

```python
from opensearch_genai_observability_sdk_py import evaluate, scorer


@scorer(name="accuracy", threshold=1)
def accuracy(input, output, expected):
    return 1.0 if expected in output else 0.0


result = evaluate(
    name="research-agent",
    task=research,
    data=[{"input": "What is OpenSearch?", "expected": "OpenSearch"}],
    scores=[accuracy],
)
```

Each case is its own agent trace, and each score is one `gen_ai.evaluation.result` event. To score live traffic, call `score()`: see [`score()`](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/api-reference.md#score) and [Getting started with agent evaluations](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/getting-started-evaluations.md).

## Documentation

| Document | What it covers |
|---|---|
| [docs/api-reference.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/api-reference.md) | Every public API: `register()`, `observe()`, `Op`, `enrich()`, `mark_agent_root()`, `score()`, `evaluate()`, `Benchmark`, `OpenSearchTraceRetriever` |
| [docs/configuration.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/configuration.md) | Environment variables, endpoints, protocols, AWS SigV4, ingestion pipeline settings, auto-instrumented libraries |
| [docs/architecture.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/architecture.md) | How spans and evaluation events flow from the SDK into OpenSearch |
| [docs/agent-evaluations.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/agent-evaluations.md) | Evaluation concepts and reference: data model, what gets stored, index template, ingestion, verify, troubleshooting |
| [docs/getting-started-evaluations.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/getting-started-evaluations.md) | Guided walkthrough from an instrumented agent to online and offline results in OpenSearch Dashboards |
| [examples/](https://github.com/opensearch-project/genai-observability-sdk-py/tree/main/examples) | Runnable examples, one SDK feature each |

The docs index is [docs/README.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/docs/README.md).

## TypeScript SDK

Looking for the TypeScript version? See [opensearch-genai-observability-sdk-ts](https://github.com/opensearch-project/genai-observability-sdk-ts).

## Contributing

See [CONTRIBUTING.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/CONTRIBUTING.md) for the development setup, tests and the pull request process.

## Security

See [SECURITY.md](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/SECURITY.md) to report a vulnerability.

## License

This project is licensed under the [Apache License 2.0](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/LICENSE).

## Copyright

Copyright OpenSearch Contributors. See [NOTICE](https://github.com/opensearch-project/genai-observability-sdk-py/blob/main/NOTICE) for details.
