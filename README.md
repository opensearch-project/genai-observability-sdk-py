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

The core package includes the OTel SDK and exporters. Auto-instrumentation of LLM libraries is opt-in: install the extras for the providers you use, for example `pip install "opensearch-genai-observability-sdk-py[openai,anthropic]"`.

| Extra | Installs |
|---|---|
| `openai`, `anthropic`, `bedrock`, `google`, `langchain`, `llamaindex` | The instrumentor for that provider or framework |
| `otel-instrumentors` | All instrumentors, including Cohere, Mistral, CrewAI, MCP and vector stores |
| `opensearch` | `opensearch-py`, for `OpenSearchTraceRetriever` |
| `all` | Everything, for development and integration testing |

The full list is in [Auto-instrumentation](docs/configuration.md#auto-instrumentation).

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

This produces the span `invoke_agent research_agent` with a child `execute_tool web_search`. A longer example and the span tree are in [Architecture](docs/architecture.md#spans).

### Evaluations

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

Each case is its own agent trace, and each score is one `gen_ai.evaluation.result` event. To score live traffic, call `score()`: see [`score()`](docs/api-reference.md#score) and [Getting started with agent evaluations](docs/getting-started-evaluations.md).

## Documentation

| Document | What it covers |
|---|---|
| [docs/api-reference.md](docs/api-reference.md) | Every public API: `register()`, `observe()`, `Op`, `enrich()`, `mark_agent_root()`, `score()`, `evaluate()`, `Benchmark`, `OpenSearchTraceRetriever` |
| [docs/configuration.md](docs/configuration.md) | Environment variables, endpoints, protocols, AWS SigV4, ingestion pipeline settings, auto-instrumented libraries |
| [docs/architecture.md](docs/architecture.md) | How spans and evaluation events flow from the SDK into OpenSearch |
| [docs/agent-evaluations.md](docs/agent-evaluations.md) | Evaluation concepts and reference: data model, what gets stored, index template, ingestion, verify, troubleshooting |
| [docs/getting-started-evaluations.md](docs/getting-started-evaluations.md) | Guided walkthrough from an instrumented agent to online and offline results in OpenSearch Dashboards |
| [examples/README.md](examples/README.md) | Every runnable example: basics, framework tracing, eval libraries, an end-to-end agent |
| [examples/runbooks/tracing.md](examples/runbooks/tracing.md) | Step by step: stand up the stack, instrument, check the spans, open Agent Traces |
| [examples/runbooks/evals.md](examples/runbooks/evals.md) | Step by step: offline evals, online evals, checking the results |

The docs index is [docs/README.md](docs/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, tests and the pull request process.

## Security

See [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

Apache-2.0

## Copyright

Copyright OpenSearch Contributors. See [NOTICE](NOTICE) for details.
