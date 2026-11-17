# Documentation

Reference and concepts for the OpenSearch GenAI SDK for Python.

| Document | What it covers |
|---|---|
| [API reference](api-reference.md) | Every public API: `register()`, `observe()`, `Op`, `enrich()`, `mark_agent_root()`, `score()`, `evaluate()`, `Benchmark`, `OpenSearchTraceRetriever` |
| [Configuration](configuration.md) | Environment variables, endpoints for traces and evaluation logs, protocols, AWS SigV4, ingestion pipeline settings, auto-instrumented libraries |
| [Architecture](architecture.md) | How spans and evaluation events flow from the SDK through Data Prepper or an OTel Collector into OpenSearch |
| [Agent evaluations](agent-evaluations.md) | Evaluation concepts and reference: data model, attributes, what gets stored, index template, ingestion, verify, troubleshooting |
| [Getting started with agent evaluations](getting-started-evaluations.md) | Guided walkthrough from an instrumented agent to online and offline results in OpenSearch Dashboards |

## Examples

The examples in [`examples/`](../examples/), one SDK feature each:

| Example | Description |
|---|---|
| [`01_tracing_basics.py`](../examples/01_tracing_basics.py) | `@observe` decorator, context manager, `enrich()` |
| [`02_scoring.py`](../examples/02_scoring.py) | Evaluation events: inline, offline, trace-level, session-level, evaluator failures |
| [`03_aws_sigv4.py`](../examples/03_aws_sigv4.py) | AWS SigV4 for traces and evaluation events with `register(sigv4=...)` |
| [`04_async_tracing.py`](../examples/04_async_tracing.py) | Async function tracing with `@observe` |
| [`05_openai_auto_instrument.py`](../examples/05_openai_auto_instrument.py) | OpenAI auto-instrumentation via `register()` |
| [`06_retrieval_and_eval.py`](../examples/06_retrieval_and_eval.py) | Retrieve traces from OpenSearch, evaluate, write scores back |
| [`07_benchmarks.py`](../examples/07_benchmarks.py) | `evaluate()` with scorers, compare agent versions |
| [`08_upload_benchmark_results.py`](../examples/08_upload_benchmark_results.py) | `Benchmark.log()`: upload results from RAGAS, DeepEval, custom, with trace links |
| [`github-action/`](../examples/github-action/) | Reusable GitHub Action: run an `evaluate()` suite in CI, link the run in OpenSearch Dashboards, fail on score regressions |
