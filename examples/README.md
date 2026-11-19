# Examples

Runnable examples for `opensearch-genai-observability-sdk-py`: tracing an agent with
`register()` and `observe()`, and evaluating it offline (`evaluate()`, `Benchmark`) and
online (`score()` on traces read back with `OpenSearchTraceRetriever`). Everything lands in
OpenSearch and shows in OpenSearch Dashboards under **Agent Traces** and **Evaluations**.

New here? Follow a runbook, each step points at the example that shows it:

- [runbooks/tracing.md](runbooks/tracing.md): stand up the stack, instrument, verify spans, view them in Agent Traces.
- [runbooks/evals.md](runbooks/evals.md): offline eval runs, online scoring, results from other eval libraries, verify results.

## The example agents

**Travel assistant** (`tracing/` and `evals/`). One small agent, built once per framework
so the results are comparable. The user asks for a short trip plan for a city; the agent
calls two tools, `get_weather(city)` and `find_events(city)`, that return fixed data from
[`shared/travel_data.json`](shared/), then answers. The only external call is the model.
Every framework example uses the same system prompt, the same tools, the agent name
`travel_assistant` and the service name `travel-assistant-<framework>-py`
(`travel-assistant-evals-<library>-py` for the eval examples); the `-py` suffix keeps them
apart from the TypeScript SDK's examples on the same stack. The eval examples
run the same 5 cases from [`shared/test_suite.jsonl`](shared/) (`id`, `name`, `input`,
`expected`), including an unknown city the agent must not invent data for.

**Observability assistant** ([`end-to-end/strands-observability-agent/`](end-to-end/strands-observability-agent/)).
The full lifecycle on one real agent: a Strands agent that answers questions about your
own OpenSearch cluster ("which services had errors in the last hour?") with read-only
tools. It is instrumented, evaluated offline with two test suites whose ground truth is
computed live (correctness, tool choice, trajectory, loops, latency, cost, safety, an LLM
judge), and evaluated online by sampling its own traces. It is the example threaded through
[docs/getting-started-evaluations.md](../docs/getting-started-evaluations.md).

**Basics** ([`basics/`](basics/)). One SDK feature per file, no framework and no model
needed (except 05):

| Example | Shows |
|---|---|
| [`01_tracing_basics.py`](basics/01_tracing_basics.py) | `@observe` decorator, context manager, `enrich()` |
| [`02_scoring.py`](basics/02_scoring.py) | `score()`: span, trace and session level results, evaluator failures |
| [`03_aws_sigv4.py`](basics/03_aws_sigv4.py) | `register(sigv4=...)` for AWS-hosted pipelines |
| [`04_async_tracing.py`](basics/04_async_tracing.py) | `@observe` on async functions |
| [`05_openai_auto_instrument.py`](basics/05_openai_auto_instrument.py) | Auto-instrumentation of an installed instrumentor (OpenAI) |
| [`06_retrieval_and_eval.py`](basics/06_retrieval_and_eval.py) | Read traces back with `OpenSearchTraceRetriever`, score them |
| [`07_benchmarks.py`](basics/07_benchmarks.py) | `evaluate()` with scorers, two agent versions |
| [`08_upload_benchmark_results.py`](basics/08_upload_benchmark_results.py) | `Benchmark.log()`: upload results computed elsewhere, with trace links |
| [`multi_agent_planner/`](basics/multi_agent_planner/) | A simulated multi-agent trace (`invoke_agent`, `chat`, `execute_tool`) |
| [`mini_collector_http.py`](basics/mini_collector_http.py) | A tiny OTLP/HTTP receiver on port 4318 that prints spans and evaluation events, no Docker (`pip install fastapi uvicorn`) |
| [`github-action/`](github-action/) | Run an `evaluate()` suite in CI, link the run, fail on regressions |

## Framework and library matrix

Tracing: the framework's own or auto instrumentation, plus `observe` where it has none.
Evals: the same travel assistant scored with that library, results sent to OpenSearch.

| Framework / library | Tracing | Evals |
|---|---|---|
| Strands Agents | [tracing/strands](tracing/strands/) | [end-to-end](end-to-end/strands-observability-agent/) |
| LangGraph | [tracing/langgraph](tracing/langgraph/) | |
| CrewAI | [tracing/crewai](tracing/crewai/) | |
| OpenAI Agents SDK | [tracing/openai-agents](tracing/openai-agents/) | |
| Amazon Bedrock (boto3 Converse) | [tracing/bedrock](tracing/bedrock/) | |
| OpenAI SDK | [tracing/openai](tracing/openai/) | |
| Anthropic SDK | [tracing/anthropic](tracing/anthropic/) | |
| LlamaIndex | [tracing/llamaindex](tracing/llamaindex/) | |
| SDK only (`evaluate()`, `score()`) | [basics](basics/) | [evals/native](evals/native/) |
| DeepEval | | [evals/deepeval](evals/deepeval/) |
| Ragas | | [evals/ragas](evals/ragas/) |
| MLflow | | [evals/mlflow](evals/mlflow/) |
| pytest | | [evals/pytest](evals/pytest/) |

The eval examples run the travel assistant themselves, so each one shows tracing and evals
together. [evals/native](evals/native/) covers both modes: an offline `evaluate()` run over
the shared suite with a run name, and online `score()` on traces found with
`OpenSearchTraceRetriever`.

## Layout

```
examples/
  README.md        this file
  runbooks/        step-by-step guides: tracing.md, evals.md
  basics/          numbered SDK basics (01_ to 08_), multi_agent_planner, mini collector
  github-action/   the GitHub Action that runs an evaluate() suite in CI
  shared/          travel_data.json and test_suite.jsonl for the travel assistant
  tracing/<name>/  one travel assistant per framework: README.md, agent.py, requirements.txt
  evals/<name>/    the travel assistant scored natively or with an eval library
  end-to-end/strands-observability-agent/   full lifecycle example
```

## Prerequisites

- Python 3.10 or later.
- A place to send telemetry. Locally, the
  [OpenSearch observability stack](https://github.com/opensearch-project/observability-stack)
  (`docker compose up -d`): OTel Collector on `localhost:4318` (HTTP) and `4317` (gRPC),
  Data Prepper, OpenSearch on `https://localhost:9200` (user `admin`, password
  `OPENSEARCH_PASSWORD` from its `.env`) and OpenSearch Dashboards on `http://localhost:5601`.
  Spans land in `otel-v1-apm-span-*`, evaluation results in the `otel-v1-genai-evaluation`
  alias. For a quick look without Docker, run [`basics/mini_collector_http.py`](basics/mini_collector_http.py).
- Model access for the framework examples. They default to Amazon Bedrock
  (`AWS_PROFILE`, `AWS_REGION`, model `us.anthropic.claude-haiku-4-5-20251001-v1:0`, override
  with each example's model id env var). Examples for OpenAI-style libraries also show the
  plain `OPENAI_API_KEY` setup. Each README lists its env vars.

## How to run

Each folder is self-contained: it has a `README.md`, its code and a `requirements.txt`.

```bash
cd examples/tracing/strands          # or any other example folder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # opensearch-genai-observability-sdk-py>=0.3.0 and the framework
python agent.py
```

Use one virtual environment per example folder. `register()` activates every installed
instrumentor, so instrumentors left over from another example can trace the same calls twice.

To run against the SDK in this checkout instead of the PyPI release, add
`pip install -e "../../..[opensearch]"` (from a folder three levels down, such as
`examples/tracing/strands`; adjust the path elsewhere).
The basics need only the SDK: `pip install "opensearch-genai-observability-sdk-py[opensearch]>=0.3.0"`,
then `python examples/basics/01_tracing_basics.py` from the repository root.

Then check that the data arrived (quote URLs that contain `?`):

```bash
curl -sk -u admin:"$OPENSEARCH_PASSWORD" -X POST https://localhost:9200/_plugins/_ppl \
  -H 'Content-Type: application/json' \
  -d '{"query": "source=otel-v1-apm-span-* | where serviceName = '\''travel-assistant-strands-py'\'' | stats count() by `attributes.gen_ai.operation.name`"}'
```

Open **Agent Traces** at `http://localhost:5601/app/agentTraces` for the traces and
**Evaluations** at `http://localhost:5601/app/agentTraces/evaluations` for the results.
