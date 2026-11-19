# Runbook: trace an agent

From nothing to an agent trace in Agent Traces. Each step names the example that shows it.
The full guide, with the reasoning behind each step, is
[docs/getting-started-evaluations.md](../../docs/getting-started-evaluations.md) (steps 2 to 5).

Commands assume the local stack and `OPENSEARCH_PASSWORD` exported from its `.env`. Quote
URLs that contain `?` (zsh treats it as a glob).

## 1. Stand up the stack

```bash
git clone https://github.com/opensearch-project/observability-stack
cd observability-stack && docker compose up -d
curl -sk -u admin:"$OPENSEARCH_PASSWORD" 'https://localhost:9200/_cluster/health?pretty'
```

OTLP goes to the collector on `localhost:4318` (HTTP) or `4317` (gRPC). Spans land in
`otel-v1-apm-span-*`. No Docker? [`basics/mini_collector_http.py`](../basics/mini_collector_http.py)
prints what the SDK sends instead (it listens on 4318, so stop the stack first).

## 2. Install and call `register()`

```bash
pip install "opensearch-genai-observability-sdk-py[opensearch]>=0.3.0"
```

```python
from opensearch_genai_observability_sdk_py import register

register(service_name="travel-assistant-strands-py")
```

Call it once, before the framework creates its first agent or client. It sends traces to
`OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`) and activates every
installed OpenTelemetry instrumentor whose library is installed (`auto_instrument=True`).
Pass `agent_name=` and `agent_version=` too if this process also emits evaluation results.

Shown in: [`basics/01_tracing_basics.py`](../basics/01_tracing_basics.py), every
[`tracing/<framework>/`](../README.md#framework-and-library-matrix) example. Remote or
AWS-hosted pipelines: [`basics/03_aws_sigv4.py`](../basics/03_aws_sigv4.py).

## 3. Instrument the agent

Pick the row for your stack:

| Your agent uses | What produces the spans | Example |
|---|---|---|
| A framework with built-in OTel (Strands, OpenAI Agents SDK, CrewAI, LangGraph) | The framework or its instrumentor, through `register()` | [tracing/strands](../tracing/strands/), [tracing/openai-agents](../tracing/openai-agents/), [tracing/crewai](../tracing/crewai/), [tracing/langgraph](../tracing/langgraph/) |
| A model SDK (OpenAI, Anthropic, Bedrock, LlamaIndex) | The SDK's instrumentor for `chat`, `observe` for the agent and tools | [tracing/openai](../tracing/openai/), [tracing/anthropic](../tracing/anthropic/), [tracing/bedrock](../tracing/bedrock/), [tracing/llamaindex](../tracing/llamaindex/) |
| Your own code | `observe` and `enrich` | [basics/01_tracing_basics.py](../basics/01_tracing_basics.py), [basics/multi_agent_planner](../basics/multi_agent_planner/) |

Rules that make Agent Traces show the run correctly:

- The agent run is one `invoke_agent` span: `@observe(name="travel_assistant", op=Op.INVOKE_AGENT)`.
  Only `invoke_agent` and `create_agent` spans carry `gen_ai.agent.name`; a `chat` span
  never names an agent.
- When the agent span is not the trace root (for example below an HTTP handler), call
  `mark_agent_root()` inside it.
- Record the question and answer on the agent span as GenAI messages, so evaluators can
  read them back: `enrich(input_messages=[{"role": "user", "parts": [{"type": "text", "content": q}]}], output_messages=...)`.
- Tools are `execute_tool` spans with `gen_ai.tool.name` and their arguments and result
  (`gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`). Wrap a tool with
  `@observe(op=Op.EXECUTE_TOOL)` if the framework does not.
- Agent Traces reads messages and tool payloads from span attributes. Some instrumentations
  write span events by default: Strands needs
  `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental,gen_ai_span_attributes_only`
  (see [end-to-end/strands-observability-agent/agent.py](../end-to-end/strands-observability-agent/agent.py)).
- One `chat` span per model call. A framework that traces its own model calls (Strands)
  plus an installed instrumentor for the same client (the Bedrock one from the `[bedrock]`
  extra) gives two. Keep one example per environment, or pass `register(auto_instrument=False)`
  when the framework traces itself ([tracing/strands](../tracing/strands/)). The Anthropic
  SDK also has built-in tracing: [tracing/anthropic](../tracing/anthropic/) turns it off.
- Async code works the same: [`basics/04_async_tracing.py`](../basics/04_async_tracing.py).
- Never put secrets in span attributes.

## 4. Run it and verify the spans

Run the agent once, then check the operations recorded for your service:

```bash
curl -sk -u admin:"$OPENSEARCH_PASSWORD" -X POST https://localhost:9200/_plugins/_ppl \
  -H 'Content-Type: application/json' \
  -d '{"query": "source=otel-v1-apm-span-* | where serviceName = '\''travel-assistant-strands-py'\'' | stats count() by `attributes.gen_ai.operation.name`"}'
```

Expect `invoke_agent`, `chat` and `execute_tool` (frameworks add their own spans, such as Strands'
event loop cycles). Then check one trace:

```bash
curl -sk -u admin:"$OPENSEARCH_PASSWORD" -X POST https://localhost:9200/_plugins/_ppl \
  -H 'Content-Type: application/json' \
  -d '{"query": "source=otel-v1-apm-span-* | where serviceName = '\''travel-assistant-strands-py'\'' | fields traceId, name, `attributes.gen_ai.agent.name`, `attributes.gen_ai.tool.name`, `attributes.gen_ai.tool.call.arguments` | head 20"}'
```

Look for: one `traceId` per run, `gen_ai.agent.name` = `travel_assistant` on the
`invoke_agent` span only, and each `execute_tool` span with its tool name and arguments.

Nothing arrived? Check the endpoint, that `register()` ran before the framework started,
and the collector counters:

```bash
curl -s http://localhost:8888/metrics | grep -E 'otelcol_(receiver_accepted|exporter_send_failed)_spans'
```

## 5. View it in Agent Traces

Open `http://localhost:5601/app/agentTraces` (add `/w/<workspaceId>` before `/app` when
workspaces are on). Each agent run is one row with its status, latency and tokens. Open it
to see the tree: `invoke_agent travel_assistant`, then each `chat` (model call) and
`execute_tool get_weather` / `execute_tool find_events` with arguments and result.

Questions to ask of a trace: did it call a tool before answering? How many model calls did
it take? Where did the time go?

## Next

Score these traces: [runbooks/evals.md](evals.md).
