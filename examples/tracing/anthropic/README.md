# Travel assistant: Anthropic Python SDK

The shared travel assistant (`../../shared/`) written directly on the
[Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python). The SDK has no
agent framework, so `agent.py` has a small tool-calling loop.

Tracing:

- `register(auto_instrument=True)` turns on the Anthropic instrumentor that the
  `[anthropic]` extra installs. Every `messages.create` call becomes a `chat` span with
  the model, token usage, input and output messages and tool definitions.
- `@observe` adds what the instrumentor cannot see: the `invoke_agent travel_assistant`
  span around the loop and one `execute_tool <name>` span per tool call (arguments and
  result).

```
invoke_agent travel_assistant
├── anthropic.chat               (chat, asks for get_weather and find_events)
├── execute_tool get_weather
├── execute_tool find_events
└── anthropic.chat               (chat, final answer)
```

The Anthropic SDK (1.14 and later) also emits its own `anthropic.messages.create` span
for each call once a tracer provider is set. With the instrumentor on, that is a second
`chat` span per call, and the token totals on the agent span double. `agent.py` creates
the client with `open_telemetry=False` to keep one span per call. To use the SDK's spans
instead, drop the `[anthropic]` extra and the `open_telemetry=False` argument (those spans
carry no message content unless `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content`).

## Run

Start the local stack first (see `../../runbooks/tracing.md`). Spans go to
`OTEL_EXPORTER_OTLP_ENDPOINT`, default `http://localhost:4318`.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py "Plan a one-day trip to Paris."
```

Without `ANTHROPIC_API_KEY` the example calls Claude on Amazon Bedrock through
`AnthropicBedrock`, with your AWS credentials:

```bash
export AWS_PROFILE=my-profile AWS_REGION=us-west-2
python agent.py "Plan a one-day trip to Paris."
```

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Anthropic API key. When unset, Bedrock is used |
| `AWS_PROFILE`, `AWS_REGION` | AWS defaults | Bedrock credentials and region |
| `MODEL_ID` | `claude-haiku-4-5` (API), `us.anthropic.claude-haiku-4-5-20251001-v1:0` (Bedrock) | Model |

## Check the spans

In OpenSearch Dashboards open Agent Traces (`/app/agentTraces`) and filter on service
`travel-assistant-anthropic-py`, or query the span index:

```bash
curl -sk -u admin:$OPENSEARCH_PASSWORD "https://localhost:9200/otel-v1-apm-span-*/_search" \
  -H 'Content-Type: application/json' -d '{
  "size": 0,
  "query": {"term": {"serviceName": "travel-assistant-anthropic-py"}},
  "aggs": {"ops": {"terms": {"field": "attributes.gen_ai.operation.name"}}}
}'
```

Expect `invoke_agent`, `chat` and `execute_tool` buckets, and one `chat` span per model call.

Set `TRACELOOP_TRACE_CONTENT=false` to keep prompts and completions off the `chat`
spans (the `@observe` spans still record tool arguments and results).
