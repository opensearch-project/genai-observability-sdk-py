# Travel assistant: OpenAI Python SDK

The shared travel assistant (`../../shared/`) written directly on the
[OpenAI Python SDK](https://github.com/openai/openai-python). The SDK has no agent
framework, so `agent.py` has a small tool-calling loop.

Tracing:

- `register(auto_instrument=True)` turns on the OpenAI instrumentor that the `[openai]`
  extra installs. Every `chat.completions.create` call becomes a `chat` span with the
  model, token usage, input and output messages and tool definitions.
- `@observe` adds what the instrumentor cannot see: the `invoke_agent travel_assistant`
  span around the loop and one `execute_tool <name>` span per tool call (arguments and
  result).

```
invoke_agent travel_assistant
├── openai.chat                  (chat, asks for get_weather)
├── execute_tool get_weather
├── openai.chat                  (chat, asks for find_events)
├── execute_tool find_events
└── openai.chat                  (chat, final answer)
```

## Run

Start the local stack first (see `../../runbooks/tracing.md`). Spans go to
`OTEL_EXPORTER_OTLP_ENDPOINT`, default `http://localhost:4318`.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python agent.py "Plan a one-day trip to Paris."
```

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | API key, read by `OpenAI()` |
| `OPENAI_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint, read by `OpenAI()` |
| `MODEL_ID` | `gpt-4o-mini` | Model name |

### Without an OpenAI key: Amazon Bedrock

Bedrock serves an OpenAI-compatible Chat Completions endpoint, so the same code runs
unchanged. Use a [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)
as the OpenAI key. To make a short-term key from your AWS credentials
(`AWS_PROFILE`, `AWS_REGION`):

```bash
pip install aws-bedrock-token-generator
export AWS_PROFILE=my-profile AWS_REGION=us-west-2
export OPENAI_API_KEY=$(python -c "from aws_bedrock_token_generator import provide_token; print(provide_token())")
export OPENAI_BASE_URL=https://bedrock-mantle.$AWS_REGION.api.aws/v1
export MODEL_ID=openai.gpt-oss-120b
python agent.py "Plan a one-day trip to Paris."
```

This endpoint serves the OpenAI-compatible models on Bedrock (for example `gpt-oss`), not
Claude. On this path the instrumentor reports `gen_ai.provider.name = aws.bedrock`.

## Check the spans

In OpenSearch Dashboards open Agent Traces (`/app/agentTraces`) and filter on service
`travel-assistant-openai-py`, or query the span index:

```bash
curl -sk -u admin:$OPENSEARCH_PASSWORD "https://localhost:9200/otel-v1-apm-span-*/_search" \
  -H 'Content-Type: application/json' -d '{
  "size": 0,
  "query": {"term": {"serviceName": "travel-assistant-openai-py"}},
  "aggs": {"ops": {"terms": {"field": "attributes.gen_ai.operation.name"}}}
}'
```

Expect `invoke_agent`, `chat` and `execute_tool` buckets.

Set `TRACELOOP_TRACE_CONTENT=false` to keep prompts and completions off the `chat`
spans (the `@observe` spans still record tool arguments and results).
