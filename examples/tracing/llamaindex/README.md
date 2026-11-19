# Travel assistant: LlamaIndex

The shared travel assistant (`../../shared/`) on a [LlamaIndex](https://www.llamaindex.ai/)
LLM (`BedrockConverse`, Claude on Amazon Bedrock). `agent.py` drives the LLM with a small
tool-calling loop (`chat_with_tools`), so the tracing story is the same as the OpenAI and
Anthropic examples.

Tracing:

- `register(auto_instrument=True)` turns on the LlamaIndex instrumentor that the
  `[llamaindex]` extra installs. Each LLM chat call becomes a `chat` span with the input
  and output messages.
- `@observe` adds the `invoke_agent travel_assistant` span around the loop and one
  `execute_tool <name>` span per tool call (arguments and result).

```
invoke_agent travel_assistant
├── BedrockConverse.workflow     (invoke_workflow, LlamaIndex prepares the tool call)
├── BedrockConverse.workflow     (chat, asks for get_weather and find_events)
├── execute_tool get_weather
├── execute_tool find_events
├── BedrockConverse.workflow     (invoke_workflow)
└── BedrockConverse.workflow     (chat, final answer)
```

What the LlamaIndex instrumentor records for `BedrockConverse` (version 0.63):

- Span names are `<LLM class>.workflow`, and the stack maps that entity name to
  `gen_ai.agent.name`, so the chat spans carry `gen_ai.agent.name =
  BedrockConverse.workflow`. The trace's agent root is still `travel_assistant`.
- `gen_ai.request.model` is empty and there is no token usage.
- A turn that only calls tools has empty output messages (the tool calls are not
  recorded). The `execute_tool` spans have the arguments and results.

With the LlamaIndex `OpenAI` LLM the LlamaIndex instrumentor hands the chat spans to the
OpenAI instrumentor instead; install the `[openai]` extra too.

## Run

Start the local stack first (see `../../runbooks/tracing.md`). Spans go to
`OTEL_EXPORTER_OTLP_ENDPOINT`, default `http://localhost:4318`.

```bash
pip install -r requirements.txt
export AWS_PROFILE=my-profile AWS_REGION=us-west-2
python agent.py "Plan a one-day trip to Paris."
```

| Variable | Default | Purpose |
|---|---|---|
| `AWS_PROFILE` | AWS default | Credentials for Bedrock |
| `AWS_REGION` | `us-west-2` | Bedrock region |
| `MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model or inference profile |

To use another provider, swap the `llm = ...` line for any LlamaIndex LLM with tool
calling, for example `llama-index-llms-openai`:

```python
from llama_index.llms.openai import OpenAI  # reads OPENAI_API_KEY

llm = OpenAI(model="gpt-4o-mini")
```

## Check the spans

In OpenSearch Dashboards open Agent Traces (`/app/agentTraces`) and filter on service
`travel-assistant-llamaindex-py`, or query the span index:

```bash
curl -sk -u admin:$OPENSEARCH_PASSWORD "https://localhost:9200/otel-v1-apm-span-*/_search" \
  -H 'Content-Type: application/json' -d '{
  "size": 0,
  "query": {"term": {"serviceName": "travel-assistant-llamaindex-py"}},
  "aggs": {"ops": {"terms": {"field": "attributes.gen_ai.operation.name"}}}
}'
```

Expect `invoke_agent`, `chat`, `execute_tool` and `invoke_workflow` buckets.

Set `TRACELOOP_TRACE_CONTENT=false` to keep prompts and completions off the LlamaIndex
spans (the `@observe` spans still record tool arguments and results).
