# Tracing: LangGraph

The travel assistant (see [`../../shared/`](../../shared/)) built as a LangGraph graph
with `ChatBedrockConverse` (Amazon Bedrock).

`register()` activates the LangChain instrumentor
(`opentelemetry-instrumentation-langchain`, from the SDK's `[langchain]` extra). It traces
the graph, every node, every model call and every tool call.

## Run

Start a local stack first (OTel Collector on `localhost:4318`, see
[`../../runbooks/tracing.md`](../../runbooks/tracing.md)).

```bash
pip install -r requirements.txt
export AWS_PROFILE=<profile with Bedrock access>   # or any standard AWS credentials
export AWS_REGION=us-west-2                         # optional, default us-west-2
export BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0   # optional
python agent.py                          # "Plan a one-day trip to Paris."
python agent.py "Plan a trip to Tokyo."  # your own question
```

Any LangChain chat model works the same way, for example `ChatOpenAI` from
`langchain-openai` with `OPENAI_API_KEY` set.

## What you get

Service `travel-assistant-langgraph-py`, one trace per question:

```
invoke_agent travel_assistant        gen_ai.agent.name=travel_assistant, tokens, conversation id
  travel_assistant.workflow
    execute_task model
      ChatBedrockConverse.chat       operation chat: model, tokens, input/output messages
    execute_task tools_condition
    execute_task tools
      execute_tool get_weather
      execute_tool find_events
    execute_task model
      ChatBedrockConverse.chat
```

## Notes

- `graph.compile(name="travel_assistant")` sets the agent name. Without it the agent is
  called `LangGraph`.
- The checkpointer's `thread_id` becomes `gen_ai.conversation.id` on the root span, so
  turns with the same thread form one session.
- The tool spans record LangChain's callback payload, not only the arguments: the
  arguments are under `inputs`, and the result is a serialized `ToolMessage`.
- The instrumentor also sets `gen_ai.agent.name` on node and tool spans (`model`,
  `tools`, `get_weather`). The `invoke_agent travel_assistant` root is still the agent run.
