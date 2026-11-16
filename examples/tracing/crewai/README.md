# Tracing: CrewAI

The travel assistant (see [`../../shared/`](../../shared/)) built as a one-agent CrewAI
crew on Amazon Bedrock.

`register()` activates two instrumentors:

- `opentelemetry-instrumentation-crewai` records the crew, the task and the agent.
- `opentelemetry-instrumentation-bedrock` records each model call as a `chat` span with
  Bedrock's token usage (CrewAI's `bedrock/...` models call Bedrock through boto3).

The CrewAI instrumentor records no tool calls, so each tool also has the SDK's
`@observe(op=Op.EXECUTE_TOOL)`, placed below CrewAI's `@tool`.

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

With OpenAI instead: set `OPENAI_API_KEY`, use `llm="gpt-4o-mini"` and install
`opentelemetry-instrumentation-openai` in place of the Bedrock instrumentor (this
variant was not run for this example).

## What you get

Service `travel-assistant-crewai`, one trace per question:

```
crewai.workflow                      operation invoke_agent, gen_ai.agent.name=travel_assistant
  <task description>.task
    travel_assistant.agent           operation invoke_agent, gen_ai.agent.name=travel_assistant
      chat claude-haiku-4-5-...      model, tokens, input/output messages
      execute_tool get_weather       from @observe: arguments and result
      execute_tool find_events
      chat claude-haiku-4-5-...
```

## Notes

- The agent's `role` becomes `gen_ai.agent.name`, so the role is `travel_assistant`.
- The task span is named after the task description (here the user's question).
- The Bedrock instrumentor records the model without the inference-profile prefix
  (`claude-haiku-4-5-20251001-v1:0`).
- CrewAI logs a token-estimate warning on every Bedrock call; the script silences the
  `crewai.llms.base_llm` logger. The spans carry Bedrock's real token counts.
