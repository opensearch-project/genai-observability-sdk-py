# Tracing: OpenAI Agents SDK

The travel assistant (see [`../../shared/`](../../shared/)) built with the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).

`register()` activates the OpenAI Agents instrumentor
(`opentelemetry-instrumentation-openai-agents`, from the SDK's `[openai]` extra). It
hooks into the Agents SDK's tracing and records the run, the agent, every model call and
every tool call.

## Run

Start a local stack first (OTel Collector on `localhost:4318`, see
[`../../runbooks/tracing.md`](../../runbooks/tracing.md)).

```bash
pip install -r requirements.txt
```

With OpenAI:

```bash
export OPENAI_API_KEY=<your key>
export OPENAI_MODEL=gpt-4o-mini            # optional
python agent.py
```

Without an OpenAI key, the script uses Claude on Amazon Bedrock through the Agents SDK's
LiteLLM model (`LitellmModel("bedrock/<model id>")`, the `[litellm]` extra plus `boto3`):

```bash
export AWS_PROFILE=<profile with Bedrock access>   # or any standard AWS credentials
export AWS_REGION=us-west-2                         # optional, default us-west-2
export BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0   # optional
python agent.py                          # "Plan a one-day trip to Paris."
python agent.py "Plan a trip to Tokyo."  # your own question
```

## What you get

Service `travel-assistant-openai-agents`, one trace per question:

```
Agent Workflow                       operation invoke_workflow
  travel_assistant.agent             operation invoke_agent, gen_ai.agent.name=travel_assistant
    openai.response                  operation chat: model, input messages
    get_weather.tool                 operation execute_tool: arguments and result
    find_events.tool
    openai.response
```

## Notes

- `Agent(name="travel_assistant")` sets `gen_ai.agent.name`.
- Through LiteLLM (the Bedrock path) the `openai.response` spans have no token usage and
  no output messages, and the model is `bedrock/<model id>`. The run's usage is still in
  `result.context_wrapper.usage`. The OpenAI path was not run for this example.
- Without `OPENAI_API_KEY` the Agents SDK prints
  `OPENAI_API_KEY is not set, skipping trace export` (its own exporter to the OpenAI
  dashboard). It is harmless; the OpenTelemetry spans are still exported.
- `result.to_input_list()` carries the conversation into the next question.
