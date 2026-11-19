# Tracing: Amazon Bedrock (boto3 Converse)

The travel assistant (see [`../../shared/`](../../shared/)) written directly against the
Amazon Bedrock Converse API with `boto3`, with no agent framework. The script runs the
tool loop itself.

- `register()` activates the Bedrock instrumentor (`opentelemetry-instrumentation-bedrock`,
  from the SDK's `[bedrock]` extra). Each `converse()` call becomes a `chat` span with
  Bedrock's token usage and the messages.
- The SDK's `@observe` adds what no instrumentor can see: the agent run
  (`@observe(name="travel_assistant", op=Op.INVOKE_AGENT)`) and your tool functions
  (`@observe(op=Op.EXECUTE_TOOL)`, which records arguments and result).
- `enrich()` puts the model, the conversation id, the summed tokens and the question and
  answer (GenAI message format) on the agent span.

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

The IAM identity needs `bedrock:InvokeModel` on the model (Converse uses it), and the
model must be enabled in the region.

## What you get

Service `travel-assistant-bedrock-py`, one trace per question:

```
invoke_agent travel_assistant        gen_ai.agent.name=travel_assistant, tokens, messages
  chat claude-haiku-4-5-...          from the Bedrock instrumentor: model, tokens, messages
  execute_tool get_weather           from @observe: arguments and result
  execute_tool find_events
  chat claude-haiku-4-5-...
```

## Notes

- `boto3.client("bedrock-runtime")` needs a region: the script passes `region_name`
  from `AWS_REGION` (boto3 itself reads `AWS_DEFAULT_REGION` or the profile's region).
- The Bedrock instrumentor records the chat model without the inference-profile prefix
  (`claude-haiku-4-5-20251001-v1:0`); the agent span has the full id from `enrich()`.
