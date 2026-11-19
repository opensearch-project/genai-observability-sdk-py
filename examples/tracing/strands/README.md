# Tracing: Strands Agents

The travel assistant (see [`../../shared/`](../../shared/)) built with
[Strands Agents](https://strandsagents.com/) on Amazon Bedrock.

Strands has built-in OpenTelemetry tracing and uses the global tracer provider that
`register()` installs. No instrumentor package and no Strands exporter setup is needed.

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

## What you get

Service `travel-assistant-strands-py`, one trace per question:

```
invoke_agent travel_assistant        gen_ai.agent.name=travel_assistant, tokens, messages
  execute_event_loop_cycle
    chat                             model, tokens, input/output messages
    execute_tool get_weather         arguments {"city": "Paris"} and result
    execute_tool find_events
  execute_event_loop_cycle
    chat
```

## Notes

- `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental,gen_ai_span_attributes_only`
  must be set before Strands is imported (the script sets it with `os.environ.setdefault`).
  Without `gen_ai_span_attributes_only` Strands writes the messages as span events, which
  Agent Traces does not read.
- `Agent(name="travel_assistant")` sets `gen_ai.agent.name` on the `invoke_agent` span.
- `trace_attributes={"gen_ai.conversation.id": ...}` groups the turns of one run into a session.
- Do not call `StrandsTelemetry().setup_otlp_exporter()`: `register()` already exports.
- `register(auto_instrument=False)`: Strands emits its own `chat` spans. With auto
  instrumentation on, an installed Bedrock instrumentor (for example from the SDK's
  `[bedrock]` extra in the same environment) also traces each Converse call, so every model
  call shows up as two `chat` spans.
- To use another model provider, replace `BedrockModel` with any Strands model class; the
  tracing stays the same.
