# Native evaluations: evaluate() and score()

The travel assistant (a [Strands](https://strandsagents.com) agent on Amazon Bedrock with
`get_weather` and `find_events` tools over `../../shared/travel_data.json`) evaluated with
the SDK alone:

- **Offline** (`offline_eval.py`): `evaluate()` runs the shared test suite
  (`../../shared/test_suite.jsonl`) twice, once per system prompt version, with run names
  `v1 prompt` and `v2 prompt`, so the two runs can be compared.
- **Online** (`online_eval.py`): reads recent agent traces from OpenSearch with
  `OpenSearchTraceRetriever`, judges them, and posts the results back with
  `score(trace_id=..., span_id=...)`.

| File | What it is |
|---|---|
| `agent.py` | The agent, its two tools and `load_test_suite()` |
| `scorers.py` | `uses_tool_data` (code check) and `judge_plan` (LLM judge on Bedrock), used offline and online |
| `offline_eval.py` | Two `evaluate()` runs over the test suite |
| `traffic.py` | Two live requests, as production traffic |
| `online_eval.py` | Judge the last hour of traces, post results with `score()` |

## Evaluators

| Evaluation | Evaluator (type) | Pass rule |
|---|---|---|
| `uses_tool_data` | `uses_tool_data` (`code`) | Names the city's weather and one of its events, or says there is no data |
| `answer_length` | `answer_length` (`code`) | At most 500 characters (`direction="minimize"`), offline only |
| `plan_quality` | `travel_plan_judge` (`llm_judge`) | Judge score 1 to 5, pass at 4 |

The judge records its context on every result: `judge_model` (`gen_ai.request.model`),
`judge_prompt_name` / `judge_prompt_version` (`gen_ai.prompt.name` / `.version`, shown as
`travel-plan-quality@v1`) and a description. Bump `JUDGE_PROMPT_VERSION` in `scorers.py`
when you change the judge prompt.

## Run

Start the observability stack (OTLP on `localhost:4318`, OpenSearch on `localhost:9200`), then:

```bash
pip install -r requirements.txt
export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-west-2
# optional: MODEL_ID=<agent model id>, JUDGE_MODEL_ID=<judge model id>
#   (both default to us.anthropic.claude-haiku-4-5-20251001-v1:0)

python offline_eval.py      # two runs: "v1 prompt" and "v2 prompt"

python traffic.py           # two live requests
export OPENSEARCH_PASSWORD=<admin password>  # OPENSEARCH_URL, OPENSEARCH_USER optional
python online_eval.py       # wait about a minute after traffic.py so the spans are indexed
```

`online_eval.py` skips the offline cases (roots with `test.suite.run.id`, scored by their
run) and traces that already have a `travel_plan_judge` result
(`find_evaluated_trace_ids(..., evaluator="travel_plan_judge")`), so nothing is judged
twice, even while the offline results are still being indexed. It sets `verify_certs=False` by default
for the local stack's self-signed certificate; set `OPENSEARCH_VERIFY_CERTS=true` elsewhere.

## What you get

- Spans (service `travel-assistant-evals-native`): one trace per case or request, with an
  `invoke_agent travel_assistant` root (`gen_ai.agent.version` v1 or v2), `chat` and
  `execute_tool` spans. `evaluate()` stamps `test.suite.name`, `test.suite.run.id` and
  `test.case.*` on each root.
- Results (`otel-v1-genai-evaluation`): offline, 15 per run (5 cases x 3 evaluations) with
  `opensearch.test.suite.run.name` = `v1 prompt` / `v2 prompt`; online, 2 per judged trace
  and no run id. Every result is parented to the agent's root span.
- In OpenSearch Dashboards: Agent Traces > Evaluations > `travel-assistant-suite` >
  Compare the two runs; online results show on the trace and on the agent's page.

Check the results:

```
GET otel-v1-genai-evaluation/_search
{
  "size": 0,
  "query": {"term": {"attributes.test.suite.name": "travel-assistant-suite"}},
  "aggs": {"run": {"terms": {"field": "attributes.opensearch.test.suite.run.name"},
    "aggs": {"evaluation": {"terms": {"field": "attributes.gen_ai.evaluation.name"}}}}}
}
```
