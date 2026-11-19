# pytest: test cases as tests, verdicts as evaluation results

The shared test suite (`../../shared/test_suite.jsonl`) as pytest tests. Each test runs
the travel assistant once, sends its checks to OpenSearch as `gen_ai.evaluation.result`
events (with `Benchmark.log()`, linked to the agent run's trace), then asserts them. A
failing check is both a failed test and a `fail` result in OpenSearch Dashboards.

- `agent.py`: the travel assistant (Strands on Amazon Bedrock, tools `get_weather` and
  `find_events` over `../../shared/travel_data.json`).
- `conftest.py`: a session fixture that calls `register()` and opens one `Benchmark` run
  for the whole session (`close()` and `force_flush()` at the end).
- `test_travel_assistant.py`: one parametrized test per case.

| Evaluation | Evaluator (type) | Pass rule |
|---|---|---|
| `uses_tool_data` | `uses_tool_data` (`code`) | Names the city's weather and one of its events, or says there is no data |
| `answer_length` | `answer_length` (`code`) | At most 800 characters (`direction="minimize"`) |

If the agent raises, the test records an error result per check (`error_type` = the
exception class) and fails.

## Run

```bash
pip install -r requirements.txt
export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-west-2
# optional: MODEL_ID=<agent model id> (default us.anthropic.claude-haiku-4-5-20251001-v1:0)
pytest -v -s                          # -s shows the run summary at the end
EVAL_RUN_NAME="PR 42" pytest -v       # name the run, for example after the CI job
```

In CI (GitHub Actions, GitLab CI, Jenkins) the SDK adds `cicd.pipeline.*` to every
result, so the run links back to the job.

## What you get

- Spans (service `travel-assistant-evals-pytest-py`): one trace per test, root
  `invoke_agent travel_assistant` with `chat` and `execute_tool` spans.
- Results: 10 per session (5 tests x 2 checks) in run `pytest` (or `EVAL_RUN_NAME`) of
  suite `travel-assistant-suite-pytest`, each parented to its agent root span.

`Benchmark` stamps `test.*` only on the results. To also stamp the agent spans and gate on
averages instead of per-case asserts, call `evaluate()` inside one test and assert on
`result.summary` (see `../native`).

```
GET otel-v1-genai-evaluation/_search
{
  "size": 0,
  "query": {"term": {"attributes.test.suite.name": "travel-assistant-suite-pytest"}},
  "aggs": {"run": {"terms": {"field": "attributes.opensearch.test.suite.run.name"},
    "aggs": {"verdict": {"terms": {"field": "attributes.gen_ai.evaluation.score.label"}}}}}
}
```
