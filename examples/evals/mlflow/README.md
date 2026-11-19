# MLflow judges on the travel assistant

Runs [MLflow 3](https://mlflow.org/docs/latest/genai/) built-in LLM judges on the travel
assistant, then uploads the verdicts to OpenSearch with `Benchmark`, one `Benchmark.log()`
per case, linked to the case's agent trace.

- `agent.py`: the travel assistant (Strands on Amazon Bedrock, tools `get_weather` and
  `find_events` over `../../shared/travel_data.json`). `run_agent()` returns the answer
  and the agent run's trace and span ids.
- `mlflow_eval.py`: `mlflow.genai.evaluate()` calls the agent (`predict_fn`) on each case
  of `../../shared/test_suite.jsonl` and grades it; the results are uploaded with the
  judge's rationale as explanation.

| Evaluation | Evaluator | Verdict |
|---|---|---|
| `correctness` | `mlflow.Correctness` | The judge's `yes` / `no` against the expected response, as `pass` / `fail` |
| `relevance_to_query` | `mlflow.RelevanceToQuery` | The judge's `yes` / `no`, as `pass` / `fail` |

Both are `llm_judge` results; the judge is `bedrock:/<JUDGE_MODEL_ID>`.

Notes:

- MLflow's `bedrock:/` judge reads AWS keys from environment variables only (not from a
  profile). The script exports the keys of the current boto3 session when
  `AWS_ACCESS_KEY_ID` is not set.
- MLflow records a failed judge call in the results table instead of raising. The script
  stops before uploading if any `<scorer>/error_message` is set, so a failed judge is
  never uploaded as a `fail`.
- The judge model is recorded on every result as `gen_ai.request.model`, set in the run
  metadata: `Benchmark(..., metadata={"gen_ai.request.model": JUDGE_MODEL_ID})`.
  `Benchmark.log(metadata=...)` and `EvalScore(metadata=...)` take the same keys (also
  `gen_ai.prompt.name` / `gen_ai.prompt.version`) per case or per score.
- MLflow calls `predict_fn` once more than there are cases (a validation call first), so
  one agent trace per run has no results.
- MLflow writes `mlruns/` and `mlflow.db` in the working directory. Set
  `MLFLOW_DISABLE_AGENT_HINT=1` to hide its "load the skill" hint.

## Run

```bash
pip install -r requirements.txt
export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-west-2
# optional: MODEL_ID=<agent model id>, JUDGE_MODEL_ID=<judge model id>
#   (both default to us.anthropic.claude-haiku-4-5-20251001-v1:0)
python mlflow_eval.py
```

With OpenAI as the judge, set `OPENAI_API_KEY` and `JUDGE = "openai:/gpt-4o-mini"`.

## What you get

- Spans (service `travel-assistant-evals-mlflow-py`): one trace per `predict_fn` call, root
  `invoke_agent travel_assistant` with `chat` and `execute_tool` spans.
- Results: 10 (5 cases x 2 judges) in run `mlflow judges` of suite
  `travel-assistant-suite-mlflow`, each parented to its agent root span and carrying the
  judge model in `gen_ai.request.model`.

```
GET otel-v1-genai-evaluation/_search
{
  "size": 0,
  "query": {"term": {"attributes.test.suite.name": "travel-assistant-suite-mlflow"}},
  "aggs": {"evaluator": {"terms": {"field": "attributes.opensearch.gen_ai.evaluation.evaluator.name"}}}
}
```
