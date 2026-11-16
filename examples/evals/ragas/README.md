# RAGAS metrics on the travel assistant

Runs [RAGAS](https://docs.ragas.io) metrics on the travel assistant's answers, then uploads
the scores to OpenSearch with `Benchmark`, one `Benchmark.log()` per case, linked to the
case's agent trace.

- `agent.py`: the travel assistant (Strands on Amazon Bedrock, tools `get_weather` and
  `find_events` over `../../shared/travel_data.json`). `run_agent()` returns the answer,
  the tool results and the agent run's trace and span ids.
- `ragas_eval.py`: runs the agent on each case of `../../shared/test_suite.jsonl`, scores
  all answers with `ragas.evaluate()` (the tool results are the `contexts`), and uploads.

| Evaluation | Evaluator | Pass at |
|---|---|---|
| `faithfulness` | `ragas.faithfulness` | 0.7 |
| `answer_relevancy` | `ragas.answer_relevancy` | 0.6 (embedding similarity runs lower) |
| `answer_correctness` | `ragas.answer_correctness` | 0.5 |

All are `llm_judge` results on a 0 to 1 scale. The judge is Claude on Bedrock
(`ChatBedrockConverse`) with Titan embeddings (`answer_relevancy` and
`answer_correctness` use embeddings).

Notes:

- `Benchmark` (and `EvalScore`) have no `judge_model` argument, so these results carry no
  `gen_ai.request.model`; the judge is named in the run description. To record the judge
  model on each result, score per case inside `evaluate()` with
  `@scorer(judge_model=...)` (see `../deepeval`).
- `raise_exceptions=True`: by default RAGAS turns a failed judge call into `NaN`. A metric
  that is still `NaN` is skipped, not sent as a score.
- RAGAS writes no reason text, so the results have no explanation.
- Expect low `faithfulness`: the agent adds landmarks and tips that are not in the tool
  results (for example the Eiffel Tower), and RAGAS counts each as an unsupported claim.
- `Benchmark` stamps `test.*` only on the results, not on the agent spans. The run page
  shows a duration of a few milliseconds because all results are logged at the end.

## Run

```bash
pip install -r requirements.txt
export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-west-2
# optional: MODEL_ID=<agent model id>, JUDGE_MODEL_ID=<judge model id>
#   (both default to us.anthropic.claude-haiku-4-5-20251001-v1:0),
#   EMBEDDINGS_MODEL_ID (default amazon.titan-embed-text-v2:0)
python ragas_eval.py
```

With OpenAI instead of Bedrock, set `OPENAI_API_KEY` and drop the `llm=` and
`embeddings=` arguments of `ragas.evaluate()`.

## What you get

- Spans (service `travel-assistant-evals-ragas`): one trace per case, root
  `invoke_agent travel_assistant` with `chat` and `execute_tool` spans.
- Results: 15 (5 cases x 3 metrics) in run `ragas judges` of suite
  `travel-assistant-suite-ragas`, each parented to its agent root span.

```
GET otel-v1-genai-evaluation/_search
{
  "size": 0,
  "query": {"term": {"attributes.test.suite.name": "travel-assistant-suite-ragas"}},
  "aggs": {"evaluator": {"terms": {"field": "attributes.opensearch.gen_ai.evaluation.evaluator.name"}}}
}
```
