# DeepEval metrics on the travel assistant

Runs [DeepEval](https://deepeval.com) metrics on the travel assistant's answers and sends
every metric result to OpenSearch as a `gen_ai.evaluation.result`.

- `agent.py`: the travel assistant (Strands on Amazon Bedrock, tools `get_weather` and
  `find_events` over `../../shared/travel_data.json`). `run_agent()` returns the answer,
  the tool results and the agent run's trace and span ids.
- `deepeval_eval.py`: `evaluate()` runs the shared test suite
  (`../../shared/test_suite.jsonl`); each `@scorer` runs one DeepEval metric with a Bedrock
  judge (`AmazonBedrockModel`) and returns `EvalScore(value=metric.score,
  explanation=metric.reason)`.

| Evaluation | Evaluator | Checks |
|---|---|---|
| `answer_relevancy` | `deepeval.AnswerRelevancyMetric` | The answer addresses the request |
| `faithfulness` | `deepeval.FaithfulnessMetric` | The answer is grounded in the tool results (`retrieval_context`) |
| `correctness` | `deepeval.GEval` | The answer meets the expected output, using only the tool results (`context`) |

All three are `llm_judge` results with `threshold` 0.7, scale 0 to 1, the metric's reason
as explanation, and the judge model in `gen_ai.request.model` (set with
`@scorer(judge_model=...)`).

Why `evaluate()` and not `Benchmark`: DeepEval metrics score one test case at a time, so
each can be a scorer. `evaluate()` then links every result to the agent run, stamps
`test.*` on the agent's root span, and turns a metric that raises
(for example `MissingTestCaseParamsError`) into an evaluator error result. For libraries
that score a whole dataset at once, see `../ragas` and `../mlflow` (`Benchmark`).

## Run

```bash
pip install -r requirements.txt
export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-west-2
export DEEPEVAL_TELEMETRY_OPT_OUT=YES   # optional: turn off DeepEval's own telemetry
# optional: MODEL_ID=<agent model id>, JUDGE_MODEL_ID=<judge model id>
#   (both default to us.anthropic.claude-haiku-4-5-20251001-v1:0)
python deepeval_eval.py
```

To use another judge, pass any DeepEval model as `judge` (for example `GPTModel` with
`OPENAI_API_KEY` set) and update `JUDGE_MODEL_ID` to match.

## What you get

- Spans (service `travel-assistant-evals-deepeval`): one trace per case, root
  `invoke_agent travel_assistant` with `chat` and `execute_tool` spans.
- Results: 15 (5 cases x 3 metrics) in run `deepeval judges` of suite
  `travel-assistant-suite-deepeval`, each parented to its agent root span.

```
GET otel-v1-genai-evaluation/_search
{
  "size": 0,
  "query": {"term": {"attributes.test.suite.name": "travel-assistant-suite-deepeval"}},
  "aggs": {"evaluator": {"terms": {"field": "attributes.opensearch.gen_ai.evaluation.evaluator.name"}}}
}
```
