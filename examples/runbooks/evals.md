# Runbook: evaluate an agent

Offline eval runs, online scoring of real traces, and results from other eval libraries,
all as `gen_ai.evaluation.result` events in OpenSearch. Each step names the example that
shows it. The full guide, with the hierarchy (agent, test suite, test case, eval run),
scorer recipes and deep links, is
[docs/getting-started-evaluations.md](../../docs/getting-started-evaluations.md) (steps 1, 6
to 9); the attribute reference is [docs/agent-evaluations.md](../../docs/agent-evaluations.md).

Before you start: the agent is traced ([runbooks/tracing.md](tracing.md)) and the stack
routes evaluation events to the `otel-v1-genai-evaluation` alias (the observability stack
does). Commands assume `OPENSEARCH_PASSWORD` is exported.

## 1. Write the test suite

One case per line, with a stable `case_id` (Compare matches cases by it) and an expected
output. The travel assistant's suite is [`shared/test_suite.jsonl`](../shared/); the
observability assistant's are
[`end-to-end/strands-observability-agent/test_suite.jsonl`](../end-to-end/strands-observability-agent/test_suite.jsonl)
and `safety_suite.jsonl`.

## 2. Write scorers

A scorer is `fn(input, output, expected)` returning a `bool` (a verdict), a number or an
`EvalScore`. Declare the rule with `@scorer(threshold=..., direction=...)` so Dashboards
shows pass / fail; without one a result is only "Scored".

- Deterministic checks first (`evaluator_type="code"`): tool called, data used, latency,
  tokens. Compute the truth when you score, not by hand:
  [end-to-end/.../offline_eval.py](../end-to-end/strands-observability-agent/offline_eval.py).
- One LLM judge (`evaluator_type="llm_judge"`) with `judge_model=`, `judge_prompt_name=`,
  `judge_prompt_version=` and `description=`. The judge prompt is not stored on the result:
  record it on the judge's own LLM span (`enrich(system_instructions=...)`) and link it with
  `EvalScore(judge_trace_id=...)`:
  [end-to-end/.../judge.py](../end-to-end/strands-observability-agent/judge.py),
  [evals/native](../evals/native/).
- A declaration fills only the scorer's own results: those that name no `evaluator`, or
  the scorer's own (the declared `evaluator=`, else the function name). A scorer that
  returns a library's result, such as `EvalScore(name="faithfulness",
  evaluator="deepeval.faithfulness")`, must declare that evaluator to get the rule:
  `@scorer(name="faithfulness", evaluator="deepeval.faithfulness", threshold=0.7)`
  ([evals/deepeval](../evals/deepeval/)). When a declaration reaches none of a scorer's
  results in a run, `evaluate()` logs a warning at the end of the run.

## 3. Run an offline eval: `evaluate()`

```python
from opensearch_genai_observability_sdk_py import evaluate, register

register(
    service_name="travel-assistant-evals-native",
    agent_name="travel_assistant",
    agent_version="1.0.0",
)
result = evaluate(
    name="travel-assistant",  # test suite (test.suite.name), stable across runs
    task=ask,  # the traced agent entry point
    data=cases,
    scores=[uses_tool_data, plan_quality],
    run_name="v1 prompt",  # shown in place of the run id
    run_description="Claude Haiku 4.5, default prompt",
    agent_name="travel_assistant",
    agent_version="1.0.0",
)
print(result.summary.run_id)
```

Each case's agent run is its own trace with `test.suite.*` and `test.case.*` on the root
span; each score is one result parented to it. Change one thing (prompt, model, tool) and
run again with a new `run_name` to have two runs to compare.

`result.cases[i].scores` / `.verdicts` and the summary are keyed by evaluation name. When
two evaluators report the same name in a run, the first keeps the plain name and the other
is keyed `"<name> (<evaluator>)"`, so no result hides another and a failing result always
fails the case. Run `metadata` is copied to every result of the run; besides your own keys
it may carry the judge context `gen_ai.request.model` and `gen_ai.prompt.*`.

Shown in: [evals/native](../evals/native/) (two runs over the shared suite),
[end-to-end/strands-observability-agent](../end-to-end/strands-observability-agent/)
(`offline_eval.py --run-name ...`), [basics/07_benchmarks.py](../basics/07_benchmarks.py).
In CI: [basics/github-action](../basics/github-action/) and [evals/pytest](../evals/pytest/).

## 4. Score production traffic online: `score()` with retrieval

```python
from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever, score

retriever = OpenSearchTraceRetriever(
    host="https://localhost:9200", auth=("admin", password), verify_certs=False
)
roots = retriever.list_root_spans(
    services=["travel-assistant-evals-native"], since=since, max_results=50
)
done = retriever.find_evaluated_trace_ids(
    [r.trace_id for r in roots], evaluator="travel_plan_judge"
)
offline = {r.trace_id for r in roots if "test.suite.run.id" in r.raw.get("attributes", {})}
for root in (r for r in roots if r.trace_id not in done | offline):
    verdict = judge(root.input_messages[0].content, root.output_messages[-1].content)
    score(
        "plan_quality",
        verdict["score"],
        trace_id=root.trace_id,
        span_id=root.span_id,
        threshold=4,
        score_min=1,
        score_max=5,
        evaluator="travel_plan_judge",
        evaluator_type="llm_judge",
        agent_name=root.agent_name,
        agent_version=root.agent_version or None,
    )
```

- `list_root_spans()` returns each trace's agent root, the row Agent Traces lists, with
  its GenAI messages, `agent_name` and `agent_version`. Pass both to `score()`. A root is
  a span marked `derived.gen_ai.is_root` or a parentless `invoke_agent` / `create_agent`
  span, so an LLM judge's own `chat` trace is not listed.
- `find_evaluated_trace_ids(..., evaluator=)` makes the job safe to re-run.
- Skip the offline runs (root spans with `test.suite.run.id`) and sample the rest.

Shown in: [evals/native](../evals/native/) (`online_eval.py`),
[end-to-end/strands-observability-agent](../end-to-end/strands-observability-agent/)
(`traffic.py`, then `online_eval.py`), [basics/06_retrieval_and_eval.py](../basics/06_retrieval_and_eval.py),
[basics/02_scoring.py](../basics/02_scoring.py) (span, trace and session level results).

## 5. Bring results from an eval library

Keep your library and send its numbers: return them from an `evaluate()` scorer (results
get the suite, case and run ids), log them with `Benchmark(...).log(...)` when the run
already happened, or `score()` them onto a trace. In `evaluate()`, declare the library's
evaluator in `@scorer(evaluator=...)` so its results get the rule and `@scorer(judge_model=...)`.
With `Benchmark`, record the judge in the metadata (`Benchmark(metadata={"gen_ai.request.model": ...})`,
also `gen_ai.prompt.name` / `gen_ai.prompt.version`): it describes every result of the run.
`log(evaluator=..., evaluator_type=...)` sets the default evaluator for the call's scores
that name no evaluator (or the same one); if that `evaluator_type` reaches no score of the
run, `close()` logs a warning. `close()` prints and returns the local summary.

| Library | Example |
|---|---|
| DeepEval | [evals/deepeval](../evals/deepeval/) |
| Ragas | [evals/ragas](../evals/ragas/) |
| MLflow | [evals/mlflow](../evals/mlflow/) |
| pytest | [evals/pytest](../evals/pytest/) |
| Any (list of dicts, with trace links) | [basics/08_upload_benchmark_results.py](../basics/08_upload_benchmark_results.py) |

## 6. Verify the results

Results by evaluation and run name for your agent:

```bash
curl -sk -u admin:"$OPENSEARCH_PASSWORD" -X POST https://localhost:9200/_plugins/_ppl \
  -H 'Content-Type: application/json' \
  -d '{"query": "source=otel-v1-genai-evaluation | where `attributes.gen_ai.agent.name` = '\''travel_assistant'\'' | stats count() by `attributes.gen_ai.evaluation.name`, `attributes.opensearch.test.suite.run.name`"}'
```

Look for one row per evaluation and run; online results have no run name (null). If
nothing arrived but spans did, the logs path is missing: check the collector's
`otelcol_receiver_accepted_log_records` counter on `http://localhost:8888/metrics`, and
that results did not land in `logs-otel-v1-*` instead (the Data Prepper route on
`eventName`).

Then open **Evaluations** at `http://localhost:5601/app/agentTraces/evaluations`:

- **Agents**: `travel_assistant` with its versions; its page shows online and offline
  results over time and its test suites.
- **Eval runs**: your runs by name. Open one for each case's expected value, scores and
  judge explanation, and click through to the agent trace. **Compare with run before**
  shows two runs case by case.
- **Eval results**: every result, filterable by evaluation, version and verdict. The
  **Online** source filter shows the `score()` results.

Direct links: `#/runs/<runId>` (`result.summary.run_id`), `#/compare/<runA>/<runB>`,
`#/agents/<agent>`; see [Deep links](../../docs/getting-started-evaluations.md#deep-links).

## 7. Close the loop

When an online result fails, open its trace, add the question as a new case with a stable
`case_id`, fix the agent, and run the suite again with a new `run_name`. Compare: the new
case should pass and nothing else should regress.
