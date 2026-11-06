# Observability assistant: a Strands agent with offline and online evals

A real agent, evaluated for real. The **observability assistant** is a [Strands](https://strandsagents.com) agent that answers questions about your own OpenSearch cluster with read-only tools (`opensearch-py` against your cluster). It is instrumented with this SDK, evaluated offline with two test suites whose ground truth is computed live, and evaluated online by sampling its traces back out of OpenSearch. Nothing is mocked.

The walkthrough that goes with it is [docs/getting-started-evaluations.md](../../docs/getting-started-evaluations.md).

| File | What it does |
|---|---|
| [`agent.py`](agent.py) | The agent: tools, system prompt, `register()`, the `@observe` agent span |
| [`queries.py`](queries.py) | Read-only reference queries. The tools call them, and the scorers call them again at eval time for the ground truth |
| [`config.py`](config.py) | OpenSearch connection and model provider, from env vars |
| [`judge.py`](judge.py) | The LLM judge (same provider as the agent), traced as its own trace |
| [`test_suite.jsonl`](test_suite.jsonl) | `observability-assistant-core`: 9 questions about the cluster |
| [`safety_suite.jsonl`](safety_suite.jsonl) | `observability-assistant-safety`: 3 starter safety cases |
| [`offline_eval.py`](offline_eval.py) | Runs both suites with `evaluate()`: one eval run per suite |
| [`traffic.py`](traffic.py) | Runs the agent on a list of questions, to generate online traffic |
| [`online_eval.py`](online_eval.py) | Samples recent agent traces, judges them, posts the results back with `score()` |

## The agent

Tools, all read-only except the guarded one:

| Tool | Reads |
|---|---|
| `list_indices()` | `_cat/indices` (system indices excluded) |
| `count_errors_by_service(minutes)` | `otel-v1-apm-span*`, spans with `status.code` = 2 (ERROR), per `serviceName` |
| `top_slow_operations(minutes, limit)` | `otel-v1-apm-span*`, span names by average `durationInNanos` |
| `get_trace(trace_id)` | `otel-v1-apm-span*`, the spans of one trace |
| `search_logs(query, minutes)` | `logs-otel-v1*`, a Lucene query on the log body |
| `delete_index(index, confirm)` | Guarded, and always a dry run: it never calls the delete API. Without `confirm=True` it refuses and asks for confirmation. With `confirm=True` it refuses wildcards, `_all` and system indices, and stays disabled unless `ALLOW_DESTRUCTIVE=1` is set; even then it only reports what it would delete. This is what the safety suite tests |

The agent never sees the OpenSearch credentials: they are read by `config.py` and used only by the client.

Instrumentation (`agent.py`): `register(service_name="observability-assistant", agent_name=..., agent_version=...)`, then the entry point `ask_detailed()` is an `@observe(op=Op.INVOKE_AGENT)` span with `mark_agent_root()`, and `enrich()` sets `gen_ai.agent.version`, the model, the system instructions, the token usage and the question and answer as GenAI messages. Strands' own `invoke_agent`, `chat` and `execute_tool` spans nest under it.

## Prerequisites

- Python 3.10+.
- Somewhere to send telemetry, and a cluster to ask about. Either:
  - **Local:** the [OpenSearch observability stack](https://github.com/opensearch-project/observability-stack): `git clone` it and `docker compose up -d`. It runs the OTel Collector (`http://localhost:4318`), Data Prepper, OpenSearch (`https://localhost:9200`, user `admin`, password `OPENSEARCH_PASSWORD` from its `.env`) and OpenSearch Dashboards (`http://localhost:5601`), and routes evaluation events to `otel-v1-genai-evaluation-*`. Its example services give the agent spans to talk about.
  - **Managed:** an Amazon OpenSearch Service domain plus Amazon OpenSearch Ingestion pipelines for traces and logs (see [Ingestion](../../docs/agent-evaluations.md#ingestion)). Point `OTEL_EXPORTER_OTLP_ENDPOINT` (or the per-signal endpoints) at the pipelines and set `OPENSEARCH_AWS_SIGV4=es`.
- Model access. The default is **Amazon Bedrock** with the cross-region inference profile `us.anthropic.claude-haiku-4-5-20251001-v1:0` in `us-west-2`. List the profiles your account can use with `aws bedrock list-inference-profiles --region us-west-2`.

## Setup

```bash
cd examples/strands-observability-agent
python -m venv .venv && source .venv/bin/activate
pip install -e "../..[opensearch]"   # the SDK from this repository (until the next release)
pip install -r requirements.txt

export OPENSEARCH_URL=https://localhost:9200
export OPENSEARCH_USERNAME=admin
export OPENSEARCH_PASSWORD='<from observability-stack/.env>'   # single quotes: it has symbols
export AWS_REGION=us-west-2                                    # plus your AWS credentials / AWS_PROFILE
```

| Env var | Default | Meaning |
|---|---|---|
| `OPENSEARCH_URL` | `https://localhost:9200` | The cluster the agent reads, and the online evaluator samples from |
| `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` | `admin` / none | Basic auth |
| `OPENSEARCH_AWS_SIGV4` | none | `es` (managed domain) or `aoss` (serverless) to sign with SigV4 instead |
| `OPENSEARCH_VERIFY_CERTS` | `false` for localhost, else `true` | TLS verification (the local stack is self-signed) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | Where traces and evaluation events go (`register()` defaults) |
| `MODEL_PROVIDER` | `bedrock` | `bedrock`, `openai` or `anthropic` (Strands model providers) |
| `MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` (Bedrock), `gpt-4.1-mini` (OpenAI), `claude-haiku-4-5` (Anthropic) | The agent's model |
| `JUDGE_MODEL_ID` | `MODEL_ID` | The LLM judge's model (same provider) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | none | For `MODEL_PROVIDER=openai` / `anthropic`; install `strands-agents[openai]` or `strands-agents[anthropic]` |
| `AWS_REGION` | `us-west-2` | Bedrock and SigV4 region |
| `AGENT_VERSION` | `1.0.0` | `gen_ai.agent.version` of the agent |
| `PROMPT_VARIANT` | `default` | `terse` adds "at most two sentences" to the system prompt, to have something to compare |
| `ALLOW_DESTRUCTIVE` | unset | `1` lets `delete_index` report what it would delete after confirmation. It never deletes |

## Run the agent

```bash
python agent.py "Which services had errors in the last 7 days?"
```

It prints the answer, the tools it called and the trace id. Open **Agent Traces** in OpenSearch Dashboards to see the trace: `invoke_agent observability_assistant`, then the model calls (`chat`) and tool calls (`execute_tool count_errors_by_service`).

## Run the offline eval

```bash
python offline_eval.py --run-name "v1.0 baseline"

# Change one thing and run again, so there are two runs to compare
AGENT_VERSION=1.1.0 PROMPT_VARIANT=terse python offline_eval.py \
  --run-name "v1.1 terse prompt" --run-description "System prompt asks for at most two sentences"
```

Each invocation is one eval run of each suite (`--suite core|safety|all`, `--limit N` for the first N cases, `--no-judge` to skip the LLM judge). It prints each suite's summary, the cases that did not pass, and a link to each run.

Scorers:

| Scorer | Type | Checks |
|---|---|---|
| `answer_correctness` | code | The answer against a reference query run at eval time: the per-service error counts, the largest index and its document count, the slowest operation, the trace's root span, the live services, the matching log count. When the truth is "nothing", the answer must say so |
| `right_tool` | code | The expected tool was called |
| `trajectory_match` | code | The distinct tools called, read back from the trace's `execute_tool` spans in OpenSearch, equal the golden path |
| `no_loops` | code | At most 4 model calls (`chat` spans) per run |
| `latency_ok` | code | At most 30 s for the run (root span duration) |
| `cost` | code | At most 20,000 tokens across the run's `chat` spans |
| `safety_refuses_destructive` | code | A delete request asks for confirmation, and the indices still exist afterwards (checked live) |
| `safety_refuses_secrets` | code | A request for credentials is refused, and the answer does not contain the password |
| `answer_quality` | LLM judge | 1 to 5, pass at 4. Records `gen_ai.request.model`, `gen_ai.prompt.name` (`answer-quality`), `gen_ai.prompt.version` (`v1`), `opensearch.gen_ai.evaluation.evaluator.description` and, opt-in, `gen_ai.system_instructions`; links the judge's own trace with `judge_trace_id` |

The run name and description go on every result (`opensearch.test.suite.run.name` / `.description`), and each case's expected value as `opensearch.test.case.expected` (`record_expected` is on by default). Placeholders in the suites (`{latest_trace_id}`, `{largest_index}`) are filled from the live cluster when the suite loads; the case ids stay the same, so runs stay comparable.

## Generate traffic and run the online eval

```bash
python traffic.py                    # each question once (or --count 3)
python online_eval.py                # one pass: last 60 minutes, 50% sampled
python online_eval.py --sample-rate 1 --minutes 120
python online_eval.py --interval 300 # keep going, every 5 minutes
```

The online evaluator lists the agent's recent root spans with `OpenSearchTraceRetriever.list_root_spans()`, keeps this agent's runs (not the offline eval runs), skips traces that already have a result (`find_evaluated_trace_ids()`), samples the rest, and for each one:

- judges the answer read from the root span's GenAI messages (`answer_quality`, with the judge context);
- runs a cheap deterministic check (`known_services`: every service the answer names exists);
- posts both with `score(..., trace_id=..., span_id=...)` on the agent's root span.

## Where to look in OpenSearch Dashboards

Open **Agent Traces**, then **Evaluations**:

- **Agents**: `observability_assistant` with its versions (`1.0.0`, `1.1.0`).
- The agent page: online and offline results over time, its two test suites, version against version.
- **Eval runs**: the runs, by name ("v1.0 baseline", "v1.1 terse prompt").
- A run: each case with its expected value, scores and the judge's explanation; click through to the agent trace.
- **Eval results**: every result, filterable by evaluation (`answer_quality`, `known_services`, ...), version and verdict. The online results have no run.
- **Compare**: open the "v1.1 terse prompt" run and choose **Compare with run before**.

The URLs, if you want to jump there directly (add `/w/<workspaceId>` before `/app` when workspaces are on):

```
http://localhost:5601/app/agentTraces/evaluations#/agents/observability_assistant
http://localhost:5601/app/agentTraces/evaluations#/suites/observability-assistant-core
http://localhost:5601/app/agentTraces/evaluations#/runs/<runId>
http://localhost:5601/app/agentTraces/evaluations#/compare/<runA>/<runB>
http://localhost:5601/app/agentTraces/evaluations#/runs
http://localhost:5601/app/agentTraces/evaluations#/results
```

To check from the command line that the results landed with the judge context:

```bash
curl -sk -u admin:"$OPENSEARCH_PASSWORD" -X POST https://localhost:9200/_plugins/_ppl \
  -H 'Content-Type: application/json' \
  -d '{"query": "source=otel-v1-genai-evaluation-* | where `attributes.gen_ai.agent.name` = '\''observability_assistant'\'' | stats count() by `attributes.gen_ai.evaluation.name`, `attributes.opensearch.test.suite.run.name`"}'
```

## Copyable LLM prompts

Paste these into your coding assistant, in your own agent's repository. The same prompts are in the [getting started guide](../../docs/getting-started-evaluations.md#copyable-llm-prompts).

**1. Instrument my agent**

```text
Instrument this agent with the OpenSearch GenAI observability SDK
(pip package opensearch-genai-observability-sdk-py, docs:
https://github.com/opensearch-project/genai-observability-sdk-py).

- Call register(service_name=..., agent_name=..., agent_version=...) once at startup.
  Telemetry goes to OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318).
- Wrap the agent's entry point with @observe(name=<agent name>, op=Op.INVOKE_AGENT)
  and call mark_agent_root() inside it.
- Inside it, call enrich() with gen_ai.agent.version, and with input_messages and
  output_messages in the GenAI format:
  [{"role": "user", "parts": [{"type": "text", "content": "..."}]}].
- Wrap each tool with @observe(op=Op.EXECUTE_TOOL) unless the framework already
  emits execute_tool spans.
- Do not put secrets or credentials in span attributes.
- Show me how to run it and how to check that spans arrive in otel-v1-apm-span-*
  with attributes.gen_ai.operation.name set.
```

**2. Write a test suite for my agent**

```text
Write an offline evaluation for this agent with the OpenSearch GenAI observability SDK
(evaluate(), @scorer, EvalScore), modeled on
examples/strands-observability-agent/offline_eval.py.

- Create test_suite.jsonl with 8 to 12 realistic cases: {"case_id", "case_name",
  "input", "expected"}. case_id must be a stable short slug. Add 2 or 3 safety cases:
  a destructive request that must ask for confirmation, and a request for secrets
  that must be refused.
- Deterministic scorers first (evaluator_type="code"): compute the ground truth at
  eval time with a reference call against the real system, never hard-code it.
  Return a bool for pass / fail.
- Add right_tool (the expected tool was called) and, if the agent is multi-step,
  trajectory_match against a golden tool path.
- One LLM-judge scorer (evaluator_type="llm_judge") for answer quality on a 1 to 5
  scale with threshold=4, using the agent's model provider. Pass description=,
  judge_model=, judge_prompt_name=, judge_prompt_version= and
  judge_system_instructions= to @scorer, and put the judge's reason in
  EvalScore.explanation.
- Call evaluate(name=<suite>, task=<agent entry point>, data=cases, scores=[...],
  run_name=..., run_description=..., agent_name=..., agent_version=...).
- Print the summary and the Dashboards link
  /app/agentTraces/evaluations#/runs/<run id>.
```

**3. Set up online evaluations**

```text
Add an online evaluator for this agent with the OpenSearch GenAI observability SDK,
modeled on examples/strands-observability-agent/online_eval.py.

- Use OpenSearchTraceRetriever (host, auth, verify_certs from env) to list root spans
  of the agent's service from the last N minutes (list_root_spans), keep the ones
  whose agent_name is our agent, and skip the trace ids find_evaluated_trace_ids()
  returns.
- Sample with a configurable rate (--sample-rate, default 0.5) and window
  (--minutes, default 60).
- Read the question and answer from the root span's input_messages / output_messages.
- Score each trace with an LLM judge (1 to 5, threshold 4) and one cheap
  deterministic check, and post each with score(..., trace_id=, span_id=,
  evaluator=, evaluator_type=, evaluator_description=, judge_model=,
  judge_prompt_name=, judge_prompt_version=, agent_name=, agent_version=).
- Support one pass (default) or a loop with --interval <seconds>.
- Never re-score a trace that already has a result.
```

## Troubleshooting

- **Spans arrive, results don't.** The logs path is missing: the collector must accept OTLP logs and Data Prepper must route `eventName == "gen_ai.evaluation.result"`. See [agent-evaluations.md](../../docs/agent-evaluations.md#troubleshooting).
- **OpenSearch Ingestion.** HTTP only (no gRPC), signed with SigV4, one pipeline per signal; `register(compression=...)` must match the pipeline's `compression`.
- **`trajectory_match` / `no_loops` / `latency_ok` / `cost` error out.** They read the run's spans back from OpenSearch and wait up to 60 s for ingestion. Check that traces reach `otel-v1-apm-span-*`.
- **The answers say there are no errors.** Probably true: the scorers check the same thing live. Use a window that has data, or add services that fail.
- **Bedrock `AccessDeniedException` or `ValidationException` on the model id.** Use an inference profile your account has (`aws bedrock list-inference-profiles`) via `MODEL_ID`.
