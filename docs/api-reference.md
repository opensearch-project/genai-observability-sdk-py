# API reference

One section per public API of `opensearch-genai-observability-sdk-py`, in this order:

- [`register()`](#register)
- [`observe()`](#observe)
- [`Op`](#op)
- [`enrich()`](#enrich)
- [`mark_agent_root()`](#mark_agent_root)
- [`score()`](#score)
- [`evaluate()`](#evaluate)
- [`Benchmark`](#benchmark)
- [`AWSSigV4OTLPExporter` and `AWSSigV4OTLPLogExporter`](#awssigv4otlpexporter-and-awssigv4otlplogexporter)
- [`OpenSearchTraceRetriever`](#opensearchtraceretriever)

Endpoints, protocols, compression, SigV4 and environment variables are in [configuration.md](configuration.md). Concepts and the full evaluation data model are in [agent-evaluations.md](agent-evaluations.md).


## `register()`

Configures the OTel tracing pipeline, plus an OTLP logs pipeline for the evaluation events emitted by `score()`. Call once at startup.

```python
register(
    endpoint="http://my-collector:4318",  # base URL for traces and logs, or use env vars
    service_name="my-app",
    batch=True,  # BatchSpanProcessor (True) or Simple (False)
    auto_instrument=True,  # discover installed instrumentor packages
    agent_name="travel_planner",  # optional: the agent this process evaluates
    agent_version="v3",  # optional: its version
)
```

### Agent identity

`agent_name` / `agent_version` are stamped as `gen_ai.agent.name` and `gen_ai.agent.version` on every `gen_ai.evaluation.result` event the SDK emits. OpenSearch Dashboards uses them to scope the Evaluations page to one agent and to compare runs version to version. Both are OpenTelemetry GenAI attributes (`gen_ai.agent.name`, `gen_ai.agent.version`).

Per field, the SDK uses the first of:

1. the explicit argument (`score(agent_name=, agent_version=)`, `evaluate(...)`, `Benchmark(...)`);
2. the evaluated span's own attribute: the active span for inline `score()`, the agent root span for `evaluate()`;
3. the `register()` default.

An unknown agent is left out, never invented. `evaluate()` also stamps the agent on the agent root spans where the span names none itself, so traces can be filtered by agent version.

### Parameters

All parameters are keyword-only. How the endpoint parameters resolve against each other and against the environment is in [Endpoints](configuration.md#endpoints-traces-and-evaluation-logs).

| Parameter | Default | Description |
|---|---|---|
| `service_name` | `OTEL_SERVICE_NAME`, else `OPENSEARCH_PROJECT`, else `"default"` | `service.name` resource attribute. `project_name` is an alias |
| `service_version` | `OTEL_SERVICE_VERSION` | `service.version` resource attribute |
| `batch` | `True` | Batch span and log processors (`True`) or simple ones (`False`) |
| `auto_instrument` | `True` | Activate the installed instrumentor packages (see [Auto-instrumentation](configuration.md#auto-instrumentation)) |
| `set_global` | `True` | Set the provider as the global `TracerProvider`. With `False` no logs pipeline is created |
| `agent_name` / `agent_version` | `None` | Default agent for evaluation results (see [Agent identity](#agent-identity)) |
| `endpoint` | resolved | OTLP base URL for all signals (no `/v1/<signal>` suffix) |
| `traces_endpoint` | resolved | Full OTLP traces URL; pass with `logs_endpoint` |
| `logs_endpoint` | resolved | Full OTLP logs URL, for example the OSIS logs pipeline; pass with `traces_endpoint` |
| `protocol` | resolved | `"http"` or `"grpc"` for both signals |
| `compression` | resolved | `"none"` or `"gzip"` for both signals; must match the pipeline |
| `headers` | `None` | Extra headers for both exporters (lowercased gRPC metadata over gRPC); default `OTEL_EXPORTER_OTLP_HEADERS` |
| `sigv4` | `None` | `True` or `{"service": ..., "region": ...}`: sign both exporters with AWS SigV4 |
| `enable_logs` | `True` | Set up the logs pipeline for evaluation events (only with `set_global=True` and no global `LoggerProvider` yet) |
| `exporter` | `None` | Custom traces `SpanExporter`, used as-is |
| `log_exporter` | `None` | Custom `LogRecordExporter`, used as-is |

Existing setups are left alone: if a global `LoggerProvider` is already configured, `register()` does not replace it and `score()` uses yours. With `set_global=False` no logs pipeline is created. The returned `TracerProvider`'s `force_flush()` and `shutdown()` also flush and shut down the logs pipeline, so `provider.force_flush()` before exit covers both signals.

## `observe()`

Single tracing primitive: works as both a **decorator** and a **context manager**. Creates an OTel span with GenAI semantic convention attributes.

**As a decorator:**

```python
@observe(name="planner", op=Op.INVOKE_AGENT)
def plan(query: str) -> str:
    enrich(model="gpt-4.1")
    return call_llm(query)


# Without parentheses (uses function name, no op)
@observe
def my_function(): ...
```

**As a context manager:**

```python
with observe("thinking", op=Op.CHAT) as span:
    enrich(model="gpt-4.1", input_tokens=1500)
    result = call_llm(prompt)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | Function `__qualname__` (decorator) or `"unnamed"` (context manager) | Span name |
| `op` | `str` | `None` | `gen_ai.operation.name` value. Use `Op` constants or any custom string |
| `kind` | `SpanKind` | `INTERNAL` | OTel span kind. Use `SpanKind.CLIENT` for external service calls |
| `name_from` | `str` | `None` | Decorator only: name of a function parameter whose value, at call time, replaces `name` (for example `@observe(op=Op.EXECUTE_TOOL, name_from="tool_name")`). Falls back to `name` when the argument is `None` or missing |

**Span naming:** When `op` is a well-known value, the span name is `"{op} {name}"` (e.g. `"invoke_agent planner"`). The well-known values are the `Op` constants below. A custom op keeps the plain name: `op="guardrail"` with `name="check"` gives a span named `check`.

**Attributes set automatically:**

| Attribute | When set |
|---|---|
| `gen_ai.operation.name` | When `op` is provided |
| `gen_ai.agent.name` | `op` is `invoke_agent` or `create_agent` only (a `chat` span, such as a traced LLM judge call, is not an agent) |
| `gen_ai.tool.name` | When `op=Op.EXECUTE_TOOL` |
| `gen_ai.tool.type` = `"function"` | When `op=Op.EXECUTE_TOOL` |
| `gen_ai.input.messages` / `gen_ai.output.messages` | All ops except `execute_tool` (decorator only) |
| `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` | When `op=Op.EXECUTE_TOOL` (decorator only) |

**Supported function types:** sync, async, generators, async generators. Errors are captured as span status + exception events.

## `Op`

Constants for well-known `gen_ai.operation.name` values. Any custom string is also accepted.

| Constant | Value | Use for |
|---|---|---|
| `Op.CHAT` | `"chat"` | LLM chat completions |
| `Op.INVOKE_AGENT` | `"invoke_agent"` | Agent invocations |
| `Op.CREATE_AGENT` | `"create_agent"` | Agent creation/setup |
| `Op.EXECUTE_TOOL` | `"execute_tool"` | Tool/function calls |
| `Op.RETRIEVAL` | `"retrieval"` | RAG retrieval steps |
| `Op.EMBEDDINGS` | `"embeddings"` | Embedding generation |
| `Op.GENERATE_CONTENT` | `"generate_content"` | Content generation |
| `Op.TEXT_COMPLETION` | `"text_completion"` | Text completions |

Custom strings work too: `@observe(name="check", op="guardrail")`.

## `enrich()`

Add GenAI semantic convention attributes to the currently active span. Call from inside an `@observe`-decorated function or a `with observe(...)` block.

```python
@observe(name="chat", op=Op.CHAT)
def chat(prompt: str) -> str:
    result = call_llm(prompt)
    enrich(
        model="gpt-4.1",
        provider="openai",
        input_tokens=150,
        output_tokens=50,
        temperature=0.7,
    )
    return result
```

**Parameters:**

| Parameter | Attribute | Description |
|---|---|---|
| `model` | `gen_ai.request.model` | Model name |
| `provider` | `gen_ai.provider.name` | Provider name (openai, anthropic, etc.) |
| `input_tokens` | `gen_ai.usage.input_tokens` | Input token count |
| `output_tokens` | `gen_ai.usage.output_tokens` | Output token count |
| `total_tokens` | `gen_ai.usage.total_tokens` | Total token count |
| `response_id` | `gen_ai.response.id` | Response/completion ID |
| `finish_reason` | `gen_ai.response.finish_reasons` | Finish reason(s) |
| `temperature` | `gen_ai.request.temperature` | Temperature setting |
| `max_tokens` | `gen_ai.request.max_tokens` | Max tokens setting |
| `session_id` | `gen_ai.conversation.id` | Session/conversation ID |
| `agent_id` | `gen_ai.agent.id` | Agent instance ID |
| `agent_description` | `gen_ai.agent.description` | Agent description |
| `tool_definitions` | `gen_ai.tool.definitions` | Tools available to the agent (list of dicts, JSON-encoded) |
| `system_instructions` | `gen_ai.system_instructions` | System prompt |
| `input_messages` / `output_messages` | `gen_ai.input.messages` / `gen_ai.output.messages` | Structured messages (JSON-encoded), for example `[{"role": "user", "parts": [{"type": "text", "content": "..."}]}]` |
| `**extra` | As provided | Any additional key-value attributes (for example `**{"gen_ai.agent.version": "v3"}`) |

All parameters are keyword-only and `None` values are skipped. `finish_reason` is stored as a one-element list. Without a recording span, `enrich()` does nothing.

## `mark_agent_root()`

Mark the current span as its trace's agent root. OpenSearch Agent Traces lists a trace by the span marked with `derived.gen_ai.is_root`. Data Prepper marks the outermost GenAI span of each trace for you (the root when it is a GenAI span, or the agent span below an HTTP handler), so you only need this to choose the span yourself. A span marked at the source wins.

```python
@observe(name="planner", op=Op.INVOKE_AGENT)
def plan(query: str) -> str:
    mark_agent_root()
    ...
```

Without the SDK, set the span attribute `derived.gen_ai.is_root` to `true` on the agent's span.

## `score()`

Records one evaluation result as an OTel **log-based event** named `gen_ai.evaluation.result`, per the [OTel GenAI event spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult). Use any evaluation framework you prefer (autoevals, RAGAS, custom) and submit the results through `score()`.

The event is a LogRecord with `event_name="gen_ai.evaluation.result"`, a timestamp, severity INFO, and the trace context of the **evaluated** span, so it joins that trace without adding a span. It is exported over OTLP logs (`/v1/logs`). Data Prepper routes these events to the **`otel-v1-genai-evaluation`** index (instead of `logs-otel-v1-*`), where they join spans on `traceId` / `spanId`.

```python
from opensearch_genai_observability_sdk_py import Op, observe, score

# Inline: score the active span (the event is parented to it)
with observe("chat", op=Op.CHAT):
    answer = call_llm(question)
    score("relevance", 0.92, label="relevant", evaluator="relevance_judge")

# Offline: score a stored span by id (works after the span has ended)
score(
    name="accuracy",
    value=0.95,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    span_id="89829115c2128845",
    explanation="Weather data matches ground truth",
    evaluator="accuracy_judge",
    evaluator_version="2.1.0",
    evaluator_type="llm_judge",
    score_min=0.0,
    score_max=1.0,
    direction="maximize",
)

# Trace-level: only the trace id is known. No span id is derived; the event
# carries the trace id as its own traceId, with no spanId.
score(
    name="relevance",
    value=0.92,
    trace_id="6ebb9835f43af1552f2cebb9f5165e39",
    attributes={"test.suite.name": "nightly_eval", "test.case.id": "case_001"},
)

# Session-level and human feedback
score(
    "user_satisfaction",
    1.0,
    label="thumbs_up",
    conversation_id="session-456",
    evaluator_type="human",
)
```

**Parameters** (all keyword-only except `name` and `value`):

| Parameter | Type | Attribute | Description |
|---|---|---|---|
| `name` | `str` | `gen_ai.evaluation.name` | Metric name (e.g., `"relevance"`) |
| `value` | `float \| None` | `gen_ai.evaluation.score.value` | Numeric score. Omit it for an error result (`error_type=`) |
| `trace_id` | `str` | event `traceId` | Hex trace ID of the evaluated trace |
| `span_id` | `str` | event `spanId` | Hex span ID of the evaluated span. Without it the result is trace-level |
| `label` | `str` | `gen_ai.evaluation.score.label` | Low-cardinality label (`"relevant"`). `"pass"` / `"fail"` (any case) is the verdict, stored lowercase. A non-verdict label moves to `opensearch.gen_ai.evaluation.judge_label` when `threshold` gives a verdict |
| `explanation` | `str` | `gen_ai.evaluation.explanation` | Evaluator rationale. Kept whole up to 8192 characters (safety cap, ends with `...[truncated]` beyond) |
| `response_id` | `str` | `gen_ai.response.id` | Completion ID of the evaluated response |
| `level` | `str` | `opensearch.gen_ai.evaluation.level` | `span`, `trace`, `session`, or `tool_call`. Default: `span` with a span context, `trace` with only `trace_id`, `session` with only `conversation_id` |
| `evaluator` | `str` | `opensearch.gen_ai.evaluation.evaluator.name` | Evaluator name |
| `evaluator_version` | `str` | `opensearch.gen_ai.evaluation.evaluator.version` | Evaluator version |
| `evaluator_type` | `str` | `opensearch.gen_ai.evaluation.evaluator.type` | `llm_judge`, `code`, `human`, or `external`. Another value (here and for `level` / `direction`) logs a warning and is recorded as-is |
| `direction` | `str` | `opensearch.gen_ai.evaluation.score.direction` | `maximize` or `minimize` |
| `score_min` / `score_max` | `float` | `opensearch.gen_ai.evaluation.score.min` / `.max` | Scale bounds |
| `threshold` | `float` | `opensearch.gen_ai.evaluation.score.threshold` | Pass threshold. `direction` defaults to `maximize` (recorded) |
| `conversation_id` | `str` | `gen_ai.conversation.id` | Evaluated session |
| `agent_name` | `str` | `gen_ai.agent.name` | Evaluated agent. Default: the active span's, else `register(agent_name=)` |
| `agent_version` | `str` | `gen_ai.agent.version` | Its version. Default: the active span's, else `register(agent_version=)` |
| `error_type` | `str` | `error.type` | Set when the evaluator itself failed |
| `judge_trace_id` | `str` | `opensearch.gen_ai.evaluation.judge.trace_id` | Trace of the judge's own run |
| `evaluator_description` | `str` | `opensearch.gen_ai.evaluation.evaluator.description` | What the evaluator checks (criteria or rubric summary). Sent as given: only a `@scorer(description=...)` is cut to 1024 characters |
| `judge_model` | `str` | `gen_ai.request.model` | Model the LLM judge used |
| `judge_prompt_name` | `str` | `gen_ai.prompt.name` | Name of the judge prompt template |
| `judge_prompt_version` | `str` | `gen_ai.prompt.version` | Version of the judge prompt |
| `attributes` | `dict` | | Extra attributes, e.g. `test.*` grouping keys from [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79). The SDK fields win over them; keys the SDK owns (`gen_ai.evaluation.*`, `opensearch.gen_ai.evaluation.*`, `error.type`, `gen_ai.response.id`, `gen_ai.conversation.id`, `gen_ai.agent.*`, `gen_ai.system_instructions`) are dropped with one warning per key per process (use the matching arguments). The judge keys `gen_ai.request.model` and `gen_ai.prompt.*` pass through; `judge_model` / `judge_prompt_*` win over them. `test.*` keys pass through |

**Judge prompt.** Results identify the judge prompt by `gen_ai.prompt.name` + `gen_ai.prompt.version` and link the judge's run by `judge_trace_id`. The full prompt text goes on the judge's own LLM span, never on a result: see [Judge prompt](agent-evaluations.md#judge-prompt).

`opensearch.gen_ai.evaluation.*` names are OpenSearch extensions that mirror the OTel `gen_ai.evaluation.*` names for fields OTel does not define yet. `score()` never sets `opensearch.test.case.expected`; pass it in `attributes` if you want it on an online result.

**Verdicts.** The verdict is `gen_ai.evaluation.score.label` = `pass` / `fail`, as in OTel's own examples. Declare a rule with `label="pass"` / `"fail"` or `threshold` (+ `direction`); without one the result is neutral ("Scored" in OpenSearch Dashboards). With `value` and `threshold`, the SDK computes it: `pass` if `value >= threshold` (`maximize`) or `value <= threshold` (`minimize`), else `fail`. Label precedence, `opensearch.gen_ai.evaluation.judge_label` and the warnings are in [Pass rules and who decides](agent-evaluations.md#pass-rules-and-who-decides).

```python
score("answer_quality", 3.0, threshold=4, score_min=1, score_max=5)  # label=fail
score("latency_s", 9.2, threshold=12, direction="minimize")  # label=pass
score("tool_selection", 1.0, label="pass")
score("relevance", 2.0, label="partially_relevant", threshold=4)  # label=fail + judge_label
```

If you configure OTel yourself instead of calling `register()`, install a global `LoggerProvider` with an OTLP log exporter, otherwise the events are dropped (see [Custom exporters](configuration.md#custom-exporters)).

**Instrumentation scope.** The instrumentation scope is the library that emitted the event, so every SDK event has scope `opensearch-genai-observability-sdk-py` with the SDK version, and no scope attributes. The evaluator is data about the result and goes in `opensearch.gen_ai.evaluation.evaluator.*`.

Trace-level scores never derive a span id from the trace id.

## `evaluate()`

New to evaluations? Start with [Getting started with agent evaluations](getting-started-evaluations.md): the agent, test suite, test case and eval run hierarchy, the online and offline paths, Dashboards deep links and copyable LLM prompts. The full reference is [agent-evaluations.md](agent-evaluations.md).

Run a task against a dataset, score outputs, and record each score as an OTel `gen_ai.evaluation.result` log event. The SDK creates no spans of its own: each case's agent run is its own trace, exactly like production.

`evaluate()` is synchronous, and so are the task and the scorers: call an async agent with `task=lambda q: asyncio.run(agent(q))`, and an async judge the same way inside a sync scorer (see [evaluate()](agent-evaluations.md#evaluate) in the guide). The TypeScript SDK's `evaluate()` also accepts async tasks and scorers: it returns a `BenchmarkResult` directly when the task and scorers are synchronous, and a `Promise<BenchmarkResult>` only when one of them returns a thenable.

```python
from opensearch_genai_observability_sdk_py import evaluate, EvalScore, observe, Op


@observe(op=Op.INVOKE_AGENT)
def my_agent(question: str) -> str:
    return call_llm(question)


def accuracy(input, output, expected) -> EvalScore:
    return EvalScore(name="accuracy", value=1.0 if expected in output else 0.0)


result = evaluate(
    name="rag-agent",
    task=my_agent,
    data=[
        {"input": "What is Python?", "expected": "programming language"},
        {"input": "What causes rain?", "expected": "water vapor"},
    ],
    scores=[accuracy],
    metadata={"app.variant": "v2"},
    run_name="v2 shorter prompt",  # optional, shown in place of the run id
    run_description="Shorter system prompt, same tools",
)
```

Produces, per case:
```
trace (the agent run)
 └── span invoke_agent my_agent        root span, stamped with test.suite.name, test.suite.run.id,
      └── ... normal GenAI spans        test.case.id, test.case.name

event gen_ai.evaluation.result          one per score, parented to the agent root span,
                                        with the test.* ids copied
```

Each `gen_ai.evaluation.result` carries `test.suite.name`, `test.suite.run.id`, `test.case.id`, `test.case.name`, `opensearch.test.case.expected` when the case has an `"expected"` value, `gen_ai.agent.name` / `gen_ai.agent.version` when known, and the run `metadata` (reserved keys are dropped; the score's own `metadata` and the SDK attributes win). Pass/fail/error counts are not sent: readers derive them from the results of a run. The summary `evaluate()` returns is computed locally. Data Prepper routes the events (`eventName == "gen_ai.evaluation.result"`) to the `otel-v1-genai-evaluation` write alias; see [Ingestion](agent-evaluations.md#ingestion).

**Case identity.** Across runs a case is `test.suite.name` + `test.case.id`; within a run, `test.suite.run.id` + `test.case.id`. The default case id is the first 16 hex characters of the SHA-256 of the JSON-encoded input, so editing the input makes a new case: pass `"case_id"` for a stable identity. The full suite, run, case and evaluator tree is in [Data model](agent-evaluations.md#data-model).

- The `test.*` ids are stamped on every root span the task starts (a span with no local parent) by a span processor that `evaluate()` installs once on the global SDK `TracerProvider`. Spans started outside `evaluate()` are not touched. Instrument the agent with the global provider (for example `@observe(op=Op.INVOKE_AGENT)` after `register()`).
- The case is carried in a `contextvars` context variable, so asyncio tasks and `asyncio.to_thread()` see it, but a plain `threading.Thread` or a `ThreadPoolExecutor` worker does not: a root span started there is not stamped. Start the agent's root span in the task's own thread.
- `test.suite.run.id` and `test.case.id` are proposed OTel attributes ([semantic-conventions#1262](https://github.com/open-telemetry/semantic-conventions/issues/1262), [semantic-conventions-genai#79](https://github.com/open-telemetry/semantic-conventions-genai/issues/79)); the other `test.*` names are OTel's.
- **CI/CD.** Inside CI, every evaluation result (and, with `evaluate()`, the agent root spans it starts) carries the OTel `cicd.pipeline.name`, `cicd.pipeline.run.id` and `cicd.pipeline.run.url.full`, read from GitHub Actions, GitLab CI or Jenkins. Pass `ci_pipeline={"cicd.pipeline.run.id": ...}` to `evaluate()` / `Benchmark()` to set or override them. Outside CI nothing is added. The variables read per CI system are in [CI pipeline attributes](agent-evaluations.md#ci-pipeline-attributes).
- `agent_name=` / `agent_version=` name the agent under test (see [Agent identity](#agent-identity)). They go on the root spans that don't name an agent, and on every evaluation result.
- If the task starts no span, the score events carry no trace context (the `test.*` attributes are still set).
- A case whose task raises is not scored: each declared scorer emits one error result instead, named after the scorer (its `@scorer(name=...)`, else the function name), with `error.type` set to the exception class name, no score value, no label and no explanation (the exception message is not sent), and the scorer's evaluator attributes. The run continues. A scorer that raises emits an error result the same way: `error.type`, no score value, no label and no explanation. The exception message stays local in `TestCaseResult.scorer_errors` (judge errors can carry URLs, keys or request bodies).
- Scorers run after the task, outside the agent trace and in the caller's context: a traced LLM judge gets its own trace, unless `evaluate()` itself is called inside an active span, in which case the judge span is a child of that span.
- Each `TestCaseResult` has `status` (see below), `error_type`, `duration_ms`, `scores` and `verdicts` (evaluation name to value and to verdict), and `trace_id` / `span_id` of the agent root span when the task created one. When results of different evaluators share an evaluation name in a run, the first evaluator to report it keeps the plain name and the others are keyed `"<name> (<evaluator>)"` (for example `"faithfulness (deepeval.faithfulness)"`), so no result hides another; the summary uses the same keys.

Each scorer is the evaluator: a named scorer function sets the grade event's `opensearch.gen_ai.evaluation.evaluator.name` (default the function name; a lambda names no evaluator). The TypeScript SDK has no default evaluator name. All events use the SDK instrumentation scope. `EvalScore.label`, `.explanation`, `.metadata` and `.judge_trace_id` (`opensearch.gen_ai.evaluation.judge.trace_id`) go on the event, in `evaluate()` and in `Benchmark.log()`. `EvalScore.evaluator` / `.evaluator_type` name that score's evaluator and win over the scorer's; a score that names a different evaluator gets nothing from the scorer's declaration (see **Declaring rules**). Evaluator names (`EvalScore.evaluator`, `@scorer(evaluator=)`, `Benchmark.log(evaluator=)`) are normalized when set: surrounding whitespace is stripped, a blank name is unset, a `str` enum member gives its value and any other non-string is converted with `str()`.

**Expected output.** A case's `"expected"` value is set on every `gen_ai.evaluation.result` of the case as `opensearch.test.case.expected` (JSON-encoded unless a string, cut at 8192 characters with `opensearch.test.case.expected_truncated`). It is exported by default; pass `record_expected=False` if your test sets hold sensitive data. Inputs and outputs are never exported. Encoding, request size and privacy details are in [Expected output](agent-evaluations.md#expected-output).

**Declaring rules.** Decorate a scorer with `@scorer(...)` to declare its evaluation name and pass rule up front. A scorer may also return a `bool` (a verdict) or an `EvalScore` with `label` (`"pass"` / `"fail"`), `passed`, `threshold`, `direction`, `score_min`, `score_max`; fields set on the `EvalScore` win over the declaration.

```python
from opensearch_genai_observability_sdk_py import scorer


@scorer(
    name="answer_quality",
    threshold=4,
    score_min=1,
    score_max=5,
    evaluator_type="llm_judge",
    description="Answer is correct and grounded in the tool output",
    judge_model="claude-haiku-4-5",
    judge_prompt_name="answer-quality",
    judge_prompt_version="v2",
)
def quality(input, output, expected):
    return judge(input, output)  # float: named "answer_quality", verdict from threshold


@scorer(name="latency_budget", direction="minimize", threshold=12)
def latency(input, output, expected):
    return seconds_taken  # pass when <= 12


@scorer(name="tool_selection")
def tools(input, output, expected):
    return EvalScore(name="tool_selection", value=hit_rate, passed=hit_rate == 1)
```

`@scorer` arguments: `name` (evaluation name of float/bool results and of the error grade; default the function name), `threshold`, `direction` (default `maximize` with a threshold), `score_min` / `score_max`, `evaluator` (`evaluator.name`; default the function name), `evaluator_type`, `description` (`opensearch.gen_ai.evaluation.evaluator.description`), `judge_model` (`gen_ai.request.model`), `judge_prompt_name` / `judge_prompt_version` (`gen_ai.prompt.name` / `.version`). The judge's full system prompt is not a scorer argument (it goes on the judge span, not the results). Everything the scorer declares (rule, `evaluator_type`, description and judge context) goes only on its own results, error results included: an `EvalScore` whose `evaluator` is unset (`""` counts as unset) or equals the scorer's evaluator (`evaluator=` or the function name). A result that names a different evaluator (for example `deepeval.faithfulness`) is never the scorer's own, whatever its `name`, and gets none of them, so its evaluator name, type and judge fields describe the same evaluator. Run and case `metadata` is not part of the declaration: it goes on every result of the run as given, other evaluators' results included, so a judge key set there (`gen_ai.request.model`, `gen_ai.prompt.*`) describes the run, not one evaluator; use the `@scorer` judge arguments to scope it to the scorer's own results. If a scorer declares any of these (a rule field, `evaluator_type`, `description` or a judge field) and returns results in an `evaluate()` run but every one of them, in every case, names another evaluator, they apply to no result and the SDK logs one warning per scorer per process at the end of the run, naming the scorer, its evaluator (declared, or the function name), the fields it declared (a `direction` defaulted from `threshold` is not listed) and the evaluators its results named. A scorer with an own result in any case of the run does not warn. The description is converted with `str()` and cut to 1024 characters, ending with the `...[truncated]` marker when cut (`score(evaluator_description=...)` is not cut; the TypeScript SDK caps `evaluatorDescription` at 1024 everywhere, without a marker). An unknown `direction` or `evaluator_type` raises `ValueError`, and a non-numeric `threshold` / `score_min` / `score_max` raises `TypeError`. A scorer that raises emits its error grade under the declared `name` (the same evaluation name as its successful results) with the scorer's own evaluator and declaration. For a wrapper whose results name another evaluator, declare that evaluator, for example `@scorer(name="faithfulness", evaluator="deepeval.faithfulness", threshold=0.7)`: its results are then the scorer's own and get the rule, and the error grade names the same evaluator. Without it, the error grade and the successful results share the evaluation name but name different evaluators, and the declared rule reaches only the error grade (with the warning above).

**Case status** (local `TestCaseResult.status`, not sent): `error` if the task or any scorer raised; else `fail` if any score of the case has a `fail` verdict; else `pass`. A score's verdict comes from the first of a `pass` / `fail` `label`, `passed`, then `threshold` + `direction`; the same verdict is stored as the event's label, so the event and the local status always agree. Scores with no declared rule never fail a case. The summary keeps `pass_count`, `fail_count` and `error_count`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Benchmark name (`test.suite.name`). Stable across runs |
| `task` | `Callable` | Function that takes input and returns output. Decorate with `@observe(op=Op.INVOKE_AGENT)` for tracing: only agent operations name the agent on the span, so with another op (or none) pass `agent_name=` |
| `data` | `list[dict]` | Dicts with `"input"` and optionally `"expected"` (passed to scorers and sent as `opensearch.test.case.expected`), `"case_id"`, `"case_name"` |
| `scores` | `list[Callable]` | Synchronous scorer functions. Each receives `(input, output, expected)` and returns `EvalScore`, `list[EvalScore]`, a number (any `numbers.Real`, for example `numpy.float32`), or `bool`. Any other return (`numpy.bool_`, `Decimal`, `None`, a dict, a coroutine) logs a warning and emits an error result with `error.type` = `TypeError`, and the case status is `error`; `async def` scorers are not called. Optionally decorated with `@scorer(...)` |
| `metadata` | `dict` | Copied to every `gen_ai.evaluation.result` of the run. SDK-reserved keys (`test.*` ids, `cicd.pipeline.*`, `opensearch.test.case.*`, evaluation fields) are filtered, with one warning per key per process. Judge context keys (`gen_ai.request.model`, `gen_ai.prompt.name`, `gen_ai.prompt.version`) are allowed; `@scorer` judge arguments win over them |
| `run_id` | `str` | `test.suite.run.id`, used as given (default: generated). Use the same id from every CI shard so they report into one run, or an existing run's id to re-score it (send new results with the same run id). Empty or whitespace-only raises `ValueError` |
| `agent_name` / `agent_version` | `str` | Agent under test (see [Agent identity](#agent-identity)) |
| `ci_pipeline` | `dict` | `cicd.pipeline.*` overrides |
| `record_expected` | `bool` | Export `"expected"` as `opensearch.test.case.expected` (default `True`). Pass `False` if test sets hold sensitive data |
| `run_name` | `str` | Human-readable run name (`opensearch.test.suite.run.name`), for example `"v5 shorter prompt"`. Optional; OpenSearch Dashboards falls back to the short run id |
| `run_description` | `str` | Free-text run description (`opensearch.test.suite.run.description`), cut to 1024 characters. Optional |

Returns `BenchmarkResult` with `.summary` (aggregate stats) and `.cases` (per-case results).

## `Benchmark`

Upload pre-computed evaluation results from any framework (RAGAS, DeepEval, pytest, custom). Each `log()` emits one `gen_ai.evaluation.result` event per score, parented to the evaluated agent span when `trace_id` and `span_id` are given; otherwise the events have no trace context. Each result carries the `test.*` ids, `opensearch.test.case.expected` when `expected` is given (unless `Benchmark(record_expected=False)`), the agent, `cicd.pipeline.*` in CI, and the run `metadata` plus the case `metadata` passed to `log()` (reserved keys are dropped; the score's own `metadata` and the SDK attributes win). `scores` is `{name: value}` or a list of `EvalScore` (to declare `label` / `threshold` / bounds, and `evaluator` / `evaluator_type`). `log(evaluator=, evaluator_type=)` sets the default evaluator (`opensearch.gen_ai.evaluation.evaluator.name` / `.type`) for the scores of the call whose `evaluator` is unset (`""` counts as unset) or equals `evaluator=`; an `EvalScore` that sets its own wins, and a score that names a different evaluator gets no default type. If `evaluator_type=` is given and every score logged with that `evaluator=` and `evaluator_type=` during the `Benchmark` run names another evaluator, it applies to no score and `close()` logs one warning per evaluator and type per process. `Benchmark` stamps `test.*` and `cicd.pipeline.*` on the results only, not on agent spans. Use when you already have results and want to visualize them in agent-health or OpenSearch Dashboards.

`log()` takes keyword arguments only. It raises `ValueError` when only one of `trace_id` / `span_id` is given or `status` is not `pass`, `fail` or `error`, and `RuntimeError` after `close()`. `Benchmark` has no evaluator description: metadata cannot set `opensearch.gen_ai.evaluation.evaluator.description` (see [Judge context](agent-evaluations.md#judge-context)).

`error`, `error_type` and `status` set the local result: `error` or `error_type` marks the case `error`; otherwise a `fail` verdict marks it `fail`; `status` sets `pass`, `fail` or `error` explicitly. With `error_type`, every score in `scores` is emitted as an evaluator error result: `error.type` = `error_type`, no score value and no label. `close()` returns the local summary and sends nothing. No spans are created.

`Benchmark(...)` also takes `run_name` and `run_description` (`opensearch.test.suite.run.name` / `.description`, the description cut to 1024 characters), set on every result of the run.

Pass `run_id` to set `test.suite.run.id` (default: generated), for example to report CI shards into one run or to re-score an existing run. An empty or whitespace-only `run_id` raises `ValueError`. See [Run ids](agent-evaluations.md#run-ids).

```python
from opensearch_genai_observability_sdk_py import Benchmark, EvalScore

# Upload results from your eval pipeline
with Benchmark(
    name="nightly-eval",
    metadata={"model": "gpt-4o"},
    agent_name="travel_planner",
    agent_version="v3",
) as b:
    b.log(input="What is Python?", output="A language", scores={"accuracy": 1.0})
    b.log(input="Capital of France?", output="Paris", scores={"accuracy": 1.0}, status="pass")

# Link to existing agent traces (click through from a failed case to the agent trace)
with Benchmark(name="ci-eval") as b:
    b.log(
        input="query",
        output="answer",
        scores={"accuracy": 0.9},
        trace_id="6ebb9835f43af1552f2cebb9f5165e39",
        span_id="89829115c2128845",
    )

# Declare verdicts: the case fails because answer_quality is below its threshold
with Benchmark(name="judged-eval") as b:
    b.log(
        input="query",
        output="answer",
        scores=[
            EvalScore(name="answer_quality", value=3.0, threshold=4, score_min=1, score_max=5),
            EvalScore(name="tool_selection", value=1.0, label="pass"),
        ],
    )

# The evaluator failed: each score is sent as an error result (error.type, no value)
with Benchmark(name="judged-eval", run_id="nightly-2026-11-01") as b:
    b.log(
        input="query",
        scores=[EvalScore(name="answer_quality", value=None)],
        error_type="TimeoutError",
    )
```

## `AWSSigV4OTLPExporter` and `AWSSigV4OTLPLogExporter`

OTLP/HTTP span and log exporters that sign every request with AWS SigV4, using botocore's credential chain (environment variables, `~/.aws/credentials`, IAM role, IMDS). `register(sigv4=...)` builds both for you; build them yourself and pass them as `register(exporter=..., log_exporter=...)` for full control.

```python
from opensearch_genai_observability_sdk_py import (
    AWSSigV4OTLPExporter,
    AWSSigV4OTLPLogExporter,
    register,
)

register(
    exporter=AWSSigV4OTLPExporter(
        endpoint="https://<traces-pipeline>.us-east-1.osis.amazonaws.com/v1/traces",
        service="osis",
        region="us-east-1",
    ),
    log_exporter=AWSSigV4OTLPLogExporter(
        endpoint="https://<logs-pipeline>.us-east-1.osis.amazonaws.com/v1/logs",
        service="osis",
        region="us-east-1",
    ),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | | Full URL of the signal (`.../v1/traces` or `.../v1/logs`); `exporter=` and `log_exporter=` are used as-is |
| `service` | `str` | `"osis"` | SigV4 signing service: `"osis"` for OpenSearch Ingestion, `"es"` for OpenSearch Service direct |
| `region` | `str` | botocore region | AWS region; when omitted, the botocore region (`AWS_DEFAULT_REGION` or `~/.aws/config`) |
| other keyword arguments | | | Passed to the OTel `OTLPSpanExporter` / `OTLPLogExporter` (for example `headers=`, `timeout=`) |

The constructor raises `RuntimeError` if botocore finds no credentials, or if no region is given and botocore has none configured. Both exporters are HTTP only; there is no gRPC variant.

## `OpenSearchTraceRetriever`

Retrieves GenAI trace spans from OpenSearch. Works with any agent library that emits OTel GenAI semantic convention spans indexed by Data Prepper into `otel-v1-apm-span-*`.

```python
from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever

# Option 1: Basic auth (local / docker-compose)
retriever = OpenSearchTraceRetriever(
    host="https://localhost:9200",
    auth=("admin", "admin"),
    verify_certs=False,
)

# Option 2: AWS OpenSearch Service (SigV4) — use this OR Option 1, not both
import boto3
from opensearchpy import RequestsAWSV4SignerAuth

credentials = boto3.Session().get_credentials()
auth = RequestsAWSV4SignerAuth(credentials, "us-west-2", "es")
retriever = OpenSearchTraceRetriever(
    host="https://search-my-domain.us-west-2.es.amazonaws.com",
    auth=auth,
)

# Retrieve all spans of a conversation's traces (or of one trace, by trace ID)
session = retriever.get_traces("my-conversation-id")
for trace in session.traces:
    for span in trace.spans:
        print(f"{span.operation_name}: {span.name} ({span.model})")

# List the agent roots of recent traces (for discovering traces to evaluate)
roots = retriever.list_root_spans(services=["my-agent"], max_results=10)
for root in roots:
    print(root.trace_id, root.agent_name, root.agent_version)

# Filter by time
from datetime import datetime

roots = retriever.list_root_spans(services=["my-agent"], since=datetime(2026, 3, 16))

# Check which traces already have evaluation results (gen_ai.evaluation.result events)
evaluated = retriever.find_evaluated_trace_ids(["trace-id-1", "trace-id-2"])
# ... only results of one evaluator (and/or one evaluation name)
mine = retriever.find_evaluated_trace_ids(["trace-id-1"], evaluator="answer_quality_judge")
```

**Constructor:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"https://localhost:9200"` | OpenSearch endpoint |
| `index` | `str` | `"otel-v1-apm-span-*"` | Index pattern for span data |
| `evaluation_index` | `str` | `"otel-v1-genai-evaluation*"` | Index pattern for evaluation events |
| `auth` | `tuple \| RequestsAWSV4SignerAuth` | `None` | Basic auth tuple or SigV4 auth |
| `verify_certs` | `bool` | `True` | Verify TLS certificates |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `get_traces(identifier, max_spans=10000, *, max_traces=1000)` | `SessionRecord` | By conversation ID: every span of each trace in the conversation (traces whose spans carry `gen_ai.conversation.id`, earliest first). Otherwise by trace ID: every span of that trace. `truncated` is `True` when a limit was reached |
| `list_root_spans(*, services=None, since=None, max_results=50)` | `list[SpanRecord]` | List the agent roots of up to `max_results` recent traces, newest first, one per trace, optionally filtered by service. `since` is read as UTC wall-clock time (its `tzinfo` is ignored), so pass a UTC datetime, for example `datetime.now(timezone.utc) - timedelta(hours=1)` or `local_dt.astimezone(timezone.utc)`; it defaults to 15 minutes ago. A root is a span marked `derived.gen_ai.is_root` (for example the agent span below an HTTP server span), or a span without a parent whose `gen_ai.operation.name` is `invoke_agent` or `create_agent`. A trace is listed by its parentless agent root when it has one, else by its earliest marked span. Other parentless GenAI spans, such as an LLM judge's own `chat` span, are not listed. When nothing matches, one INFO log line names this rule |
| `find_evaluated_trace_ids(trace_ids, *, evaluator=None, evaluation_name=None)` | `set[str]` | Return subset of trace IDs that already have evaluation results (`gen_ai.evaluation.result` events in the evaluation index). Without a filter, a result from any evaluator counts; `evaluator=` / `evaluation_name=` count only that evaluator's (`opensearch.gen_ai.evaluation.evaluator.name`) or that evaluation's (`gen_ai.evaluation.name`) results. A missing index returns an empty set; other search errors raise |

Each `SpanRecord` carries the span's `agent_name` and `agent_version` (`gen_ai.agent.name`, `gen_ai.agent.version`). Pass both to `score()` when you post a result for a retrieved trace, or the result is grouped under "(no version)" on the agent page.

Requires the `[opensearch]` extra: `pip install "opensearch-genai-observability-sdk-py[opensearch]"`
