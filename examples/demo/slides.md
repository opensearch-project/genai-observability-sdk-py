---
title: OpenSearch GenAI Observability SDK
sub_title: Experiments Demo
author: OpenSearch Team
---

# The Problem

Agent developers have no standard way to answer:

> "Is my agent getting better or worse?"

You change a prompt, swap a model, add a tool...

**But how do you measure the impact?**

<!-- end_slide -->

# What We Built

The SDK bridges **two worlds**:

| Observability | Evaluation |
|---------------|------------|
| What happened inside the agent | How good was the output |
| LLM calls, tools, tokens, latency | Scores, pass rates, regressions |

Both flow as **standard OTel spans** → OpenSearch → one UI.

<!-- end_slide -->

# Three Ways to Use It

```
┌─────────────────────────────────────────────────┐
│  1. evaluate()        All-in-one eval runner     │
│  2. Experiment.log()  Bridge for DeepEval/RAGAS  │
│  3. trace_id linking  Scores → production traces │
└─────────────────────────────────────────────────┘
```

<!-- end_slide -->

# Demo 1: evaluate()

**Story**: I changed my prompt. Did the agent get better?

```python
result = evaluate(
    name="opensearch-qa",
    task=my_agent,          # @observe → full traces
    data=TEST_CASES,
    scores=[keyword_coverage, completeness],
    metadata={"prompt": "detailed"},
)
```

```bash +exec
echo "▶ Running Demo 1..."
echo "  python examples/demo/demo1_evaluate.py"
```

<!-- end_slide -->

# Demo 1: What to See

**In the UI** → `http://localhost:4001/experiments`

1. Click **opensearch-qa** → 2 runs (basic vs detailed)
2. Score metrics: green/yellow/red progress bars
3. Select both → **Compare** → delta badges
4. **View Trace** → agent waterfall:

```
invoke_agent opensearch-qa-agent
 ├── retrieval retrieve
 └── chat call_bedrock  (tokens, latency)
```

<!-- end_slide -->

# Demo 2: DeepEval Integration

**Story**: My team uses DeepEval. I don't want to rewrite our eval pipeline.

```python
# Run your existing eval framework...
answer_relevancy.measure(test_case)
correctness.measure(test_case)

# ...then upload with 3 lines:
with Experiment(name="deepeval-run") as exp:
    exp.log(
        input=question, output=answer,
        scores={"relevancy": metric.score},
    )
```

```bash +exec
echo "▶ Running Demo 2..."
echo "  python examples/demo/demo2_deepeval.py"
```

<!-- end_slide -->

# Demo 3: Trace-Linked Experiments

**Story**: Agent runs in CI. Link eval scores back to execution traces.

```python
# Step 1: Run agent, capture trace ID
with tracer.start_as_current_span("ci_run") as span:
    output = agent(question)
    trace_id = format(span.get_span_context().trace_id, '032x')

# Step 2: Upload scores with trace link
exp.log(
    input=question, output=output,
    scores={"accuracy": 0.95},
    trace_id=trace_id,  # ← click "View Trace" in UI
)
```

```bash +exec
echo "▶ Running Demo 3..."
echo "  python examples/demo/demo3_trace_linked.py"
```

<!-- end_slide -->

# The Differentiator

```
Failed test case
     │
     │  click "View Trace"
     ▼
Full agent execution waterfall
 ├── retrieval retrieve
 └── chat call_bedrock
      ├── input tokens: 850
      ├── output tokens: 200
      └── latency: 1.2s
```

**Eval scores linked to live traces. One click.**

No other tool does this.

<!-- end_slide -->

# Architecture

```
┌──────────────────────────────────┐
│  Your Agent (@observe decorated) │
│  retrieve() → call_bedrock()     │
└──────────────┬───────────────────┘
               │ OTel spans
               ▼
      ┌─────────────────┐
      │ agent-health     │
      │ :4001            │
      │                  │
      │ /v1/traces       │ ← SDK sends here
      │ /experiments     │ ← compare runs
      │ /agent-traces    │ ← debug traces
      └─────────────────┘
```

<!-- end_slide -->

# Summary

| Mode | When to use |
|------|-------------|
| `evaluate()` | Iterating on agent during dev |
| `Experiment.log()` | Already have DeepEval/RAGAS/pytest |
| `log(trace_id=...)` | Link CI scores to production traces |

**SDK**: `pip install opensearch-genai-observability-sdk-py`

**UI**: `http://localhost:4001/experiments`
