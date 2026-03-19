# Agent Health Demo — End-to-End

Three demos showing how an agent developer uses the OpenSearch GenAI Observability SDK
to trace, evaluate, and compare AI agents.

All demos use a **real RAG agent** calling AWS Bedrock (Claude 3.5 Haiku) to answer
questions about OpenSearch, with full OTel tracing.

---

## Prerequisites

### AWS Bedrock access

Your AWS credentials need Bedrock model access. Quick test:

```bash
python -c "
import boto3, json
client = boto3.client('bedrock-runtime', region_name='us-east-1')
r = client.invoke_model(
    modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    body=json.dumps({'anthropic_version':'bedrock-2023-05-31','messages':[{'role':'user','content':[{'type':'text','text':'Hi'}]}],'max_tokens':10})
)
print('OK:', json.loads(r['body'].read())['content'][0]['text'])
"
```

If using a different region or model:
```bash
export AWS_REGION=us-west-2
export BEDROCK_MODEL=us.anthropic.claude-3-5-haiku-20241022-v1:0
```

### Agent-health server (Terminal 1)

```bash
cd repos/agent-health
npm run server
# Serves UI + OTLP receiver on http://localhost:4001
```

Keep this running for all demos. Data is in-memory — restarting clears everything.

After running all 3 demos, open `http://localhost:4001/experiments` to see:
- **opensearch-qa** — 2 runs (basic vs detailed prompt), custom scorers
- **opensearch-qa-deepeval** — 1 run, DeepEval LLM-judged scores
- **opensearch-qa-traced** — 1 run, scores linked to agent execution traces

---

## Demo 1: Built-in Eval Runner — `evaluate()`

**Story**: *"I'm iterating on my agent. I changed the prompt — did it get better?"*

Runs 5 questions against 2 agent versions (`basic` vs `detailed` prompt), scores with
3 custom Python scorers (`keyword_coverage`, `completeness`, `has_specifics`), and sends
all traces + scores as OTel spans to agent-health.

### Setup

```bash
cd /path/to/genai-observability-sdk-py

python -m venv .venv-demo1
source .venv-demo1/bin/activate

pip install -e .          # SDK (editable install)
pip install boto3 requests
```

### Run

```bash
source .venv-demo1/bin/activate
python examples/demo/demo1_evaluate.py    # ~40 seconds
```

**What to show**:
1. `http://localhost:4001/experiments` → click **opensearch-qa**
2. See 2 runs: "basic" and "detailed" — click each to see scores
3. Donut chart shows pass rate, progress bars show score metrics (green/yellow/red)
4. Scroll down → test case table with colored score badges per case
5. Check both run checkboxes → click **Compare** → delta badges (green = improved)
6. Click **View Trace** on any test case → full agent waterfall:
   ```
   invoke_agent opensearch-qa-agent
    ├── retrieval retrieve        (keyword search over knowledge base)
    └── chat call_bedrock         (Claude 3.5 Haiku, tokens visible)
   ```

**SDK code that powers this**:
```python
from opensearch_genai_observability_sdk_py import evaluate, Score

result = evaluate(
    name="opensearch-qa",
    task=my_agent,                          # @observe-decorated → traced
    data=TEST_CASES,                        # list of {input, expected}
    scores=[keyword_coverage, completeness], # scorer functions
    metadata={"model": "claude-3.5-haiku", "prompt": "detailed"},
    record_io=True,
)
```

---

## Demo 2: External Eval Framework — DeepEval

**Story**: *"My team already uses DeepEval. I don't want to rewrite our eval pipeline."*

Runs the agent on 5 test cases, scores with DeepEval's LLM-as-judge metrics
(`AnswerRelevancyMetric`, `GEval correctness`) via Bedrock, and uploads results
via `Experiment.log()`.

### Setup

```bash
cd /path/to/genai-observability-sdk-py

python -m venv .venv-demo2
source .venv-demo2/bin/activate

pip install -e .          # SDK (editable install)
pip install boto3 requests
pip install deepeval      # LLM-as-judge framework
```

### Run

```bash
source .venv-demo2/bin/activate
python examples/demo/demo2_deepeval.py    # ~90 seconds (LLM judge calls)
```

**What to show**:
1. `http://localhost:4001/experiments` → click **opensearch-qa-deepeval**
2. See LLM-judged scores: `answer_relevancy`, `correctness`
3. Compare with Demo 1's heuristic scores — different approaches, same UI

**Key message**: The SDK is a bridge. You keep your eval framework. You add 3 lines
to upload results to OpenSearch:

```python
from opensearch_genai_observability_sdk_py import Experiment

with Experiment(name="deepeval-run", metadata={"framework": "deepeval"}) as exp:
    for case in results:
        exp.log(
            input=case["question"],
            output=case["answer"],
            scores={"relevancy": deepeval_metric.score},
        )
```

---

## Demo 3: Trace-Linked Experiments

**Story**: *"My agent runs in CI. I want to connect eval scores back to execution traces."*

Runs the agent, captures the trace ID from each execution, scores the outputs, and
uploads via `Experiment.log(trace_id=...)`. In the UI, each test case has a **View Trace**
button linked to the real agent execution waterfall.

### Setup

```bash
cd /path/to/genai-observability-sdk-py

python -m venv .venv-demo3
source .venv-demo3/bin/activate

pip install -e .          # SDK (editable install)
pip install boto3 requests
```

### Run

```bash
source .venv-demo3/bin/activate
python examples/demo/demo3_trace_linked.py    # ~30 seconds
```

**What to show**:
1. `http://localhost:4001/experiments` → click **opensearch-qa-traced**
2. Click **View Trace** on any test case
3. Navigate to the full agent execution waterfall:
   ```
   production_run
    └── invoke_agent opensearch-qa-agent
         ├── retrieval retrieve
         └── chat call_bedrock
   ```

**Key message**: This is the differentiator. No other tool connects eval scores
to live traces in one view. Failed score → one click → see exactly which LLM call
produced the bad output.

```python
exp.log(
    input=question,
    output=answer,
    scores={"accuracy": 0.95},
    trace_id="7e46b7ee955aab8f...",   # ← links to the agent execution trace
    span_id="a3b4c5d6e7f8a9b0...",
)
```

---

## Clean Start

If you want to clear old data and start fresh:

```bash
# 1. Stop agent-health (Ctrl+C in Terminal 1, or:)
kill $(lsof -ti:4001) 2>/dev/null

# 2. Restart it
cd repos/agent-health && npm run server

# 3. Re-run demos (activate the appropriate venv first)
source .venv-demo1/bin/activate && python examples/demo/demo1_evaluate.py
source .venv-demo2/bin/activate && python examples/demo/demo2_deepeval.py
source .venv-demo3/bin/activate && python examples/demo/demo3_trace_linked.py
```

To remove all venvs:
```bash
rm -rf .venv-demo1 .venv-demo2 .venv-demo3
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ResourceNotFoundException: Access denied. This Model is marked by provider as Legacy` | Model ID is legacy. Use `us.anthropic.claude-3-5-haiku-20241022-v1:0` (set via `BEDROCK_MODEL` env var) |
| `ValidationException: Invocation of model ID ... with on-demand throughput isn't supported` | Need cross-region inference profile. Use `us.` prefixed model IDs |
| Experiments page is empty | Agent-health stores spans in memory. Re-run the demo scripts |
| `DependencyConflict` warnings from DeepEval | Harmless — optional integrations not installed. Demos work fine |
| Agent-health not responding on :4001 | Run `cd repos/agent-health && npm run server` |
| `ModuleNotFoundError: No module named 'boto3'` | Run `pip install boto3 requests` |
| `ModuleNotFoundError: No module named 'deepeval'` | Run `pip install deepeval` (only needed for Demo 2) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Your Agent Code (@observe decorated)                    │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │  retrieve() │──▶│ call_bedrock()│──▶│   response    │  │
│  └────────────┘   └──────────────┘   └───────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │ OTel spans (traces + scores)
                           ▼
              ┌────────────────────────┐
              │  agent-health:4001     │
              │  ├── /v1/traces (OTLP) │  ← SDK sends spans here
              │  ├── /experiments (UI) │  ← view + compare results
              │  └── /agent-traces (UI)│  ← view execution traces
              └────────────────────────┘
              (in-memory, ~0.1 MB for all 3 demos)

Scoring flows:
  Demo 1: evaluate()          → traces + scores as child spans (all-in-one)
  Demo 2: DeepEval + log()    → external scores uploaded to OpenSearch
  Demo 3: run + log(trace_id) → scores linked back to production traces
```

## Files

```
examples/demo/
├── README.md                  ← you are here
├── _common.py                 ← shared: OTLP exporter, Bedrock agent, test cases
├── demo1_evaluate.py          ← evaluate() with custom scorers
├── demo2_deepeval.py          ← DeepEval LLM-as-judge + Experiment.log()
└── demo3_trace_linked.py      ← trace-linked experiments
```
