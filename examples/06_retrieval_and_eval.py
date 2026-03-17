"""06_retrieval_and_eval.py — Retrieve stored traces and evaluate them.

Connects to OpenSearch, retrieves recent agent traces, and shows how to
feed them into an evaluation framework. The eval section uses strands-agents
as a concrete example but is commented out to avoid adding it as a dependency.

Prerequisites:
    pip install opensearch-genai-observability-sdk-py[opensearch]
    # For eval (optional): pip install strands-agents-evals
"""

from datetime import datetime, timedelta, timezone

from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever, score

# --- Connect to OpenSearch ---

retriever = OpenSearchTraceRetriever(
    host="https://localhost:9200",
    auth=("admin", "My_password_123!@#"),
    verify_certs=False,
)

# --- Discover recent traces ---

since = datetime.now(timezone.utc) - timedelta(hours=1)
roots = retriever.list_root_spans(since=since, max_results=5)
print(f"Found {len(roots)} recent traces:\n")
for r in roots:
    print(f"  {r.trace_id[:12]}…  {r.name}  ({r.model or 'no model'})")

if not roots:
    print("No traces found. Send some first with 01_tracing_basics.py")
    exit()

# --- Skip already-evaluated traces ---

trace_ids = [r.trace_id for r in roots]
already_evaluated = retriever.find_evaluated_trace_ids(trace_ids)
to_evaluate = [r for r in roots if r.trace_id not in already_evaluated]
print(f"\n{len(already_evaluated)} already evaluated, {len(to_evaluate)} to evaluate")

# --- Retrieve full trace for evaluation ---

if to_evaluate:
    target = to_evaluate[0]
    session = retriever.get_traces(target.trace_id)
    trace = session.traces[0]

    print(f"\nTrace {target.trace_id[:12]}… — {len(trace.spans)} spans:")
    for span in trace.spans:
        indent = "  " if span.parent_span_id else ""
        tokens = f" ({span.input_tokens}→{span.output_tokens} tok)" if span.input_tokens else ""
        print(f"  {indent}{span.operation_name}: {span.name}{tokens}")

    # --- Evaluate the trace ---
    #
    # This is where you'd pass the retrieved trace data to your eval framework.
    # Below is a concrete example using strands-agents/evals (commented out to
    # avoid adding it as a dependency).
    #
    # from strands_agents_evals.evaluators import HelpfulnessEvaluator
    #
    # # Extract the user input and agent output from the root span
    # root = next(s for s in trace.spans if not s.parent_span_id)
    # user_input = root.input_messages[0].content if root.input_messages else ""
    # agent_output = root.output_messages[0].content if root.output_messages else ""
    #
    # evaluator = HelpfulnessEvaluator(model="us.anthropic.claude-sonnet-4-20250514-v1:0")
    # result = evaluator.evaluate(input=user_input, output=agent_output)
    # print(f"\nHelpfulness: {result.score} — {result.explanation}")

    # --- Write score back to OpenSearch ---
    #
    # Whether you use strands-agents, RAGAS, autoevals, or your own logic,
    # call score() to write the result back as an OTel span in the same trace.

    # Example: hardcoded score (replace with your evaluator's output)
    score(
        name="helpfulness",
        value=0.85,
        trace_id=target.trace_id,
        label="pass",
        explanation="Agent addressed the user query with relevant information",
    )
    print(f"\n✅ Score written to trace {target.trace_id[:12]}…")
