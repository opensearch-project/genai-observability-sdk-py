"""06_retrieval_and_eval.py — Retrieve stored traces and evaluate them.

Connects to OpenSearch, retrieves recent agent traces, and shows how to
feed them into an evaluation framework. The eval helper uses strands-agents
as a concrete example but is commented out by default.

Prerequisites:
    pip install opensearch-genai-observability-sdk-py[opensearch]

To enable strands evaluation:
    pip install strands-agents-evals
    Then uncomment the evaluate_with_strands() call below.
"""

from datetime import datetime, timedelta, timezone

from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever, score


def main():
    retriever = OpenSearchTraceRetriever(
        host="https://localhost:9200",
        auth=("admin", "My_password_123!@#"),
        verify_certs=False,
    )

    # Discover recent traces
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    roots = retriever.list_root_spans(since=since, max_results=5)
    print(f"Found {len(roots)} recent traces:\n")
    for r in roots:
        print(f"  {r.trace_id[:12]}…  {r.name}  ({r.model or 'no model'})")

    if not roots:
        print("No traces found. Send some first with 01_tracing_basics.py")
        return

    # Skip already-evaluated traces
    trace_ids = [r.trace_id for r in roots]
    already_evaluated = retriever.find_evaluated_trace_ids(trace_ids)
    to_evaluate = [r for r in roots if r.trace_id not in already_evaluated]
    print(f"\n{len(already_evaluated)} already evaluated, {len(to_evaluate)} to evaluate")

    if not to_evaluate:
        return

    # Retrieve full trace
    target = to_evaluate[0]
    session = retriever.get_traces(target.trace_id)
    trace = session.traces[0]

    print(f"\nTrace {target.trace_id[:12]}… — {len(trace.spans)} spans:")
    for span in trace.spans:
        indent = "  " if span.parent_span_id else ""
        tokens = f" ({span.input_tokens}→{span.output_tokens} tok)" if span.input_tokens else ""
        print(f"  {indent}{span.operation_name}: {span.name}{tokens}")

    root = next(s for s in trace.spans if not s.parent_span_id)  # noqa: F841

    # --- Uncomment to evaluate with strands-agents ---
    # evaluate_with_strands(trace, root, target.trace_id)

    # --- Or write a score manually from any eval framework ---
    score(
        name="helpfulness",
        value=0.85,
        trace_id=target.trace_id,
        label="pass",
        explanation="Agent addressed the user query with relevant info",
    )
    print(f"\n✅ Score written to trace {target.trace_id[:12]}…")


def evaluate_with_strands(trace_record, root_span, trace_id):
    """Evaluate a trace using strands-agents HelpfulnessEvaluator and write score back.

    Requires: pip install strands-agents-evals
    """
    import json

    from strands_evals.evaluators import HelpfulnessEvaluator
    from strands_evals.types.evaluation import EvaluationData
    from strands_evals.types.trace import (
        AgentInvocationSpan,
        InferenceSpan,
        Session,
        SpanInfo,
        ToolCall,
        ToolExecutionSpan,
        ToolResult,
        Trace,
    )

    def parse_time(t):
        if not t:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    strands_spans = []
    for s in trace_record.spans:
        info = SpanInfo(
            trace_id=s.trace_id,
            span_id=s.span_id,
            session_id=trace_id,
            parent_span_id=s.parent_span_id or None,
            start_time=parse_time(s.start_time),
            end_time=parse_time(s.end_time),
        )
        if s.operation_name == "execute_tool":
            args = {}
            if s.tool_call_arguments:
                try:
                    args = json.loads(s.tool_call_arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            strands_spans.append(
                ToolExecutionSpan(
                    span_info=info,
                    tool_call=ToolCall(name=s.tool_name, tool_call_id=s.span_id, arguments=args),
                    tool_result=ToolResult(
                        tool_call_id=s.span_id, content=s.tool_call_result or ""
                    ),
                )
            )
        elif s.operation_name in ("invoke_agent", ""):
            strands_spans.append(
                AgentInvocationSpan(
                    span_info=info,
                    user_prompt=(s.input_messages[0].content if s.input_messages else ""),
                    agent_response=(s.output_messages[0].content if s.output_messages else ""),
                    available_tools=[],
                )
            )
        else:
            strands_spans.append(InferenceSpan(span_info=info, messages=[]))

    strands_session = Session(
        session_id=trace_id,
        traces=[Trace(trace_id=trace_id, session_id=trace_id, spans=strands_spans)],
    )

    user_input = root_span.input_messages[0].content if root_span.input_messages else ""
    agent_output = root_span.output_messages[0].content if root_span.output_messages else ""

    evaluator = HelpfulnessEvaluator(model="us.anthropic.claude-sonnet-4-20250514-v1:0")
    case = EvaluationData(
        input=user_input,
        actual_output=agent_output,
        actual_trajectory=strands_session,
    )
    results = evaluator.evaluate(case)
    result = results[0]
    print(f"\nHelpfulness: {result.score} — {result.reason[:200]}")

    score(
        name="helpfulness",
        value=result.score,
        trace_id=trace_id,
        label="pass" if result.score >= 0.7 else "fail",
        explanation=result.reason[:500] if result.reason else "",
    )
    print(f"✅ Eval score written to trace {trace_id[:12]}…")


if __name__ == "__main__":
    main()
