# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Mini OTEL Collector — HTTP receiver, prints spans to terminal.

No Docker needed. Receives OTLP/HTTP protobuf on port 4318 and
pretty-prints every span.

Usage:
    # Terminal 1: start the collector
    python examples/mini_collector_http.py

    # Terminal 2: run the agent
    python examples/agent_http.py
"""

import uvicorn
from fastapi import FastAPI, Request, Response
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

app = FastAPI()

# Colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

SPAN_KIND_MAP = {
    "workflow": f"{MAGENTA}workflow{RESET}",
    "invoke_agent": f"{CYAN}agent{RESET}",
    "task": f"{GREEN}task{RESET}",
    "execute_tool": f"{YELLOW}tool{RESET}",
}


def format_trace_id(tid: bytes) -> str:
    return tid.hex()


def format_span_id(sid: bytes) -> str:
    return sid.hex()


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000


def print_span(span, resource_attrs: dict):
    """Pretty-print a single span."""
    attrs = {}
    for kv in span.attributes:
        key = kv.key
        val = kv.value
        if val.HasField("string_value"):
            attrs[key] = val.string_value
        elif val.HasField("int_value"):
            attrs[key] = val.int_value
        elif val.HasField("double_value"):
            attrs[key] = val.double_value
        elif val.HasField("bool_value"):
            attrs[key] = val.bool_value
        else:
            attrs[key] = str(val)

    span_kind = attrs.get("gen_ai.operation.name", "")
    is_score = "gen_ai.evaluation.name" in attrs
    kind_label = SPAN_KIND_MAP.get(span_kind, f"{RED}score{RESET}" if is_score else "internal")

    trace_id = format_trace_id(span.trace_id)
    span_id = format_span_id(span.span_id)
    parent_id = format_span_id(span.parent_span_id) if span.parent_span_id else "none (root)"
    duration_ms = ns_to_ms(span.end_time_unix_nano - span.start_time_unix_nano)
    service = resource_attrs.get("service.name", "unknown")

    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"  {BOLD}Span:{RESET} {span.name}  [{kind_label}]  {DIM}{duration_ms:.1f}ms{RESET}")
    print(f"  {DIM}Trace:  {trace_id}{RESET}")
    print(f"  {DIM}Span:   {span_id}{RESET}")
    print(f"  {DIM}Parent: {parent_id}{RESET}")
    print(f"  {DIM}Service: {service}{RESET}")

    # Print interesting attributes
    for key in [
        "gen_ai.entity.input",
        "gen_ai.entity.output",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.evaluation.name",
        "gen_ai.evaluation.score.value",
        "gen_ai.evaluation.source",
        "gen_ai.evaluation.explanation",
        "gen_ai.evaluation.trace_id",
        "gen_ai.evaluation.span_id",
        "gen_ai.conversation.id",
    ]:
        if key in attrs:
            val = attrs[key]
            # Truncate long values
            val_str = str(val)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(f"  {CYAN}{key}{RESET}: {val_str}")


@app.post("/v1/traces")
async def receive_traces(request: Request):
    """OTLP/HTTP trace receiver."""
    body = await request.body()

    req = ExportTraceServiceRequest()
    req.ParseFromString(body)

    span_count = 0
    for resource_spans in req.resource_spans:
        # Extract resource attributes
        resource_attrs = {}
        for kv in resource_spans.resource.attributes:
            if kv.value.HasField("string_value"):
                resource_attrs[kv.key] = kv.value.string_value

        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                print_span(span, resource_attrs)
                span_count += 1

    if span_count > 0:
        print(f"\n{GREEN}✓ Received {span_count} span(s) via HTTP{RESET}\n")

    resp = ExportTraceServiceResponse()
    return Response(
        content=resp.SerializeToString(),
        media_type="application/x-protobuf",
    )


if __name__ == "__main__":
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Mini OTEL Collector — HTTP on port 4318{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print("  Listening: http://0.0.0.0:4318/v1/traces")
    print("  Protocol:  OTLP/HTTP (protobuf)")
    print()
    print("  Test with: python examples/agent_http.py")
    print(f"{BOLD}{'=' * 70}{RESET}\n")
    uvicorn.run(app, host="0.0.0.0", port=4318, log_level="warning")  # noqa: S104
