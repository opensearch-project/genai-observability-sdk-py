# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""OpenSearch AI Observability SDK.

OTEL-native tracing and scoring for LLM applications. Evaluation results are
emitted as ``gen_ai.evaluation.result`` OTel log events.
"""

from opensearch_genai_observability_sdk_py.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSummary,
    EvalScore,
    ScorerSpec,
    ScoreSummary,
    TestCaseResult,
    evaluate,
    scorer,
)
from opensearch_genai_observability_sdk_py.enrich import enrich, mark_agent_root
from opensearch_genai_observability_sdk_py.exporters import (
    AWSSigV4OTLPExporter,
    AWSSigV4OTLPLogExporter,
)
from opensearch_genai_observability_sdk_py.observe import Op, observe
from opensearch_genai_observability_sdk_py.register import register
from opensearch_genai_observability_sdk_py.retrieval import OpenSearchTraceRetriever
from opensearch_genai_observability_sdk_py.score import score

__all__ = [
    # Setup
    "register",
    # Tracing
    "observe",
    "Op",
    "enrich",
    "mark_agent_root",
    # Scoring
    "score",
    # Benchmarks
    "Benchmark",
    "evaluate",
    "EvalScore",
    "scorer",
    "ScorerSpec",
    "BenchmarkResult",
    "BenchmarkSummary",
    "ScoreSummary",
    "TestCaseResult",
    # Retrieval
    "OpenSearchTraceRetriever",
    # Exporters
    "AWSSigV4OTLPExporter",
    "AWSSigV4OTLPLogExporter",
]
