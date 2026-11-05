# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Agent identity (``gen_ai.agent.name``, ``gen_ai.agent.version``) on SDK events."""

import pytest
from opentelemetry import trace

from opensearch_genai_observability_sdk_py import Benchmark, evaluate, score
from opensearch_genai_observability_sdk_py._internal import (
    agent_defaults,
    resolve_agent,
    set_agent_defaults,
)

EVALUATION = "gen_ai.evaluation.result"
NAME = "gen_ai.agent.name"
VERSION = "gen_ai.agent.version"


@pytest.fixture()
def events(log_exporter):
    def _get(name):
        return [
            r.log_record.attributes
            for r in log_exporter.get_finished_logs()
            if r.log_record.event_name == name
        ]

    return _get


@pytest.fixture(autouse=True)
def _reset_defaults():
    set_agent_defaults(None, None)
    yield
    set_agent_defaults(None, None)


class TestResolveAgent:
    def test_nothing_known_leaves_the_agent_out(self):
        assert resolve_agent() == {}

    def test_register_defaults(self):
        set_agent_defaults("travel_planner", "v3")
        assert agent_defaults() == ("travel_planner", "v3")
        assert resolve_agent() == {NAME: "travel_planner", VERSION: "v3"}

    def test_explicit_beats_span_beats_default(self):
        set_agent_defaults("default_agent", "v1")
        tracer = trace.get_tracer("t")
        with tracer.start_as_current_span("invoke_agent", attributes={NAME: "span_agent"}) as span:
            assert resolve_agent(span=span) == {NAME: "span_agent", VERSION: "v1"}
            assert resolve_agent("explicit", "v9", span=span) == {NAME: "explicit", VERSION: "v9"}


class TestScore:
    def test_register_default_is_stamped(self, events):
        set_agent_defaults("travel_planner", "v2")
        score("helpfulness", 0.8, trace_id="6ebb9835f43af1552f2cebb9f5165e39")
        a = events(EVALUATION)[0]
        assert a[NAME] == "travel_planner"
        assert a[VERSION] == "v2"

    def test_inline_reads_the_active_span(self, events):
        set_agent_defaults("default_agent")
        tracer = trace.get_tracer("t")
        with tracer.start_as_current_span(
            "invoke_agent", attributes={NAME: "support_bot", VERSION: "v4"}
        ):
            score("helpfulness", 0.8)
        a = events(EVALUATION)[0]
        assert a[NAME] == "support_bot"
        assert a[VERSION] == "v4"

    def test_explicit_override(self, events):
        set_agent_defaults("default_agent", "v1")
        score("helpfulness", 0.8, agent_name="rag_faq", agent_version="v7")
        a = events(EVALUATION)[0]
        assert a[NAME] == "rag_faq"
        assert a[VERSION] == "v7"

    def test_unknown_agent_is_left_out(self, events):
        score("helpfulness", 0.8)
        a = events(EVALUATION)[0]
        assert NAME not in a
        assert VERSION not in a


class TestBenchmark:
    def test_declared_agent_on_score_events(self, events):
        with Benchmark(name="suite", agent_name="travel_planner", agent_version="v3") as b:
            b.log(input="q", output="a", scores={"accuracy": 1.0, "tone": 0.5})
        recs = events(EVALUATION)
        assert len(recs) == 2
        for a in recs:
            assert a[NAME] == "travel_planner"
            assert a[VERSION] == "v3"

    def test_register_default(self, events):
        set_agent_defaults("support_bot", "v1")
        with Benchmark(name="suite") as b:
            b.log(input="q", output="a", scores={"accuracy": 1.0})
        assert events(EVALUATION)[0][NAME] == "support_bot"
        assert events(EVALUATION)[0][VERSION] == "v1"

    def test_declared_version_beats_metadata(self, events):
        with Benchmark(name="suite", metadata={VERSION: "meta"}, agent_version="v5") as b:
            b.log(input="q", output="a", scores={"accuracy": 1.0})
        assert events(EVALUATION)[0][VERSION] == "v5"

    def test_metadata_version_kept_without_a_declared_one(self, events):
        with Benchmark(name="suite", metadata={VERSION: "meta"}) as b:
            b.log(input="q", output="a", scores={"accuracy": 1.0})
        assert events(EVALUATION)[0][VERSION] == "meta"


class TestEvaluate:
    def _agent(self, attributes=None):
        tracer = trace.get_tracer("agent")

        def task(question):
            with tracer.start_as_current_span("invoke_agent", attributes=attributes or {}):
                return f"answer to {question}"

        return task

    def test_declared_agent_on_root_spans_and_events(self, events, exporter):
        evaluate(
            "suite",
            task=self._agent(),
            data=[{"input": "q1"}, {"input": "q2"}],
            scores=[lambda i, o, e: 1.0],
            agent_name="travel_planner",
            agent_version="v3",
        )
        roots = [s for s in exporter.get_finished_spans() if s.name == "invoke_agent"]
        assert len(roots) == 2
        for span in roots:
            assert span.attributes[NAME] == "travel_planner"
            assert span.attributes[VERSION] == "v3"
            assert span.attributes["test.suite.name"] == "suite"
        recs = events(EVALUATION)
        assert len(recs) == 2
        assert all(a[NAME] == "travel_planner" and a[VERSION] == "v3" for a in recs)

    def test_root_span_agent_wins_over_the_default(self, events, exporter):
        set_agent_defaults("default_agent", "v1")
        evaluate(
            "suite",
            task=self._agent({NAME: "support_bot"}),
            data=[{"input": "q1"}],
            scores=[lambda i, o, e: 1.0],
        )
        root = [s for s in exporter.get_finished_spans() if s.name == "invoke_agent"][0]
        # The span's own name is kept; the missing version is filled from the default.
        assert root.attributes[NAME] == "support_bot"
        assert root.attributes[VERSION] == "v1"
        assert events(EVALUATION)[0][NAME] == "support_bot"
        assert events(EVALUATION)[0][VERSION] == "v1"

    def test_each_case_reports_its_own_agent(self, events):
        tracer = trace.get_tracer("agent")
        agents = iter(["a1", "a2"])

        def task(question):
            with tracer.start_as_current_span("invoke_agent", attributes={NAME: next(agents)}):
                return question

        evaluate(
            "suite",
            task=task,
            data=[{"input": "x"}, {"input": "y"}],
            scores=[lambda i, o, e: 1.0],
        )
        assert [a[NAME] for a in events(EVALUATION)] == ["a1", "a2"]

    def test_unknown_agent_is_left_out(self, events):
        evaluate("suite", task=lambda q: q, data=[{"input": "x"}], scores=[lambda i, o, e: 1.0])
        assert NAME not in events(EVALUATION)[0]
