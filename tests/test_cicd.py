"""OTel ``cicd.pipeline.*`` on evaluation results inside CI, never outside it."""

from __future__ import annotations

import pytest

from opensearch_genai_observability_sdk_py import evaluate
from opensearch_genai_observability_sdk_py._internal import ci_pipeline_attributes

GITHUB = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_WORKFLOW": "agent-evals",
    "GITHUB_RUN_ID": "120912",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "acme/agents",
}


def test_outside_ci_sets_nothing():
    assert ci_pipeline_attributes(environ={}) == {}


def test_github_actions():
    assert ci_pipeline_attributes(environ=GITHUB) == {
        "cicd.pipeline.name": "agent-evals",
        "cicd.pipeline.run.id": "120912",
        "cicd.pipeline.run.url.full": "https://github.com/acme/agents/actions/runs/120912",
    }


def test_gitlab_ci():
    env = {
        "GITLAB_CI": "true",
        "CI_PIPELINE_NAME": "evals",
        "CI_PIPELINE_ID": "7",
        "CI_PIPELINE_URL": "https://gl/p/7",
    }
    assert ci_pipeline_attributes(environ=env) == {
        "cicd.pipeline.name": "evals",
        "cicd.pipeline.run.id": "7",
        "cicd.pipeline.run.url.full": "https://gl/p/7",
    }


def test_jenkins():
    env = {
        "JENKINS_URL": "https://ci",
        "BUILD_ID": "42",
        "JOB_NAME": "evals",
        "BUILD_URL": "https://ci/job/evals/42/",
    }
    assert ci_pipeline_attributes(environ=env)["cicd.pipeline.run.id"] == "42"


def test_override_and_unknown_keys(caplog):
    out = ci_pipeline_attributes({"cicd.pipeline.run.id": "x1", "cicd.bogus": "y"}, environ={})
    assert out == {"cicd.pipeline.run.id": "x1"}
    assert "cicd.bogus" in caplog.text


@pytest.fixture()
def events(log_exporter):
    def _get(name):
        return [r for r in log_exporter.get_finished_logs() if r.log_record.event_name == name]

    return _get


def _ci_env(monkeypatch):
    for k in ("GITLAB_CI", "JENKINS_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in GITHUB.items():
        monkeypatch.setenv(k, v)


def test_evaluation_events_and_root_span_carry_cicd(monkeypatch, events, exporter):
    from opensearch_genai_observability_sdk_py import observe

    @observe(name="agent")
    def agent(x):
        return x

    _ci_env(monkeypatch)
    evaluate(name="ci", task=agent, data=[{"input": "q1"}], scores=[lambda i, o, e: 1.0])
    attrs = events("gen_ai.evaluation.result")[0].log_record.attributes
    assert attrs["cicd.pipeline.run.id"] == "120912"
    assert attrs["cicd.pipeline.name"] == "agent-evals"
    roots = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert roots and all(s.attributes["cicd.pipeline.run.id"] == "120912" for s in roots)


def test_outside_ci_events_have_no_cicd(monkeypatch, events):
    for k in ("GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL"):
        monkeypatch.delenv(k, raising=False)
    evaluate(name="local", task=lambda x: x, data=[{"input": "q1"}], scores=[lambda i, o, e: 1.0])
    attrs = events("gen_ai.evaluation.result")[0].log_record.attributes
    assert not any(k.startswith("cicd.") for k in attrs)


def test_benchmark_score_events_carry_cicd(monkeypatch, events):
    from opensearch_genai_observability_sdk_py import Benchmark

    _ci_env(monkeypatch)
    with Benchmark(name="ci-upload") as b:
        b.log(input="q1", scores={"a": 1.0})
    attrs = events("gen_ai.evaluation.result")[0].log_record.attributes
    assert attrs["cicd.pipeline.run.id"] == "120912"
    assert attrs["cicd.pipeline.run.url.full"].endswith("/actions/runs/120912")
