"""Test the helpers shipped with the Python examples."""

import io
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from agentwerk import Task, Werk


example_path = Path(__file__).parents[1] / "examples" / "web_search.py"
example_spec = spec_from_file_location("web_search_example", example_path)
assert example_spec is not None and example_spec.loader is not None
web_search = module_from_spec(example_spec)
example_spec.loader.exec_module(web_search)

coding_harness_path = Path(__file__).parents[1] / "examples" / "coding_harness.py"
coding_harness_spec = spec_from_file_location(
    "coding_harness_example", coding_harness_path
)
assert coding_harness_spec is not None and coding_harness_spec.loader is not None
coding_harness = module_from_spec(coding_harness_spec)
coding_harness_spec.loader.exec_module(coding_harness)


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_brave_search_tool_sends_bounded_authenticated_request(monkeypatch):
    requests = []

    def open_request(request, timeout, context):
        requests.append((request, timeout, context))
        return Response(
            b'{"web":{"results":[{"title":"Agentwerk","url":"https://example.com",'
            b'"description":"A minimal agentic loop."}]}}'
        )

    monkeypatch.setattr(web_search, "urlopen", open_request)
    search = web_search.brave_search_tool("secret")

    result = search(query="agent harness", count=100)

    assert search._agentwerk_name == "brave_search"
    assert search._agentwerk_concurrent is True
    assert search._agentwerk_timeout == 60
    assert "leads, not sources." in search._agentwerk_description
    request, timeout, context = requests[0]
    assert timeout == 60
    assert context.verify_mode == web_search.ssl.CERT_REQUIRED
    assert request.get_header("X-subscription-token") == "secret"
    assert parse_qs(urlparse(request.full_url).query) == {
        "q": ["agent harness"],
        "count": ["20"],
    }
    assert result == (
        "## Agentwerk\nhttps://example.com\nA minimal agentic loop."
    )


def test_brave_search_tool_reports_empty_results(monkeypatch):
    monkeypatch.setattr(
        web_search,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"web":{"results":[]}}'),
    )

    search = web_search.brave_search_tool("secret")

    assert search(query="nothing") == "No results found."


def test_coding_harness_requires_a_task_or_resume():
    assert coding_harness.parse_args(["fix", "the", "test"]) == "fix the test"
    assert coding_harness.parse_args(["--resume"]) is None

    with pytest.raises(ValueError, match="usage"):
        coding_harness.parse_args([])
    with pytest.raises(ValueError, match="usage"):
        coding_harness.parse_args(["--resume", "extra"])


def test_coding_harness_session_mode_prevents_overwrite_and_missing_resume(tmp_path):
    session = tmp_path / "session"
    with pytest.raises(RuntimeError, match="no session"):
        coding_harness.open_werk(session, None)

    werk = coding_harness.open_werk(session, "change")
    werk.add_task(Task("change", label=coding_harness.PLAN))

    with pytest.raises(RuntimeError, match="already exists"):
        coding_harness.open_werk(session, "other")
    resumed = coding_harness.open_werk(session, None)
    assert len(resumed.find_tasks(coding_harness.PLAN)) == 1


def test_coding_harness_stages_create_one_coder_and_return_its_result():
    werk = Werk()
    planner = werk.add_task(Task("fix addition", label=coding_harness.PLAN))
    assert coding_harness.next_stage(werk) == (coding_harness.PLAN, planner)

    plan = {
        "plan": ["Implement addition", "Run cargo test"],
        "files": ["src/lib.rs"],
        "checks": ["cargo test"],
    }
    werk.set_task_finished(planner, plan)
    assert coding_harness.next_stage(werk) == (
        "create_coder",
        ("fix addition", plan),
    )

    coder = werk.add_task(Task("implement the plan", label=coding_harness.CODING))
    assert coding_harness.next_stage(werk) == (coding_harness.CODING, coder)
    assert len(werk.find_tasks(coding_harness.CODING)) == 1

    result = {
        "summary": "Implemented addition.",
        "changed_files": ["src/lib.rs"],
        "checks": ["cargo test: passed"],
    }
    werk.set_task_finished(coder, result)
    assert coding_harness.next_stage(werk) == ("done", result)
    assert len(werk.find_tasks(coding_harness.CODING)) == 1


def test_coding_harness_condition_creates_one_coder_after_plan_finishes():
    werk = Werk()
    werk.add_condition(coding_harness.coder_condition())
    plan = werk.add_task(Task("fix addition", label=coding_harness.PLAN))

    werk.set_task_finished(plan, {"plan": ["Fix it"]})

    coders = werk.find_tasks(coding_harness.CODING)
    assert len(coders) == 1
    assert coders[0].get_task() == coding_harness.CODER_TASK


def test_coding_harness_rejects_duplicate_coder_tasks():
    werk = Werk()
    planner = werk.add_task(Task("fix addition", label=coding_harness.PLAN))
    werk.set_task_finished(planner, {"plan": ["Fix it"]})
    werk.add_task(Task("first", label=coding_harness.CODING))
    werk.add_task(Task("duplicate", label=coding_harness.CODING))

    with pytest.raises(RuntimeError, match="more than one coder task"):
        coding_harness.next_stage(werk)
