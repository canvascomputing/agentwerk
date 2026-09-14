"""Test the helpers shipped with the Python examples."""

import io
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.parse import parse_qs, urlparse


example_path = Path(__file__).parents[1] / "examples" / "web_search.py"
example_spec = spec_from_file_location("web_search_example", example_path)
assert example_spec is not None and example_spec.loader is not None
web_search = module_from_spec(example_spec)
example_spec.loader.exec_module(web_search)


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
