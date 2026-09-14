"""Brave Search tool used by the README research workflow."""

import json
import ssl
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agentwerk import tool


_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_MAX_RESULTS = 20
_SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")
_DESCRIPTION = """Searches the web and returns titles, URLs, and descriptions.

Use the results to choose pages to open with `fetch`. Result descriptions are
leads, not sources.

Usage:
- `query`: the exact search query
- `count`: results from 1 to 20. Defaults to 5

# Instructions
- Write a focused query for the assigned research topic
- Open a result before citing it, because descriptions can omit page context

Example usage:
brave_search({"query": "software API maintenance", "count": 5})"""


def _ssl_context() -> ssl.SSLContext:
    """Use the system CA bundle when Python's OpenSSL bundle is unavailable."""

    default_ca_file = ssl.get_default_verify_paths().cafile
    if default_ca_file is not None or not _SYSTEM_CA_FILE.is_file():
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=_SYSTEM_CA_FILE)


def brave_search_tool(api_key: str):
    """Create a Brave Search tool authenticated with ``api_key``."""

    context = _ssl_context()

    @tool(
        name="brave_search",
        description=_DESCRIPTION,
        concurrent=True,
        timeout=60,
    )
    def brave_search(query: str, count: int = 5) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        count = max(1, min(count, _MAX_RESULTS))
        url = f"{_ENDPOINT}?{urlencode({'q': query, 'count': count})}"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        request = Request(url, headers=headers)
        with urlopen(request, timeout=60, context=context) as response:
            body = json.load(response)

        results = body.get("web", {}).get("results", [])
        if not results:
            return "No results found."

        rendered_results = (
            f"## {result.get('title', '')}\n"
            f"{result.get('url', '')}\n"
            f"{result.get('description', '')}"
            for result in results
        )
        return "\n\n".join(rendered_results)

    return brave_search
