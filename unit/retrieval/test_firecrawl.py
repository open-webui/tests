"""Firecrawl checks that no route reaches (`open_webui/retrieval/web/firecrawl.py`).

- #23966 Bug 1 (NameError): in v0.9.1 `SafeFireCrawlLoader.lazy_load` called `requests.post`
  without importing `requests`, and `continue_on_failure` swallowed the NameError into an empty
  result. The Firecrawl path itself is pinned over HTTP; this audit holds every module under
  `retrieval/web/` to it.
- The timeout setting reaches Firecrawl only as an integer, so the parsing of strings, invalid
  and non-positive values is checked on the helpers directly.
- A Firecrawl that refuses connections is retried with real backoff sleeps, too slow for HTTP.

Request shapes, both search answer shapes (#23966 Bug 2), the header without a key, retries and
`Retry-After` moved to integration/retrieval/test_firecrawl.py.

Discriminates: passes on dev bbfa876af; deleting `import requests` from firecrawl.py fails the
audit, and dropping the connection-error retry fails the retry count.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import requests


def module_level_names(tree: ast.Module) -> set[str]:
    """Names bound by imports at module scope, including inside top-level `try` and `if`."""
    names: set[str] = set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            pending += node.body + node.orelse + getattr(node, "finalbody", [])
            pending += [
                child for handler in getattr(node, "handlers", []) for child in handler.body
            ]
    return names


def calls_requests(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "requests"
        for node in ast.walk(tree)
    )


@pytest.mark.regression
def test_every_web_module_calling_requests_imports_it(open_webui_backend):
    web_dir = open_webui_backend / "open_webui" / "retrieval" / "web"
    modules = sorted(web_dir.rglob("*.py"))
    assert (web_dir / "firecrawl.py") in modules, f"retarget: firecrawl.py is gone from {web_dir}"

    offenders = []
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        if calls_requests(tree) and "requests" not in module_level_names(tree):
            offenders.append(module.relative_to(open_webui_backend).as_posix())

    assert not offenders, f"#23966: these call requests.<x>() without importing it: {offenders}"


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        (30, 30.0),
        ("30", 30.0),
        ("45.5", 45.5),
        (None, None),
        ("", None),
        ("soon", None),
        (-5, None),
    ],
)
def test_timeout_setting_parses_to_positive_seconds(firecrawl_module, value, seconds):
    assert firecrawl_module.get_firecrawl_timeout_seconds(timeout=value) == seconds


def test_client_timeout_outlasts_the_scrape_timeout(firecrawl_module):
    assert firecrawl_module.get_firecrawl_client_timeout_seconds(timeout=30) == 40.0
    assert (
        firecrawl_module.get_firecrawl_client_timeout_seconds(timeout=None, fallback=120) == 130.0
    )


def test_a_refused_connection_is_retried_then_raised(firecrawl_module):
    refused = requests.ConnectionError("connection refused")

    with (
        patch.object(requests, "request", side_effect=refused) as sent,
        patch("time.sleep") as slept,
    ):
        with pytest.raises(requests.ConnectionError):
            firecrawl_module.request_firecrawl_json(
                method="POST",
                url="http://127.0.0.1:9/v2/scrape",
                headers={},
                json={"url": "http://127.0.0.1:9/page"},
                timeout=5,
            )

    assert sent.call_count == 3
    assert slept.call_count == 2
