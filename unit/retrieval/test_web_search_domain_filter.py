"""Regression: web-search result domain filtering matches on the URL HOST, not
netloc, so a blocked host with a :port (or userinfo) can't slip past the filter.

open-webui 0.10.2 fix `688bda09f` (#26 security batch): `get_filtered_results`
extracted the domain with `urlparse(url).netloc`, which includes the port and
userinfo. `is_host_allowed` matches on hostname *labels* (its own docstring says
"pass a parsed hostname, never a full URL"), so `blocked.example:8443` did NOT
match a `blocked.example` block entry and the result leaked through. The fix uses
`urlparse(url).hostname`.

`resolve_hostname` (which does DNS) is mocked to isolate host extraction and keep
this offline/deterministic — the netloc-vs-hostname behaviour is the whole point.

Discriminates: passes on v0.10.2 (hostname), fails on v0.10.1 and any build that
regressed to netloc.
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.regression


def _results(*urls):
    return [{"url": u, "title": "t", "snippet": "s"} for u in urls]


def test_blocked_host_with_port_is_filtered(web_search_main_module):
    """A blocked host carrying a :port must still be filtered out. On netloc
    matching, `blocked.example:8443` != `blocked.example` and it leaks."""
    mod = web_search_main_module
    with patch.object(mod, "resolve_hostname", return_value=([], [])):
        out = mod.get_filtered_results(
            _results("http://blocked.example:8443/path?q=1"), ["!blocked.example"]
        )
    assert out == [], (
        "a blocked host with a :port slipped the filter — the domain is being "
        "matched on netloc (host:port) instead of the parsed hostname"
    )


def test_blocked_host_bare_is_filtered(web_search_main_module):
    """Baseline (passes pre- and post-fix): a bare blocked host is filtered."""
    mod = web_search_main_module
    with patch.object(mod, "resolve_hostname", return_value=([], [])):
        out = mod.get_filtered_results(_results("https://blocked.example/x"), ["!blocked.example"])
    assert out == []


def test_allowed_host_with_port_passes(web_search_main_module):
    """Allow-list side of the same bug: an allowed host with a :port must still
    pass. On netloc matching `allowed.example:8443` != `allowed.example`, so it
    would be wrongly excluded."""
    mod = web_search_main_module
    with patch.object(mod, "resolve_hostname", return_value=([], [])):
        out = mod.get_filtered_results(
            _results("http://allowed.example:8443/"), ["allowed.example"]
        )
    assert len(out) == 1, "an allowed host with a :port was wrongly filtered out (netloc match)"


def test_userinfo_host_is_matched_against_real_host(web_search_main_module):
    """`http://allowed.example@blocked.example/` — the real host is
    blocked.example; netloc is `allowed.example@blocked.example`, which evades the
    block. Only hostname parsing catches it."""
    mod = web_search_main_module
    with patch.object(mod, "resolve_hostname", return_value=([], [])):
        out = mod.get_filtered_results(
            _results("http://allowed.example@blocked.example/"), ["!blocked.example"]
        )
    assert out == [], "a userinfo@host URL evaded the block filter (netloc vs hostname)"
