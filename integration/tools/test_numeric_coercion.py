"""Regression: a builtin tool called with a number sent as a string failed (#25641).

A model using native function calling may send `"count": "3"` where the tool declares an `int`.
Uncoerced, `min(count, max_count)` in `search_web` raised "'<' not supported between instances
of 'int' and 'str'" and `calculate_timestamp` failed on `days_ago + weeks_ago * 7`. Fix
`c4688b958` made the wrapper every builtin tool is called through (`coerce_kwargs` in
`get_async_tool_function_and_apply_extra_params`) convert a string to `int` for any `int` or
`Optional[int]` parameter before the tool runs.

The scripted model sends the numbers as strings; the search engine is a local listener.

Twin of unit/tools/test_numeric_coercion.py.

Discriminates: passes on dev bbfa876af, fails with the wrapper calling the tool without
`coerce_kwargs` (both tools return an error instead of a result); the nearby tests pass on both.
"""

from __future__ import annotations

import json

import pytest

from harness.tool_calls import run_tool
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DAY_SECONDS = 86_400
LINKS = [f"https://example.com/{index}" for index in range(5)]


@pytest.fixture
def web_search(admin, preserve, listener):
    """Web search on, answered by a listener with five results, five the configured count."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, LINKS))


def _call(actor, upstream, tool: str, **arguments):
    with actor.client() as client:
        output = run_tool(client, upstream, tool, arguments, features={"web_search": True})
    return json.loads(output)


def test_string_offsets_reach_calculate_timestamp_as_numbers(make_user, upstream):
    result = _call(make_user(), upstream, "calculate_timestamp", days_ago="3", weeks_ago="1")

    assert "error" not in result, f"string offsets broke calculate_timestamp (#25641): {result}"
    assert result["current_timestamp"] - result["calculated_timestamp"] == 10 * DAY_SECONDS


def test_a_string_count_limits_the_web_search(make_user, upstream, web_search):
    results = _call(make_user(), upstream, "search_web", query="portland weather", count="3")

    assert isinstance(results, list), f"a string count broke search_web (#25641): {results}"
    assert [result["link"] for result in results] == LINKS[:3]


def test_calculate_timestamp_without_offsets_is_now(make_user, upstream):
    result = _call(make_user(), upstream, "calculate_timestamp")

    assert result["current_timestamp"] == result["calculated_timestamp"]


def test_a_count_above_the_configured_one_is_capped(make_user, upstream, web_search):
    results = _call(make_user(), upstream, "search_web", query="portland weather", count=9)

    assert len(results) == len(LINKS)
