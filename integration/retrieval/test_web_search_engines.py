"""Journey: every web search engine the admin can pick hands the model its hits, filtered by domain.

With an engine saved and Web Search asked for in the chat, the model's `search_web` call reaches
that engine's API with the query and the admin's key, and the model is given back the links,
titles and snippets the engine listed. With `!elsewhere.example` in the domain filter list, the
hit from that domain is kept from the model, whichever engine found it. Every engine's API is
played by `harness/search_apis.py`, the fixed public hosts through its proxy, on an instance of
its own. The browser twin, e2e/retrieval/test_web_search_engines.py, sets each engine up in the
admin panel. DuckDuckGo, SearXNG and the external engine have modules of their own.

Not here: Azure AI Search, which the engine list of the admin panel does not offer, and Sougou,
whose Tencent Cloud SDK is not among the backend's requirements. Exa is sent the filter list and
filters on its side, and the docs say the list is not applied to Jina, so neither has a filter
case.

Red on dev ebc6add67, each naming the engine: Kagi applies the filter to its parsed results, which
have no `get`, so a filtered search fails; Perplexity Search is handed the filter list and never
applies it.

Discriminates: passes on dev ebc6add67 for every other engine; in a backend copy whose `search_web`
tool hands the model the engine's hits without their snippets, every engine case fails, and with
the domain filter skipped in `get_filtered_results` every filter case fails.
"""

from __future__ import annotations

import json

import pytest

from harness.actors import admin_of, create_user
from harness.search_apis import (
    API_KEY,
    ENGINES,
    Hit,
    carries,
    engine_settings,
    search_apis_env,
    serving_search_apis,
)
from harness.tool_calls import run_tool
from harness.web_retrieval import save_web_settings, web_settings_restored

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

WITH_WEB_SEARCH = {"features": {"web_search": True}}
HARBOUR = Hit("https://harbour.example/ferry", "Ferry times", "The ferry leaves at noon.")
ELSEWHERE = Hit("https://elsewhere.example/ferry", "Other ferries", "Another ferry at one.")
NO_FILTER_CASE = {"exa", "jina"}  # Exa filters on its side; the docs exempt Jina


@pytest.fixture(scope="module")
def apis():
    with serving_search_apis() as serving:
        yield serving


@pytest.fixture(scope="module")
def searching(instance_with, apis):
    return instance_with(search_apis_env(apis))


@pytest.fixture
def searching_admin(searching):
    with admin_of(searching).client() as client, web_settings_restored(client):
        yield client


def search(searching, query: str) -> list[dict] | dict:
    with create_user(searching).client() as client:
        found = run_tool(
            client, searching.upstream, "search_web", {"query": query}, **WITH_WEB_SEARCH
        )
    return json.loads(found)


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.name)
def test_the_model_gets_the_hits_the_engine_found(engine, searching, searching_admin, apis):
    engine.serve(apis, [HARBOUR, ELSEWHERE])
    save_web_settings(searching_admin, **engine_settings(engine, apis, result_count=2))

    found = search(searching, "ferry harbour")

    assert found == engine.found([HARBOUR, ELSEWHERE])
    [sent] = engine.searches(apis)[-1:]
    assert carries(sent, "ferry harbour"), f"{engine.name} was not asked the query: {sent}"
    if engine.name not in {"yacy", "openserp"}:  # neither takes a key
        assert carries(sent, API_KEY), f"{engine.name} was not sent the admin's key: {sent}"


@pytest.mark.parametrize(
    "engine",
    [engine for engine in ENGINES if engine.name not in NO_FILTER_CASE],
    ids=lambda engine: engine.name,
)
def test_a_blocked_domain_is_kept_from_the_model(engine, searching, searching_admin, apis):
    engine.serve(apis, [HARBOUR, ELSEWHERE])
    save_web_settings(
        searching_admin,
        **engine_settings(engine, apis, result_count=2),
        WEB_SEARCH_DOMAIN_FILTER_LIST=["!elsewhere.example"],
    )

    found = search(searching, "ferry filtered")

    assert isinstance(found, list), f"{engine.name} failed with the domain filter on: {found}"
    links = [hit["link"] for hit in found]
    assert ELSEWHERE.link not in links, f"{engine.name} ignored the domain filter list"
    assert HARBOUR.link in links
