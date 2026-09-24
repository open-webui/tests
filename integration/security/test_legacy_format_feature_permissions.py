"""Regression: the legacy chat-features block trusted client-supplied feature flags.

open-webui 0.11.0 fix `897d69a35` (#26703): with `params.function_calling` set to `legacy`,
`process_chat_payload` ran the web search and image generation handlers whenever the request
carried `features.web_search` / `features.image_generation`, without the per-user permission
check the native format and the /images routes apply. A user denied both could still make the
server call the image engine (billable) and start a web search. The fix gates both legacy
branches on admin or `has_permission`.

The search provider and the image engine are one listener. The web search route re-checks the
permission on its own, so for search the leak shows as the "Searching the web" status stored on
the reply; for images the engine itself is called.

Twin of unit/security/test_legacy_format_feature_permissions.py.

Discriminates: passes on dev bbfa876af, fails with either gate removed from the legacy block
(the denied user's reply carries the web search statuses, or the image engine is called).
"""

from __future__ import annotations

import pytest

from harness.chat import ask
from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
IMAGES_CONFIG = ("/api/v1/images/config", "/api/v1/images/config/update")
PERMISSIONS = "/api/v1/users/default/permissions"

PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
SEARCH_HITS = {
    "results": [{"url": "https://example.com/page", "title": "Page", "content": "a hit"}]
}
BOTH_FEATURES = {"web_search": True, "image_generation": True}
GATED_TOOLS = {"search_web", "fetch_url", "generate_image"}


def _allow_features(admin, allowed: bool) -> None:
    with admin.client() as client:
        permissions = client.get(PERMISSIONS).json()
        permissions["features"].update(web_search=allowed, image_generation=allowed)
        client.post(PERMISSIONS, json=permissions).raise_for_status()


@pytest.fixture
def services(admin, listener, preserve):
    """Web search on a SearXNG listener, image generation on the same listener, both denied."""
    preserve("permissions", RETRIEVAL_CONFIG, IMAGES_CONFIG)
    listener.route("GET", "/search", json_answer(SEARCH_HITS))
    listener.route("POST", "/images/generations", json_answer({"data": [{"b64_json": PNG_BASE64}]}))
    with admin.client() as client:
        web = client.get(RETRIEVAL_CONFIG[0]).json()["web"]
        web.update(
            ENABLE_WEB_SEARCH=True,
            WEB_SEARCH_ENGINE="searxng",
            SEARXNG_QUERY_URL=f"{listener.base_url}/search",
            BYPASS_WEB_SEARCH_WEB_LOADER=True,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
        )
        client.post(RETRIEVAL_CONFIG[1], json={"web": web}).raise_for_status()
        images = client.get(IMAGES_CONFIG[0]).json()
        images.update(
            ENABLE_IMAGE_GENERATION=True,
            ENABLE_IMAGE_PROMPT_GENERATION=False,
            IMAGE_GENERATION_ENGINE="openai",
            IMAGES_OPENAI_API_BASE_URL=listener.base_url,
            IMAGES_OPENAI_API_KEY="sk-images",
        )
        client.post(IMAGES_CONFIG[1], json=images).raise_for_status()
    _allow_features(admin, False)
    return listener


def _chat(actor, function_calling: str) -> dict:
    with actor.client() as client:
        _, reply = ask(
            client,
            "draw a cat and look up cats",
            features=BOTH_FEATURES,
            params={"function_calling": function_calling},
        )
    return reply


def _calls(listener) -> dict[str, int]:
    return {
        "web_search": len(listener.requests_to("/search")),
        "image_generation": len(listener.requests_to("/images/generations")),
    }


def _status_actions(reply: dict) -> list[str]:
    return [
        status.get("action") or status.get("description", "")
        for status in reply.get("statusHistory", [])
    ]


def _offered_tools(upstream) -> set[str]:
    tools = upstream.chat_requests()[-1].get("tools") or []
    return {tool["function"]["name"] for tool in tools}


def test_a_denied_user_legacy_request_starts_neither_feature(services, make_user):
    reply = _chat(make_user(), "legacy")

    assert _calls(services)["image_generation"] == 0, (
        "a user denied image generation made the server call the image engine by sending the "
        "legacy request format (#26703)"
    )
    assert _status_actions(reply) == [], (
        f"a denied user's legacy request ran the feature handlers: {_status_actions(reply)} "
        "(#26703)"
    )
    assert _calls(services)["web_search"] == 0


def test_the_native_format_offers_a_denied_user_no_gated_tools(services, make_user, upstream):
    _chat(make_user(), "native")

    assert not _offered_tools(upstream) & GATED_TOOLS
    assert _calls(services) == {"web_search": 0, "image_generation": 0}


def test_an_admin_legacy_request_reaches_both_services(services, admin):
    reply = _chat(admin, "legacy")

    assert _calls(services) == {"web_search": 1, "image_generation": 1}
    assert "web_search" in _status_actions(reply)


def test_a_permitted_user_legacy_request_reaches_both_services(services, admin, make_user):
    _allow_features(admin, True)

    _chat(make_user(), "legacy")

    assert _calls(services) == {"web_search": 1, "image_generation": 1}


def test_the_native_format_never_runs_the_legacy_handlers(services, admin, upstream):
    reply = _chat(admin, "native")

    assert _calls(services) == {"web_search": 0, "image_generation": 0}
    assert _status_actions(reply) == []
    assert {"search_web", "generate_image"} <= _offered_tools(upstream)
