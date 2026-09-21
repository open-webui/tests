"""Regression: the per-connection model listings must check the caller's role.

open-webui 0.11.4 PR #29619 (`4b1019009`): the endpoints that list the models or
version of a single Ollama or OpenAI connection (`/api/tags/{idx}`,
`/api/version/{idx}`, `/models/{idx}`, `/v1/models/{idx}`) were reachable by any
signed-in account, exposing every connection's configured model ids and
upstream version. The handlers now check the role themselves: a specific
connection index is admin-only, while the aggregate listing stays available to
everyone.

The role check lives inside each handler, so the shipped handlers are driven
directly with the request boundary faked.

Discriminates: passes on dev 344ea5306; on the pre-fix ref a non-admin receives
the connection's raw list instead of a 401.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression


def _user(role="user"):
    return SimpleNamespace(id="u1", role=role, email="u@example.com", name="u")


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))


def _call(handler, url_idx, user):
    kwargs = {"request": _request(), "user": user}
    params = inspect.signature(handler).parameters
    if "url_idx" in params:
        kwargs["url_idx"] = url_idx
    return handler(**kwargs)


HANDLERS = [
    ("openai", "open_webui.routers.openai", "get_models"),
    ("ollama tags", "open_webui.routers.ollama", "get_ollama_tags"),
    ("ollama version", "open_webui.routers.ollama", "get_ollama_versions"),
    ("ollama v1", "open_webui.routers.ollama", "get_openai_models"),
]


# ── narrow: a non-admin is refused on every per-connection listing ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize("label,module_name,handler_name", HANDLERS)
async def test_a_non_admin_is_refused_on_the_single_connection_listing(
    owui_module, label, module_name, handler_name
):
    router = owui_module(module_name)
    handler = getattr(router, handler_name)

    with pytest.raises(HTTPException) as excinfo:
        await _call(handler, 0, _user("user"))

    assert excinfo.value.status_code in (401, 403), (
        f"a non-admin received the raw model list from the {label} endpoint, "
        "exposing every connection's models and ids (#29619)"
    )


# ── nearby: the aggregate listings stay open, and admins keep single ones ──


@pytest.mark.asyncio
@pytest.mark.parametrize("label,module_name,handler_name", HANDLERS)
async def test_an_admin_still_reaches_the_single_connection_listing(
    owui_module, label, module_name, handler_name, monkeypatch
):
    router = owui_module(module_name)
    handler = getattr(router, handler_name)
    monkeypatch.setattr(
        router, "get_all_models", AsyncMock(return_value={"data": []})
    )

    try:
        await _call(handler, 0, _user("admin"))
    except HTTPException as e:
        assert e.status_code not in (401, 403), (
            f"an admin is refused on the {label} endpoint, locking admins out of "
            "their own connections (#29619)"
        )


@pytest.mark.asyncio
async def test_the_aggregate_listing_stays_open_to_users(owui_module):
    openai_router = owui_module("open_webui.routers.openai")
    with patch.object(openai_router, "get_all_models", AsyncMock(return_value={"data": []})):
        result = await _call(openai_router.get_models, None, _user("user"))

    assert not isinstance(result, Exception)
