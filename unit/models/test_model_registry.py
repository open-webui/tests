"""Model routing regressions from open-webui v0.11.1 that no HTTP test can reach.

* `16f118d77a`: `validate_ollama_backend_idx` returned early under `BYPASS_MODEL_ACCESS_CONTROL`,
  so a request naming one Ollama backend skipped the check that the model is served there.
  Reaching it over HTTP needs an instance booted with that env var and two Ollama backends.
* `243a39dc9` (#27821): the direct-connection branches merged `request.app.state.MODELS` with
  `{**pool}`. On a `RedisDict` that is HKEYS then one HGET per key, so a refresh deleting a key
  in between raised `KeyError`; `dict(pool.items())` is one HGETALL. The race cannot be timed
  over HTTP, so an `ast` audit holds every merge to the snapshot form.
* `9cf1a0796` (#27595): native Anthropic requests were signed with a bearer token; the resolver
  now sends `x-api-key` and `anthropic-version` to api.anthropic.com, a host no test can serve.

The other fixes this file covered (the listing gates, the alias deactivation, re-sync, the
Responses prefix, the Anthropic passthrough, the fallback and self-based models) are pinned by
integration/models/test_model_registry.py and integration/security/test_connection_listing_roles.py.

Discriminates: passes on dev `bbfa876af`; fails with the bypass early return restored, with one
task endpoint splatting the pool again, and with the api.anthropic.com header rewrite removed
(one mutation each).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, create_autospec, patch

import pytest

pytest.importorskip("fastapi")
redis = pytest.importorskip("redis")

from fastapi import FastAPI, HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def ollama_router(owui_module):
    return owui_module("open_webui.routers.ollama")


@pytest.fixture(scope="module")
def openai_router(owui_module):
    return owui_module("open_webui.routers.openai")


def _request(**app_state) -> Request:
    app = FastAPI()
    for name, value in app_state.items():
        setattr(app.state, name, value)
    return Request({"type": "http", "app": app, "method": "POST", "path": "/", "headers": []})


def _user(owui_module, role: str):
    users = owui_module("open_webui.models.users")
    return users.UserModel(
        id=f"{role}-1",
        email=f"{role}@example.com",
        name=role,
        role=role,
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


# --- 16f118d77a: a named Ollama backend is checked even when access control is bypassed ---

SERVED_FROM_FIRST = {"llama3:latest": {"urls": [0]}}


@pytest.mark.asyncio
async def test_a_named_backend_is_checked_when_access_control_is_bypassed(
    ollama_router, owui_module
):
    request = _request(OLLAMA_MODELS=SERVED_FROM_FIRST)
    with patch.object(ollama_router, "BYPASS_MODEL_ACCESS_CONTROL", True):
        with pytest.raises(HTTPException) as refused:
            await ollama_router.validate_ollama_backend_idx(
                request=request, model="llama3:latest", url_idx=1, user=_user(owui_module, "user")
            )

    assert refused.value.status_code == 403, (
        "BYPASS_MODEL_ACCESS_CONTROL waives per-model permissions, not the check that the named "
        "backend serves the model; a user could aim a request at any backend"
    )


@pytest.mark.asyncio
async def test_a_backend_that_serves_the_model_is_accepted(ollama_router, owui_module):
    request = _request(OLLAMA_MODELS={"llama3:latest": {"urls": [0, 2]}})

    await ollama_router.validate_ollama_backend_idx(
        request=request, model="llama3:latest", url_idx=2, user=_user(owui_module, "user")
    )


@pytest.mark.asyncio
async def test_admins_and_index_less_calls_skip_the_backend_check(ollama_router, owui_module):
    request = _request(OLLAMA_MODELS=SERVED_FROM_FIRST)

    await ollama_router.validate_ollama_backend_idx(
        request=request, model="llama3:latest", url_idx=1, user=_user(owui_module, "admin")
    )
    await ollama_router.validate_ollama_backend_idx(
        request=request, model="llama3:latest", url_idx=None, user=_user(owui_module, "user")
    )


# --- 243a39dc9: the model pool is read as one snapshot ------------------------------------


def _model_pool_splats(tree: ast.AST) -> list[int]:
    """Lines of dict literals that unpack an attribute named MODELS directly."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if key is None and isinstance(value, ast.Attribute) and value.attr == "MODELS"
    ]


def _model_pool_snapshots(tree: ast.AST) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "items"
        and getattr(node.func.value, "attr", None) == "MODELS"
    )


def test_no_merge_splats_the_model_pool(open_webui_backend: Path):
    package = open_webui_backend / "open_webui"
    offenders, snapshots = [], 0
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{path.relative_to(package)}:{line}" for line in _model_pool_splats(tree)]
        snapshots += _model_pool_snapshots(tree)

    assert snapshots, "no code reads MODELS.items() any more: retarget this audit at the merges"
    assert offenders == [], (
        f"these merges unpack the model pool directly: {offenders}. On a RedisDict that is one "
        "HGET per key, and a refresh deleting a key in between raises KeyError; use "
        "dict(MODELS.items()), a single HGETALL (#27821)"
    )


def test_a_pool_snapshot_survives_a_key_evicted_mid_read(owui_module):
    """Why the audit asks for items(): the splat walks keys the refresh already dropped."""
    socket_utils = owui_module("open_webui.socket.utils")
    client = create_autospec(redis.Redis, instance=True)
    client.hkeys.return_value = ["gpt-4o", "evicted-model"]
    client.hget.side_effect = lambda _name, key: '{"id": "gpt-4o"}' if key == "gpt-4o" else None
    client.hgetall.return_value = {"gpt-4o": '{"id": "gpt-4o"}'}
    with patch.object(socket_utils, "get_redis_connection", return_value=client):
        pool = socket_utils.RedisDict("models", redis_url="redis://unused")

    with pytest.raises(KeyError):
        {**pool}
    assert dict(pool.items()) == {"gpt-4o": {"id": "gpt-4o"}}


# --- 9cf1a0796: api.anthropic.com gets x-api-key, not a bearer token ----------------------


def _connection(url: str, key: str):
    return AsyncMock(return_value=(url, key, {}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected", "absent"),
    [
        ("https://api.anthropic.com/v1", {"x-api-key": "sk-ant"}, "Authorization"),
        ("https://litellm.internal/v1", {"Authorization": "Bearer sk-ant"}, "x-api-key"),
    ],
)
async def test_native_anthropic_requests_carry_the_key_the_host_expects(
    openai_router, owui_module, url, expected, absent
):
    request = _request(OPENAI_MODELS={"claude-sonnet-4": {"urlIdx": 0}})
    # the connection list lives in the config store, so its read is the boundary
    with patch.object(openai_router, "get_openai_connection", _connection(url, "sk-ant")):
        *_, headers, _cookies = await openai_router.get_anthropic_request_target(
            request=request,
            form_data={"model": "claude-sonnet-4", "messages": []},
            user=_user(owui_module, "admin"),
        )

    assert expected.items() <= headers.items(), headers
    assert absent not in headers, (
        "Anthropic's native endpoints reject a bearer token and other gateways need one (#27595)"
    )
    assert ("anthropic-version" in headers) == ("api.anthropic.com" in url)
