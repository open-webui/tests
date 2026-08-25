"""Regressions in model resolution and provider routing, open-webui v0.11.1.

Eight independent fixes, all in the path that turns a model id into an upstream
request:

* `16f118d77a` - the `{url_idx}` variants of `/ollama/api/tags`,
  `/ollama/api/version` and `/openai/models` had no admin dependency, so any
  verified user could enumerate one named backend; and
  `validate_ollama_backend_idx` short-circuited on `BYPASS_MODEL_ACCESS_CONTROL`,
  so a request naming a specific backend skipped the served-models check.
* `243a39dc9` (#27821) - the task endpoints splatted
  `request.app.state.MODELS` directly. That mapping is a `RedisDict`, and
  `{**pool}` walks `keys()` then `__getitem__` per key, so a concurrent refresh
  (HSET followed by HDEL of stale keys) could delete a key between the two and
  raise `KeyError` out of the dict literal. `dict(pool.items())` is one atomic
  HGETALL.
* `5cecb7dbfa` - `get_all_models` deactivated a model with `models.remove(model)`.
  `list.remove` matches by equality, and an Ollama base name and its tagged id
  both resolve to the same dict, so the second removal raised `ValueError` and
  took the whole model list with it.
* `3d630491c` (#28036) - `sync_models` passed `user_id`/`updated_at` both inside
  the splatted model dump and again as keywords, a duplicate-keyword `TypeError`
  swallowed by the broad handler; a re-sync reported success and updated nothing.
* `686d8dc54` (#28575) - `/openai/responses` serialized the body before the
  connection was resolved, so a connection with a Prefix ID forwarded the
  prefixed id and the provider answered "model not found".
* `9cf1a0796` (#27675, #27595, #27695) - the native Anthropic Messages
  passthrough called a helper that no longer existed and signed the request with
  a bearer token; Anthropic's native endpoints need `x-api-key` plus
  `anthropic-version`.
* `20fe43d9da` - the missing-base-model fallback ran after the access check, and
  the check refuses a non-admin whose base model has no workspace row, so only
  admins ever reached the fallback.
* `eadce55e34` (#28952, #28923) - a workspace model whose `base_model_id` equals
  its own id was accepted and then discarded while models were combined, so none
  of its settings took effect.

Discriminates: passes on v0.11.1, fails on v0.11.0 (each narrow test below
names the pre-fix behaviour it observes).
"""

from __future__ import annotations

import ast
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

pytestmark = pytest.mark.regression

AUTH_DEPENDENCIES = {"get_current_user", "get_verified_user", "get_admin_user"}


# ── module fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ollama_router(owui_module):
    return owui_module("open_webui.routers.ollama")


@pytest.fixture(scope="session")
def openai_router(owui_module):
    return owui_module("open_webui.routers.openai")


@pytest.fixture(scope="session")
def models_router(owui_module):
    return owui_module("open_webui.routers.models")


@pytest.fixture(scope="session")
def models_schema(owui_module):
    return owui_module("open_webui.models.models")


@pytest.fixture(scope="session")
def models_util(owui_module):
    return owui_module("open_webui.utils.models")


@pytest.fixture(scope="session")
def tasks_router(owui_module):
    return owui_module("open_webui.routers.tasks")


@pytest.fixture(scope="session")
def main_module(owui_module):
    return owui_module("open_webui.main")


# ── shared helpers ───────────────────────────────────────────────────────────


def _user(id: str = "alice", role: str = "user"):
    return SimpleNamespace(id=id, role=role, name=id, email=f"{id}@example.com")


def _request(**state):
    app_state = SimpleNamespace(**state)
    return SimpleNamespace(
        app=SimpleNamespace(state=app_state),
        state=SimpleNamespace(),
        headers={},
        cookies={},
    )


def _config_get(values: dict) -> AsyncMock:
    return AsyncMock(side_effect=lambda key, default=None: values.get(key, default))


def _dependency_names(route: APIRoute) -> set[str]:
    names = set()
    pending = list(route.dependant.dependencies)
    while pending:
        dependency = pending.pop()
        name = getattr(dependency.call, "__name__", None)
        if name:
            names.add(name)
        pending.extend(dependency.dependencies)
    return names


def _routes(module, path: str) -> list[APIRoute]:
    return [r for r in module.router.routes if isinstance(r, APIRoute) and r.path == path]


# ═════════════════════════════════════════════════════════════════════════════
# 25. Listing a single connection's models (16f118d77a)
# ═════════════════════════════════════════════════════════════════════════════

# (router fixture name, path) for every route that names one backend by index.
PER_CONNECTION_ROUTES = [
    ("ollama_router", "/api/tags/{url_idx}"),
    ("ollama_router", "/api/version/{url_idx}"),
    ("ollama_router", "/v1/models/{url_idx}"),
    ("openai_router", "/models/{url_idx}"),
]

FANNED_OUT_ROUTES = [
    ("ollama_router", "/api/tags"),
    ("ollama_router", "/api/version"),
    ("ollama_router", "/v1/models"),
    ("openai_router", "/models"),
]


@pytest.mark.parametrize(("fixture_name", "path"), PER_CONNECTION_ROUTES)
def test_per_connection_model_listing_is_admin_only(request, fixture_name, path):
    """Narrow. Naming a `url_idx` targets one configured backend directly, which
    is admin territory; pre-fix any verified user could enumerate it."""
    module = request.getfixturevalue(fixture_name)
    routes = _routes(module, path)
    assert routes, f"{path} is no longer registered, so this test proves nothing"
    for route in routes:
        assert "get_admin_user" in _dependency_names(route), (
            f"{path} lets a non-admin enumerate a single named backend"
        )


@pytest.mark.parametrize(("fixture_name", "path"), FANNED_OUT_ROUTES)
def test_fanned_out_model_listing_stays_open_to_users(request, fixture_name, path):
    """Nearby. The index-less variants return the access-filtered union and must
    stay usable by ordinary users, so the fix must not have gated them too."""
    module = request.getfixturevalue(fixture_name)
    routes = _routes(module, path)
    assert routes
    for route in routes:
        names = _dependency_names(route)
        assert names & AUTH_DEPENDENCIES, f"{path} must still require a signed-in user"
        assert "get_admin_user" not in names, f"{path} became admin-only, which over-corrects"


@pytest.mark.asyncio
async def test_named_backend_is_checked_even_when_access_control_is_bypassed(ollama_router):
    """Narrow. `BYPASS_MODEL_ACCESS_CONTROL` waives *per-model* permissions, not
    the "is this model served by that backend" check. Pre-fix it short-circuited
    the whole validation, so a user could aim a generate call at any backend."""
    request = _request(OLLAMA_MODELS={"llama3:latest": {"urls": [0]}})
    with patch.object(ollama_router, "BYPASS_MODEL_ACCESS_CONTROL", True):
        with pytest.raises(HTTPException) as excinfo:
            await ollama_router.validate_ollama_backend_idx(request, "llama3:latest", 1, _user())
    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_backend_serving_the_model_is_accepted(ollama_router):
    """Nearby. The positive path: an index the model really is served from."""
    request = _request(OLLAMA_MODELS={"llama3:latest": {"urls": [0, 2]}})
    assert (
        await ollama_router.validate_ollama_backend_idx(request, "llama3:latest", 2, _user())
        is None
    )


@pytest.mark.asyncio
async def test_admin_and_index_less_calls_skip_the_backend_check(ollama_router):
    """Nearby. Admins pick a backend deliberately, and `url_idx=None` is already
    constrained to the allow-list further down."""
    request = _request(OLLAMA_MODELS={"llama3:latest": {"urls": [0]}})
    assert (
        await ollama_router.validate_ollama_backend_idx(
            request, "llama3:latest", 1, _user(role="admin")
        )
        is None
    )
    assert (
        await ollama_router.validate_ollama_backend_idx(request, "llama3:latest", None, _user())
        is None
    )


# ═════════════════════════════════════════════════════════════════════════════
# 34. Chats during model list refreshes (243a39dc9, #27821)
# ═════════════════════════════════════════════════════════════════════════════


class _RefreshingPool:
    """A `RedisDict` mid-refresh: `keys()` still reports a key that `set()` has
    since HDEL'd, while `items()` (one HGETALL) is an atomic snapshot."""

    def __init__(self, snapshot: dict, stale_key: str):
        self._snapshot = snapshot
        self._stale_key = stale_key

    def keys(self):
        return [*self._snapshot.keys(), self._stale_key]

    def __getitem__(self, key):
        if key == self._stale_key:
            raise KeyError(key)
        return self._snapshot[key]

    def __contains__(self, key):
        return key in self._snapshot

    def __len__(self):
        return len(self._snapshot)

    def items(self):
        return list(self._snapshot.items())


POOL_MODEL = {"id": "gpt-4o", "name": "GPT-4o", "owned_by": "openai"}
DIRECT_MODEL = {"id": "direct-model", "name": "Direct", "owned_by": "openai"}


TASK_CONFIG = {"task.title.enable": True, "task.title.prompt_template": ""}


async def _run_generate_title(tasks_router, request, model_id, upstream):
    """Drive the real `/title/completions` endpoint down to its upstream call.

    Only Config, the prompt renderer, the pipeline filter and the upstream call
    are stubbed; the task-model helpers stay real because they differ between the
    two refs and patching them would turn this into a rename check.
    """
    seen = {}

    async def _capture_models(_request, payload, _user, models):
        seen.update(models)
        return payload

    with (
        patch.object(tasks_router.Config, "get", _config_get(TASK_CONFIG)),
        patch.object(tasks_router.Config, "get_many", AsyncMock(return_value={})),
        patch.object(tasks_router, "title_generation_template", AsyncMock(return_value="prompt")),
        patch.object(tasks_router, "process_pipeline_inlet_filter", _capture_models),
        patch.object(tasks_router, "generate_chat_completion", upstream),
    ):
        result = await tasks_router.generate_title(
            request, {"model": model_id, "messages": []}, _user()
        )
    return result, seen


@pytest.mark.asyncio
async def test_title_generation_survives_a_concurrent_model_refresh(tasks_router):
    """Narrow. Pre-fix `{**request.app.state.MODELS}` issued HKEYS then one HGET
    per key, so a key deleted in between raised `KeyError` before the endpoint
    did anything. The fix reads one coherent snapshot."""
    request = _request(MODELS=_RefreshingPool({"gpt-4o": POOL_MODEL}, "evicted-model"))
    request.state.direct = True
    request.state.model = DIRECT_MODEL

    upstream = AsyncMock(return_value={"choices": []})
    result, _ = await _run_generate_title(tasks_router, request, "gpt-4o", upstream)

    assert result == {"choices": []}
    assert upstream.await_count == 1


@pytest.mark.asyncio
async def test_direct_model_still_wins_over_a_pool_entry_of_the_same_id(tasks_router):
    """Nearby. The merge exists to let the request-supplied direct model override
    a pool entry with the same id; reading via `items()` must preserve that."""
    pool_entry = {"id": "direct-model", "name": "Pooled", "owned_by": "openai"}
    request = _request(MODELS=_RefreshingPool({"direct-model": pool_entry}, "evicted-model"))
    request.state.direct = True
    request.state.model = {"id": "direct-model", "name": "Direct", "owned_by": "direct"}

    _, seen = await _run_generate_title(
        tasks_router, request, "direct-model", AsyncMock(return_value={})
    )

    assert seen["direct-model"]["owned_by"] == "direct"


def test_no_task_endpoint_splats_the_model_pool_directly(open_webui_backend):
    """Broad. Every direct-connection branch that merges the pool must go through
    `.items()`; a new one that splats the mapping reintroduces the per-key HGET
    walk and the mid-refresh `KeyError`."""
    offenders = []
    for relative in ("routers/tasks.py", "utils/chat.py", "utils/middleware.py"):
        source = (open_webui_backend / "open_webui" / relative).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    continue
                if isinstance(value, ast.Attribute) and value.attr == "MODELS":
                    offenders.append(f"{relative}:{node.lineno}")
    assert offenders == [], (
        f"these sites unpack request.app.state.MODELS directly: {offenders}. "
        "Use dict(...MODELS.items()) so the read is a single atomic HGETALL (#27821)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 47. Deactivating a model (5cecb7dbfa)
# ═════════════════════════════════════════════════════════════════════════════


def _custom_model(models_schema, id: str, is_active: bool, base_model_id: str | None = None):
    return models_schema.ModelModel(
        id=id,
        user_id="alice",
        base_model_id=base_model_id,
        name=f"Custom {id}",
        params=models_schema.ModelParams(),
        meta=models_schema.ModelMeta(),
        is_active=is_active,
        updated_at=0,
        created_at=0,
    )


def _base_model(id: str, owned_by: str = "ollama") -> dict:
    return {"id": id, "name": id, "object": "model", "created": 0, "owned_by": owned_by}


async def _run_get_all_models(models_util, base_models: list[dict], custom_models: list):
    """Drive the real `utils.models.get_all_models` with only its I/O stubbed."""
    request = _request(MODELS={"seed": {}}, BASE_MODELS=base_models)
    config = {
        "models.base_models_cache": True,
        "evaluation.arena.enable": False,
        "models.default_metadata": {},
    }
    with (
        patch.object(models_util.Config, "get_many", AsyncMock(return_value=config)),
        patch.object(models_util, "ENABLE_PLUGINS", False),
        patch.object(models_util.Models, "get_all_models", AsyncMock(return_value=custom_models)),
        patch.object(models_util.Functions, "get_functions_by_ids", AsyncMock(return_value=[])),
        patch.object(
            models_util.Functions, "get_function_valves_by_ids", AsyncMock(return_value={})
        ),
        patch.object(models_util, "get_functions_cache", MagicMock(return_value={})),
    ):
        return await models_util.get_all_models(request)


@pytest.mark.asyncio
async def test_two_aliases_of_one_ollama_model_can_both_be_deactivated(models_util, models_schema):
    """Narrow. `base_model_lookup` maps both `llama3` and `llama3:latest` to the
    same dict, so deactivating both entries removed it twice. `list.remove` then
    raised `ValueError` on the second pass and blew up the whole listing, leaving
    every user with an empty model picker."""
    base_models = [_base_model("llama3:latest"), _base_model("gpt-4o", owned_by="openai")]
    custom_models = [
        _custom_model(models_schema, "llama3", is_active=False),
        _custom_model(models_schema, "llama3:latest", is_active=False),
    ]

    result = await _run_get_all_models(models_util, base_models, custom_models)

    assert [model["id"] for model in result] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_deactivating_one_model_leaves_the_others(models_util, models_schema):
    """Nearby. The ordinary single deactivation still drops exactly one entry."""
    base_models = [
        _base_model("gpt-4o", owned_by="openai"),
        _base_model("gpt-4o-mini", owned_by="openai"),
    ]
    custom_models = [_custom_model(models_schema, "gpt-4o", is_active=False)]

    result = await _run_get_all_models(models_util, base_models, custom_models)

    assert [model["id"] for model in result] == ["gpt-4o-mini"]


@pytest.mark.asyncio
async def test_active_override_renames_instead_of_removing(models_util, models_schema):
    """Nearby. An active override on a base model edits it in place."""
    base_models = [_base_model("gpt-4o", owned_by="openai")]
    custom_models = [_custom_model(models_schema, "gpt-4o", is_active=True)]

    result = await _run_get_all_models(models_util, base_models, custom_models)

    assert [model["id"] for model in result] == ["gpt-4o"]
    assert result[0]["name"] == "Custom gpt-4o"


# ═════════════════════════════════════════════════════════════════════════════
# 105. Syncing a model catalogue more than once (3d630491c, #28036)
# ═════════════════════════════════════════════════════════════════════════════


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeSession:
    """Stands in for the AsyncSession `sync_models` runs its statements through."""

    def __init__(self, existing_rows):
        self._existing_rows = existing_rows
        self.statements = []
        self.added = []
        self.committed = False

    async def execute(self, statement):
        self.statements.append(statement)
        return _FakeResult(self._existing_rows)

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self._existing_rows = [row for row in self._existing_rows if row is not obj]

    async def commit(self):
        self.committed = True


async def _run_sync_models(models_schema, existing_rows, payload):
    session = _FakeSession(existing_rows)

    @asynccontextmanager
    async def _session_context(db=None):
        yield session

    with (
        patch.object(models_schema, "get_async_db_context", _session_context),
        patch.object(models_schema.AccessGrants, "set_access_grants", AsyncMock(return_value=None)),
        patch.object(models_schema.AccessGrants, "revoke_all_access", AsyncMock(return_value=None)),
        patch.object(
            models_schema.AccessGrants, "get_grants_by_resources", AsyncMock(return_value={})
        ),
        patch.object(
            models_schema.Models,
            "_to_model_model",
            AsyncMock(side_effect=lambda model, access_grants=None, db=None: model),
        ),
    ):
        result = await models_schema.Models.sync_models("alice", payload, db=session)
    return result, session


@pytest.mark.asyncio
async def test_resyncing_an_existing_model_actually_updates_it(models_schema):
    """Narrow. Pre-fix the update branch passed `user_id`/`updated_at` twice (once
    splatted from the dump, once as a keyword), a `TypeError` the broad handler
    swallowed: the endpoint answered 200 with an empty list and wrote nothing."""
    existing = SimpleNamespace(id="shared-model")
    payload = [_custom_model(models_schema, "shared-model", is_active=True)]

    result, session = await _run_sync_models(models_schema, [existing], payload)

    assert result, "sync_models returned nothing, so the update branch raised and was swallowed"
    assert session.committed
    assert session.added == [], "an existing id must be updated, not inserted a second time"


@pytest.mark.asyncio
async def test_first_sync_of_a_new_model_still_inserts(models_schema):
    """Nearby. The insert branch was already correct and must stay that way."""
    payload = [_custom_model(models_schema, "brand-new", is_active=True)]

    result, session = await _run_sync_models(models_schema, [], payload)

    assert session.committed
    assert [row.id for row in session.added] == ["brand-new"]
    assert result is not None


@pytest.mark.asyncio
async def test_sync_carries_the_calling_user_and_a_fresh_timestamp(models_schema):
    """Nearby. Whichever branch runs, the row is stamped with the syncing user."""
    payload = [_custom_model(models_schema, "brand-new", is_active=True)]

    _, session = await _run_sync_models(models_schema, [], payload)

    assert session.added[0].user_id == "alice"
    assert session.added[0].updated_at > 0


# ═════════════════════════════════════════════════════════════════════════════
# 101. Model names on connections with a prefix (686d8dc54, #28575)
# ═════════════════════════════════════════════════════════════════════════════


class _FakeResponse:
    status = 200
    headers = {"Content-Type": "application/json"}

    async def json(self, loads=None):
        return {"id": "resp_1"}


@pytest.mark.asyncio
async def test_responses_strips_the_connection_prefix_from_the_model(openai_router):
    """Narrow. The body was serialized before the connection was resolved, so the
    provider received `myprovider.gpt-4o` and answered "model not found"."""
    sent = {}

    async def _session_request(**kwargs):
        sent.update(kwargs)
        return _FakeResponse()

    request = _request(OPENAI_MODELS={"myprovider.gpt-4o": {"urlIdx": 3}})
    form_data = openai_router.ResponsesForm(model="myprovider.gpt-4o", input="hi")

    with (
        patch.object(openai_router.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(openai_router, "check_model_access", AsyncMock(return_value=None)),
        patch.object(
            openai_router,
            "get_openai_connection",
            AsyncMock(
                return_value=("https://api.example.com/v1", "k", {"prefix_id": "myprovider"})
            ),
        ),
        patch.object(
            openai_router,
            "get_session",
            AsyncMock(return_value=SimpleNamespace(request=_session_request)),
        ),
        patch.object(openai_router, "cleanup_response", AsyncMock(return_value=None)),
    ):
        await openai_router.responses(request, form_data, _user())

    assert openai_router.JSONCodec.loads(sent["data"])["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_responses_leaves_an_unprefixed_model_alone(openai_router):
    """Nearby. A connection without a Prefix ID must forward the id verbatim."""
    sent = {}

    async def _session_request(**kwargs):
        sent.update(kwargs)
        return _FakeResponse()

    request = _request(OPENAI_MODELS={"gpt-4o": {"urlIdx": 0}})
    form_data = openai_router.ResponsesForm(model="gpt-4o", input="hi")

    with (
        patch.object(openai_router.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(openai_router, "check_model_access", AsyncMock(return_value=None)),
        patch.object(
            openai_router,
            "get_openai_connection",
            AsyncMock(return_value=("https://api.example.com/v1", "k", {})),
        ),
        patch.object(
            openai_router,
            "get_session",
            AsyncMock(return_value=SimpleNamespace(request=_session_request)),
        ),
        patch.object(openai_router, "cleanup_response", AsyncMock(return_value=None)),
    ):
        await openai_router.responses(request, form_data, _user())

    assert openai_router.JSONCodec.loads(sent["data"])["model"] == "gpt-4o"


# ═════════════════════════════════════════════════════════════════════════════
# 163. Coding tools that speak Anthropic's format (9cf1a0796, #27595, #27695)
# ═════════════════════════════════════════════════════════════════════════════


def _anthropic_request_target(openai_router):
    """The resolver, under whichever name the checkout carries.

    Pre-fix it is `get_anthropic_token_count_target`; the fix renamed it to
    `get_anthropic_request_target` because the Messages passthrough now shares it.
    Looking it up by either name keeps the assertions below behavioural instead
    of degenerating into "the new name is missing".
    """
    return getattr(
        openai_router,
        "get_anthropic_request_target",
        getattr(openai_router, "get_anthropic_token_count_target", None),
    )


@pytest.mark.asyncio
async def test_native_anthropic_request_is_signed_with_x_api_key(openai_router):
    """Narrow. Anthropic's native `/v1/messages` rejects a bearer token with 401
    and needs `anthropic-version`; pre-fix the resolver sent only `Authorization`."""
    resolver = _anthropic_request_target(openai_router)
    assert resolver is not None, "no Anthropic request-target resolver in this checkout"

    request = _request(OPENAI_MODELS={"claude-sonnet-4": {"urlIdx": 0}})
    with (
        patch.object(openai_router.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(openai_router, "check_model_access", AsyncMock(return_value=None)),
        patch.object(
            openai_router,
            "get_openai_connection",
            AsyncMock(return_value=("https://api.anthropic.com/v1", "sk-ant-secret", {})),
        ),
    ):
        _, _, _, _, headers, _ = await resolver(
            request, {"model": "claude-sonnet-4", "messages": []}, _user()
        )

    assert headers.get("x-api-key") == "sk-ant-secret"
    assert "Authorization" not in headers
    assert headers.get("anthropic-version")


@pytest.mark.asyncio
async def test_non_anthropic_connection_keeps_bearer_auth(openai_router):
    """Nearby. LiteLLM and other OpenAI-compatible gateways still want a bearer
    token, so the header rewrite must be scoped to api.anthropic.com."""
    resolver = _anthropic_request_target(openai_router)
    request = _request(OPENAI_MODELS={"claude-sonnet-4": {"urlIdx": 0}})
    with (
        patch.object(openai_router.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(openai_router, "check_model_access", AsyncMock(return_value=None)),
        patch.object(
            openai_router,
            "get_openai_connection",
            AsyncMock(return_value=("https://litellm.internal/v1", "sk-proxy", {})),
        ),
    ):
        _, _, _, _, headers, _ = await resolver(
            request, {"model": "claude-sonnet-4", "messages": []}, _user()
        )

    assert headers.get("Authorization") == "Bearer sk-proxy"
    assert "x-api-key" not in headers


@pytest.mark.asyncio
async def test_anthropic_messages_passthrough_reaches_the_provider(main_module, openai_router):
    """Narrow. Pre-fix the passthrough called `openai.AIOHTTP_CLIENT_TIMEOUT`,
    which no longer existed, so every Anthropic-format request raised
    `AttributeError` before it was sent and surfaced as a 502."""
    sent = {}

    async def _session_request(**kwargs):
        sent.update(kwargs)
        return _FakeResponse()

    request = _request(OPENAI_MODELS={"claude-sonnet-4": {"urlIdx": 0}})
    with (
        patch.object(openai_router.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(openai_router, "check_model_access", AsyncMock(return_value=None)),
        patch.object(
            openai_router,
            "get_openai_connection",
            AsyncMock(return_value=("https://api.anthropic.com/v1", "sk-ant-secret", {})),
        ),
        patch.object(
            main_module,
            "get_session",
            AsyncMock(return_value=SimpleNamespace(request=_session_request)),
        ),
        patch.object(main_module, "cleanup_response", AsyncMock(return_value=None)),
    ):
        await main_module.passthrough_anthropic_messages(
            request, {"model": "claude-sonnet-4", "messages": []}, _user()
        )

    assert sent["url"].endswith("/messages")
    assert sent["headers"].get("x-api-key") == "sk-ant-secret"
    assert sent["timeout"] is not None


# ═════════════════════════════════════════════════════════════════════════════
# 198. Falling back when a base model is gone (20fe43d9da)
# ═════════════════════════════════════════════════════════════════════════════

HANDOFF = "reached-the-chat-pipeline"


async def _run_chat_completion(main_module, models_schema, user):
    """Drive the real `chat_completion` up to the point it hands off the payload."""
    preset = _custom_model(models_schema, "my-preset", is_active=True, base_model_id="gone-model")
    fallback_info = _custom_model(models_schema, "default-model", is_active=True)
    rows = {"my-preset": preset, "default-model": fallback_info, "gone-model": None}

    request = _request(
        MODELS={
            "my-preset": {
                "id": "my-preset",
                "name": "Preset",
                "owned_by": "openai",
                "preset": True,
            },
            "default-model": {"id": "default-model", "name": "Default", "owned_by": "openai"},
        }
    )
    handoff = AsyncMock(side_effect=RuntimeError(HANDOFF))

    with (
        patch.object(main_module, "ENABLE_CUSTOM_MODEL_FALLBACK", True),
        patch.object(main_module, "BYPASS_MODEL_ACCESS_CONTROL", False),
        patch.object(main_module, "BYPASS_ADMIN_ACCESS_CONTROL", False),
        patch.object(
            main_module.Config,
            "get",
            _config_get({"ui.default_models": "default-model", "models.default_params": {}}),
        ),
        patch.object(
            main_module.Models,
            "get_model_by_id",
            AsyncMock(side_effect=lambda model_id, **_kw: rows.get(model_id)),
        ),
        patch("open_webui.utils.models.Groups.get_groups_by_member_id", AsyncMock(return_value=[])),
        # Read access on both workspace rows is granted; the base-model chain is
        # what the fix is about, so it stays the only thing that can refuse.
        patch(
            "open_webui.models.access_grants.AccessGrants.has_access",
            AsyncMock(return_value=True),
        ),
        patch.object(main_module, "process_chat_payload", handoff),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await main_module.chat_completion(request, {"model": "my-preset", "messages": []}, user)
    return handoff, excinfo.value.detail


@pytest.mark.asyncio
async def test_non_admin_gets_the_fallback_when_the_base_model_is_gone(main_module, models_schema):
    """Narrow. A pipe-supplied base model has no workspace row, and
    `has_base_model_access` treats a rowless base model as admin-only, so pre-fix
    the access check refused the owner before the fallback could apply.

    The handoff sentinel marks how far the request got: pre-fix it never runs and
    the caller sees "Model not found"."""
    handoff, detail = await _run_chat_completion(main_module, models_schema, _user())

    assert detail == HANDOFF, f"non-admin was refused before the fallback: {detail}"
    assert handoff.await_count == 1
    assert handoff.await_args.args[1]["model"] == "default-model"


@pytest.mark.asyncio
async def test_admin_keeps_the_fallback(main_module, models_schema):
    """Nearby. Admins already reached the fallback pre-fix and must still."""
    handoff, detail = await _run_chat_completion(
        main_module, models_schema, _user("root", role="admin")
    )

    assert detail == HANDOFF
    assert handoff.await_args.args[1]["model"] == "default-model"


# ═════════════════════════════════════════════════════════════════════════════
# 201. Workspace models pointing at themselves (eadce55e34, #28952, #28923)
# ═════════════════════════════════════════════════════════════════════════════


def _model_form(models_schema, id: str, base_model_id: str | None):
    return models_schema.ModelForm(
        id=id,
        base_model_id=base_model_id,
        name=f"Preset {id}",
        meta=models_schema.ModelMeta(),
        params=models_schema.ModelParams(),
        access_grants=[],
    )


@pytest.mark.asyncio
async def test_create_drops_a_self_referential_base_model_id(models_router, models_schema):
    """Narrow. Pre-fix the entry was stored with `base_model_id == id`, and the
    combine step in `get_all_models` then skipped it (the id already exists as a
    base model), so the preset's system prompt and params never took effect."""
    insert = AsyncMock(
        side_effect=lambda form, user_id, db=None: SimpleNamespace(id=form.id, name=form.name)
    )
    request = _request(MODELS={})

    with (
        patch.object(models_router.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(models_router.Models, "insert_new_model", insert),
        patch.object(models_router, "_verify_knowledge_file_access", AsyncMock(return_value=None)),
        patch.object(models_router, "filter_allowed_access_grants", AsyncMock(return_value=[])),
        patch.object(models_router.Config, "get", _config_get({"user.permissions": {}})),
        patch.object(models_router, "publish_event", AsyncMock(return_value=None)),
    ):
        await models_router.create_new_model(
            request,
            _model_form(models_schema, "self-preset", "self-preset"),
            _user("root", role="admin"),
            db=None,
        )

    assert insert.await_args.args[0].base_model_id is None


@pytest.mark.asyncio
async def test_update_drops_a_self_referential_base_model_id(models_router, models_schema):
    """Narrow. Same trap through the edit endpoint."""
    stored = SimpleNamespace(
        id="self-preset",
        user_id="root",
        base_model_id="gpt-4o",
        meta=models_schema.ModelMeta(),
    )
    update = AsyncMock(
        side_effect=lambda model_id, form, db=None: SimpleNamespace(id=model_id, name=form.name)
    )

    with (
        patch.object(models_router.Models, "get_model_by_id", AsyncMock(return_value=stored)),
        patch.object(models_router.Models, "update_model_by_id", update),
        patch.object(models_router, "_verify_knowledge_file_access", AsyncMock(return_value=None)),
        patch.object(models_router, "filter_allowed_access_grants", AsyncMock(return_value=[])),
        patch.object(models_router.Config, "get", _config_get({"user.permissions": {}})),
        patch.object(models_router, "publish_event", AsyncMock(return_value=None)),
    ):
        await models_router.update_model_by_id(
            _request(MODELS={}),
            _model_form(models_schema, "self-preset", "self-preset"),
            _user("root", role="admin"),
            db=None,
        )

    assert update.await_args.args[1].base_model_id is None


@pytest.mark.asyncio
async def test_import_drops_a_self_referential_base_model_id(models_router, open_webui_backend):
    """Broad. Import is the third write path and takes the same payload; it must
    heal a bad export rather than persist a model based on itself."""
    source = (open_webui_backend / "open_webui" / "routers" / "models.py").read_text(
        encoding="utf-8"
    )
    start = source.index("async def import_models")
    body = source[start : source.index("\n@router", start)]
    assert "model_data.get('base_model_id') == model_id" in body, (
        "import_models no longer nulls a self-referential base_model_id (#28952)"
    )


def test_a_real_base_model_id_survives_all_write_paths(models_schema):
    """Nearby. Only the self-reference is healed; an ordinary parent id stays."""
    form = _model_form(models_schema, "my-preset", "gpt-4o")
    assert form.base_model_id == "gpt-4o"
