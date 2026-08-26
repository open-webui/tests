"""Regressions for the 0.11.0 provider connection/dispatch fixes.

Seven separate 0.10.2 defects on the path between a chat request and an
upstream OpenAI/Ollama connection:

1. `stream_options` (routers/openai.py). A non-streaming request still carried
   `stream_options: {'include_usage': True}`, which strict OpenAI-compatible
   providers reject with a 400. 0.11.0 pops it unless `stream` is truthy.
2. Shared pipe model tool calls (#26906, commit f8c0d2fdd, functions.py).
   `generate_function_chat_completion` resolved the base model by writing into
   the CALLER's `form_data`. The tool-call continuation re-submits that dict,
   so after the first tool call the model id was already the base pipe id and
   the response stopped silently. Fix copies the dict first (and replaces the
   mutable `models={}` default with None).
3. Ownerless tool or function (#26850, commit 48f78ca58). The Function/Tool
   read models declared `user_id: str` while the columns are nullable, so a
   NULL row raised ValidationError inside `get_functions()`/`get_tools()`,
   which run at startup, taking the whole app down. Fix: `str | None = None`.
4. Connection prefix stripping (commit ed663f1). Five copies of
   `model.replace(f'{prefix_id}.', '')` stripped the prefix ANYWHERE in the
   name. Replaced by `utils/model_ids.py::strip_provider_model_prefix`, which
   only strips a real leading prefix.
5. Newly pulled Ollama models (#27353, commit ed663f1). `get_ollama_url` raised
   MODEL_NOT_FOUND for a model missing from the cached `OLLAMA_MODELS`; 0.11.0
   clears the cache and refetches before giving up.
6. Connection changes take effect immediately. Both `/config/update` handlers
   now clear the model caches and state; pre-fix the stale list stayed live
   until a restart.
7. Disabled OpenAI API is enforced. 0.10.2 only guarded `get_models`, so a
   direct chat request still reached the provider and `get_all_models` left the
   previously cached models selectable.

Discriminates: passes on v0.11.0, fails on v0.10.2 (each narrow test observes
the pre-fix behaviour directly: the stale `stream_options` key, the mutated
caller dict, the ValidationError on a NULL user_id, the absent helper module,
the MODEL_NOT_FOUND raise, the untouched state dicts, the missing 503).
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

_STATE_TIMEOUT = 15  # backstop; every driven coroutine is fully stubbed at the I/O edge


def _request(**app_state) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**app_state)),
        state=SimpleNamespace(),
        cookies={},
    )


def _user(module, role: str = "admin"):
    return module.UserModel(
        id="u-provider-dispatch",
        email="dispatch@example.com",
        name="Dispatch",
        role=role,
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.status = 200
        self.headers = {"Content-Type": "application/json"}
        self.closed = False
        self._payload = payload

    async def json(self, loads=None):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)

    def close(self):
        self.closed = True


class _RecordingSession:
    """Stands in for the shared aiohttp session; records the serialized body."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse({"id": "resp", "choices": []})

    @property
    def sent_payload(self) -> dict:
        return json.loads(self.calls[-1]["data"])


class _ConfigStub:
    """Replaces `Config` in a router namespace. Never touches the real store."""

    def __init__(self, values: dict) -> None:
        self.values = values
        self.upserts: list[dict] = []

    async def get(self, key, default=None):
        return self.values.get(key, default)

    async def get_many(self, *keys):
        return {key: self.values[key] for key in keys if key in self.values}

    async def upsert(self, data, *args, **kwargs):
        self.upserts.append(data)


class _RecordingCache:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.clear_calls = 0

    async def clear(self, *args, **kwargs):
        self.clear_calls += 1
        return await self._inner.clear(*args, **kwargs)


def _openai_connection_config(prefix_id: str | None = None) -> dict:
    api_config = {"0": {"prefix_id": prefix_id}} if prefix_id is not None else {"0": {}}
    return {
        "openai.enable": True,
        "openai.api_base_urls": ["http://provider.invalid/v1"],
        "openai.api_keys": ["k"],
        "openai.api_configs": api_config,
    }


def _patch_openai_dispatch(monkeypatch, module, config_values: dict) -> _RecordingSession:
    """Stub the DB, config and network edges of `generate_chat_completion`."""
    session = _RecordingSession()
    config = _ConfigStub(config_values)
    monkeypatch.setattr(module, "Config", config)
    monkeypatch.setattr(
        module,
        "Models",
        SimpleNamespace(get_model_by_id=_none_coroutine),
    )
    monkeypatch.setattr(module, "get_session", _returning_coroutine(session))
    return session


def _none_coroutine(*args, **kwargs):
    async def _run():
        return None

    return _run()


def _returning_coroutine(value):
    async def _call(*args, **kwargs):
        return value

    return _call


async def _drive(coro):
    return await asyncio.wait_for(coro, timeout=_STATE_TIMEOUT)


# ---------------------------------------------------------------------------
# 1. stream_options on a non-streaming request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("form_stream", [None, False])
async def test_non_streaming_request_drops_stream_options(owui_module, monkeypatch, form_stream):
    """Narrow: a non-streaming body must not carry `stream_options` upstream."""
    module = owui_module("open_webui.routers.openai")
    session = _patch_openai_dispatch(monkeypatch, module, _openai_connection_config())

    form_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream_options": {"include_usage": True},
    }
    if form_stream is not None:
        form_data["stream"] = form_stream

    request = _request(OPENAI_MODELS={"gpt-4o": {"urlIdx": 0}})
    await _drive(module.generate_chat_completion(request, form_data, user=_user(module)))

    assert "stream_options" not in session.sent_payload


@pytest.mark.asyncio
async def test_streaming_request_keeps_stream_options(owui_module, monkeypatch):
    """Nearby: the usage opt-in still reaches a genuinely streaming request."""
    module = owui_module("open_webui.routers.openai")
    session = _patch_openai_dispatch(monkeypatch, module, _openai_connection_config())

    request = _request(OPENAI_MODELS={"gpt-4o": {"urlIdx": 0}})
    await _drive(
        module.generate_chat_completion(
            request,
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            user=_user(module),
        )
    )

    assert session.sent_payload["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# 2. pipe payload mutation (#26906)
# ---------------------------------------------------------------------------


def _pipe_model_info(base_model_id: str):
    return SimpleNamespace(
        id="workspace-pipe",
        base_model_id=base_model_id,
        user_id="owner",
        params=SimpleNamespace(model_dump=dict),
    )


def _patch_pipe_dispatch(monkeypatch, module, base_model_id: str) -> None:
    def pipe(body):
        return "pipe output"

    monkeypatch.setattr(
        module,
        "Models",
        SimpleNamespace(get_model_by_id=_returning_coroutine(_pipe_model_info(base_model_id))),
    )
    monkeypatch.setattr(
        module,
        "get_function_module_by_id",
        _returning_coroutine(SimpleNamespace(pipe=pipe)),
    )


def _pipe_request() -> SimpleNamespace:
    request = _request()
    request.cookies = {"oauth_session_id": "session-1"}
    request.app.state.oauth_manager = SimpleNamespace(
        get_oauth_token=_returning_coroutine({"access_token": "t"})
    )
    return request


@pytest.mark.asyncio
async def test_pipe_completion_does_not_rewrite_caller_model_id(owui_module, monkeypatch):
    """Narrow: the caller's dict keeps the workspace model id across the call."""
    module = owui_module("open_webui.functions")
    _patch_pipe_dispatch(monkeypatch, module, "mypipe.variant")

    form_data = {"model": "workspace-pipe", "messages": [{"role": "user", "content": "hi"}]}
    response = await _drive(
        module.generate_function_chat_completion(
            _pipe_request(), form_data, _user(owui_module("open_webui.models.users"))
        )
    )

    assert form_data["model"] == "workspace-pipe"
    assert response["model"] == "mypipe.variant"


@pytest.mark.asyncio
async def test_pipe_completion_does_not_strip_caller_metadata(owui_module, monkeypatch):
    """Narrow: `metadata` is popped from the copy, so the resubmit still has it."""
    module = owui_module("open_webui.functions")
    _patch_pipe_dispatch(monkeypatch, module, "mypipe.variant")

    form_data = {
        "model": "workspace-pipe",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"chat_id": "c1"},
    }
    await _drive(
        module.generate_function_chat_completion(
            _pipe_request(), form_data, _user(owui_module("open_webui.models.users"))
        )
    )

    assert form_data["metadata"] == {"chat_id": "c1"}


def test_pipe_completion_models_default_is_not_mutable(owui_module):
    """Broad: a shared mutable default leaks state between unrelated requests."""
    import inspect

    module = owui_module("open_webui.functions")
    signature = inspect.signature(module.generate_function_chat_completion)
    default = signature.parameters["models"].default

    assert default is None


# ---------------------------------------------------------------------------
# 3. ownerless tool or function (#26850)
# ---------------------------------------------------------------------------

_OWNERLESS_ROW = {
    "id": "orphan",
    "user_id": None,
    "name": "Orphan",
    "type": "pipe",
    "content": "def pipe(body):\n    return ''\n",
    "specs": [],
    "meta": {"description": "d"},
    "is_active": False,
    "is_global": False,
    "updated_at": 0,
    "created_at": 0,
}

_OWNED_ROW = {**_OWNERLESS_ROW, "user_id": "u1"}


def _read_models_with_user_id(module) -> list:
    from pydantic import BaseModel

    return [
        obj
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and getattr(obj, "__module__", None) == module.__name__
        and "user_id" in obj.model_fields
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("dotted", ["open_webui.models.functions", "open_webui.models.tools"])
async def test_read_models_accept_null_user_id(owui_module, dotted):
    """Broad: every read model mirroring the nullable owner column takes None.

    Pre-fix this is the ValidationError that `get_functions()`/`get_tools()`
    raised at startup, blocking boot for the whole instance.
    """
    module = owui_module(dotted)
    read_models = _read_models_with_user_id(module)

    assert read_models, f"no read model with a user_id field found in {dotted}"
    for model in read_models:
        model.model_validate(_OWNERLESS_ROW)


@pytest.mark.asyncio
@pytest.mark.parametrize("dotted", ["open_webui.models.functions", "open_webui.models.tools"])
async def test_read_models_still_accept_a_real_owner(owui_module, dotted):
    """Nearby: widening the annotation did not stop preserving a real owner."""
    module = owui_module(dotted)
    read_models = _read_models_with_user_id(module)

    assert read_models, f"no read model with a user_id field found in {dotted}"
    for model in read_models:
        assert model.model_validate(_OWNED_ROW).user_id == "u1"


# ---------------------------------------------------------------------------
# 4. connection prefix stripping (commit ed663f1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "prefix_id", "expected"),
    [
        ("acme.llama3", "acme", "llama3"),
        ("llama3.acme.tuned", "acme", "llama3.acme.tuned"),
        ("acme.acme.llama3", "acme", "acme.llama3"),
        ("acmellama3", "acme", "acmellama3"),
        ("acme.llama3", None, "acme.llama3"),
        ("acme.llama3", "", "acme.llama3"),
    ],
)
def test_strip_provider_model_prefix(model_id, prefix_id, expected):
    """Narrow: only a real leading `<prefix>.` is removed.

    Imported inside the test on purpose: the module does not exist pre-fix, and
    the resulting collection error is the evidence.
    """
    helper = importlib.import_module("open_webui.utils.model_ids")

    assert helper.strip_provider_model_prefix(model_id, prefix_id) == expected


@pytest.mark.parametrize("dotted", ["open_webui.routers.openai", "open_webui.routers.ollama"])
def test_no_inline_prefix_replace_left(owui_module, dotted):
    """Broad: no router still strips the prefix with a bare `str.replace`."""
    module = owui_module(dotted)
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "replace(f'{prefix_id}." not in source
    assert 'replace(f"{prefix_id}.' not in source


@pytest.mark.asyncio
async def test_prefixed_model_is_stripped_before_dispatch(owui_module, monkeypatch):
    """Nearby: the ordinary prefixed-connection case still reaches upstream bare."""
    module = owui_module("open_webui.routers.openai")
    session = _patch_openai_dispatch(monkeypatch, module, _openai_connection_config("acme"))

    request = _request(OPENAI_MODELS={"acme.gpt-4o": {"urlIdx": 0}})
    await _drive(
        module.generate_chat_completion(
            request,
            {"model": "acme.gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            user=_user(module),
        )
    )

    assert session.sent_payload["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# 5. newly pulled Ollama models (#27353)
# ---------------------------------------------------------------------------


def _patch_ollama_refresh(monkeypatch, module, populates: dict | None) -> _RecordingCache:
    cache = _RecordingCache(module.get_all_models.cache)

    async def fake_get_all_models(request, user=None):
        if populates is not None:
            request.app.state.OLLAMA_MODELS = populates
        return {"models": []}

    fake_get_all_models.cache = cache
    monkeypatch.setattr(module, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(
        module, "Config", _ConfigStub({"ollama.base_urls": ["http://ollama.invalid"]})
    )
    return cache


@pytest.mark.asyncio
async def test_get_ollama_url_refetches_for_a_newly_pulled_model(owui_module, monkeypatch):
    """Narrow: a model absent from the cached list is refetched, not rejected."""
    module = owui_module("open_webui.routers.ollama")
    cache = _patch_ollama_refresh(monkeypatch, module, {"fresh:latest": {"urls": [0]}})

    request = _request(OLLAMA_MODELS={"stale:latest": {"urls": [0]}})
    url, url_idx = await _drive(
        module.get_ollama_url(request, "fresh:latest", None, _user(module))
    )

    assert (url, url_idx) == ("http://ollama.invalid", 0)
    assert cache.clear_calls == 1


@pytest.mark.asyncio
async def test_get_ollama_url_uses_the_cached_entry_when_present(owui_module, monkeypatch):
    """Nearby: a known model resolves without a refetch."""
    module = owui_module("open_webui.routers.ollama")
    cache = _patch_ollama_refresh(monkeypatch, module, None)

    request = _request(OLLAMA_MODELS={"known:latest": {"urls": [0]}})
    url, url_idx = await _drive(
        module.get_ollama_url(request, "known:latest", None, _user(module))
    )

    assert (url, url_idx) == ("http://ollama.invalid", 0)
    assert cache.clear_calls == 0


@pytest.mark.asyncio
async def test_get_ollama_url_still_rejects_an_unknown_model(owui_module, monkeypatch):
    """Nearby: a refetch that finds nothing still raises MODEL_NOT_FOUND."""
    module = owui_module("open_webui.routers.ollama")
    _patch_ollama_refresh(monkeypatch, module, {})

    request = _request(OLLAMA_MODELS={"stale:latest": {"urls": [0]}})
    with pytest.raises(module.HTTPException) as excinfo:
        await _drive(module.get_ollama_url(request, "ghost:latest", None, _user(module)))

    assert excinfo.value.status_code == 400


# ---------------------------------------------------------------------------
# 6. connection changes take effect immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_config_update_clears_cached_models(owui_module, monkeypatch):
    """Narrow: saving a connection must invalidate the live OpenAI model list."""
    module = owui_module("open_webui.routers.openai")
    monkeypatch.setattr(module, "Config", _ConfigStub({}))
    monkeypatch.setattr(module, "publish_event", _returning_coroutine(None))

    request = _request(
        OPENAI_MODELS={"stale-model": {"urlIdx": 0}},
        BASE_MODELS=[{"id": "stale-model"}],
        MODELS={"stale-model": {}},
    )
    form = module.OpenAIConfigForm(
        ENABLE_OPENAI_API=True,
        OPENAI_API_BASE_URLS=["http://provider.invalid/v1"],
        OPENAI_API_KEYS=["k"],
        OPENAI_API_CONFIGS={},
    )
    await _drive(module.update_config(request, form, user=_user(module)))

    assert request.app.state.OPENAI_MODELS == {}
    assert request.app.state.BASE_MODELS == []
    assert request.app.state.MODELS == {}


@pytest.mark.asyncio
async def test_ollama_config_update_clears_cached_models(owui_module, monkeypatch):
    """Narrow: same for Ollama, whose handler also clears the aiocache entry."""
    module = owui_module("open_webui.routers.ollama")
    monkeypatch.setattr(module, "Config", _ConfigStub({}))
    monkeypatch.setattr(module, "publish_event", _returning_coroutine(None))
    cache = _RecordingCache(module.get_all_models.cache)
    monkeypatch.setattr(module.get_all_models, "cache", cache, raising=False)

    request = _request(
        OLLAMA_MODELS={"stale:latest": {"urls": [0]}},
        BASE_MODELS=[{"id": "stale:latest"}],
        MODELS={"stale:latest": {}},
    )
    form = module.OllamaConfigForm(
        ENABLE_OLLAMA_API=True,
        OLLAMA_BASE_URLS=["http://ollama.invalid"],
        OLLAMA_API_CONFIGS={},
    )
    await _drive(module.update_config(request, form, user=_user(module)))

    assert request.app.state.OLLAMA_MODELS == {}
    assert request.app.state.BASE_MODELS == []
    assert request.app.state.MODELS == {}
    assert cache.clear_calls == 1


# ---------------------------------------------------------------------------
# 7. disabled OpenAI connections are enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dotted", "enable_key", "models_attr", "model_id"),
    [
        ("open_webui.routers.openai", "openai.enable", "OPENAI_MODELS", "gpt-4o"),
        ("open_webui.routers.ollama", "ollama.enable", "OLLAMA_MODELS", "llama3:latest"),
    ],
)
async def test_disabled_provider_blocks_chat_completion(
    owui_module, monkeypatch, dotted, enable_key, models_attr, model_id
):
    """Narrow for OpenAI, nearby for Ollama: with the API off, nothing goes out.

    Ollama already carried this guard in 0.10.2; OpenAI did not, so a disabled
    connection was hidden in the UI while direct requests still went out.
    """
    module = owui_module(dotted)
    config_values = _openai_connection_config()
    config_values["ollama.base_urls"] = ["http://ollama.invalid"]
    config_values[enable_key] = False
    session = _patch_openai_dispatch(monkeypatch, module, config_values)

    request = _request(**{models_attr: {model_id: {"urlIdx": 0, "urls": [0]}}})
    with pytest.raises(module.HTTPException) as excinfo:
        await _drive(
            module.generate_chat_completion(
                request,
                {"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
                user=_user(module),
            )
        )

    assert excinfo.value.status_code == 503
    assert session.calls == []


@pytest.mark.asyncio
async def test_disabled_openai_clears_cached_model_state(owui_module, monkeypatch):
    """Narrow: `get_all_models` must drop the previously cached selectable models."""
    module = owui_module("open_webui.routers.openai")
    config_values = _openai_connection_config()
    config_values["openai.enable"] = False
    monkeypatch.setattr(module, "Config", _ConfigStub(config_values))
    await module.get_all_models.cache.clear()

    request = _request(OPENAI_MODELS={"stale-model": {"urlIdx": 0}})
    result = await _drive(module.get_all_models(request, user=_user(module)))

    assert result == {"data": []}
    assert request.app.state.OPENAI_MODELS == {}
