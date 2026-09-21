"""Regression: the memories switch and permission were not honoured behind the chat path.

Two 0.11.4 fixes share the `memories.enable` switch and the `features.memories`
permission:

* `702da1e47` (PR #30228, issue #30227): with Memories switched off in the admin
  settings, `process_chat_payload` still folded stored memories into the system
  prompt (it checked only the narrower `memories.system_context.enable`) and
  `get_builtin_tools` still offered the eight memory tools (it checked the
  category, the model capability and the user permission, but no global switch).
* `e8c26f839` (PR #30309): `review_memory_after_turn` drafted new memories from
  the conversation on every interval turn even with memories switched off or the
  account barred from them, because it read neither the switch nor the
  permission; the review spent a task-model call and only then failed at the
  write, which the router's permission check refuses.

Discriminates: passes on 344ea5306 (0.11.4), fails on 702da1e47^ / e8c26f839^
(the context is injected, the tools are offered, and the review task runs, all
behind the switch).
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

MEMORY_TOOLS = {
    "search_memories",
    "list_memory_paths",
    "read_memory_path",
    "list_memories",
    "update_memory",
    "add_memory",
    "replace_memory_content",
    "delete_memory",
}

# The config reads get_builtin_tools makes, all on so a refusal is only ever
# caused by the memories switch or the permission.
TOOL_SWITCHES = {
    "web.search.enable": True,
    "image_generation.enable": True,
    "images.edit.enable": True,
    "code_interpreter.enable": True,
    "notes.enable": True,
    "channels.enable": True,
    "automations.enable": True,
    "calendar.enable": True,
    "ui.enable_user_webhooks": True,
    "subagents.enable": True,
    "subagents.background_enabled": True,
}


@pytest.fixture(scope="module")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="module")
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


@pytest.fixture(scope="module")
def memory_module(owui_module):
    return owui_module("open_webui.utils.memory")


@pytest.fixture(scope="module")
def groups_model(owui_module):
    return owui_module("open_webui.models.groups").Groups


@pytest.fixture(scope="module")
def user_model(owui_module):
    return owui_module("open_webui.models.users").UserModel


@pytest.fixture(scope="module")
def config_store(owui_module):
    owui_module("open_webui.config")
    return owui_module("open_webui.models.config").Config


def _chat_user(user_model, role="user"):
    return user_model(
        id="alice",
        name="Alice",
        email="alice@example.com",
        role=role,
        profile_image_url="",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


def _request():
    app = SimpleNamespace(state=SimpleNamespace(MODELS={"m": {"id": "m"}}))
    return SimpleNamespace(
        app=app,
        cookies={},
        headers={},
        state=SimpleNamespace(direct=False, internal=False),
    )


@contextmanager
def _config(config_store, **overrides):
    """Serve the config reads the production code makes, without writing the store."""
    values = {
        "memories.enable": True,
        "memories.system_context.enable": True,
        "memories.background_review.enable": True,
        "memories.review_interval_turns": 2,
        "user.permissions": {"features": {"memories": True}},
        **TOOL_SWITCHES,
        **overrides,
    }
    real_get, real_get_many = config_store.get, config_store.get_many

    async def get(key, default=None):
        return values[key] if key in values else await real_get(key, default)

    async def get_many(*keys):
        return {
            **await real_get_many(*keys),
            **{key: values[key] for key in keys if key in values},
        }

    with patch.object(config_store, "get", get), patch.object(config_store, "get_many", get_many):
        yield


# --- narrow: #30228, system prompt injection -----------------------------------


@pytest.mark.asyncio
async def test_switched_off_memories_are_not_injected_into_the_prompt(
    middleware_module, groups_model, user_model, config_store
):
    handler = AsyncMock(side_effect=lambda request, form_data, user, model: form_data)
    form_data = {
        "model": "m",
        "messages": [{"role": "user", "content": "what do you remember"}],
        "features": {"memory": True},
    }
    metadata = {
        "chat_id": "",
        "params": {"function_calling": "legacy"},
        "features": {"memory": True},
    }

    with (
        _config(config_store, **{"memories.enable": False}),
        patch.object(middleware_module, "add_memory_context", handler),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        await middleware_module.process_chat_payload(
            _request(), form_data, _chat_user(user_model), metadata, {"id": "m"}
        )

    assert handler.called is False, (
        "with memories switched off, stored memories were still folded into the "
        "system prompt behind the switch the person can no longer manage (#30228)"
    )


@pytest.mark.asyncio
async def test_switched_off_memories_are_not_offered_as_tools(
    tools_module, groups_model, user_model, config_store
):
    """The same switch must stop the memory tools, or the model can still read and delete them."""
    user = _chat_user(user_model)
    extra_params = {"__user__": user.model_dump(), "__metadata__": {"chat_id": ""}}

    with (
        _config(config_store, **{"memories.enable": False}),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        tools = await tools_module.get_builtin_tools(
            _request(), extra_params, {"memory": True}, {"id": "m"}
        )

    offered = set(tools) & MEMORY_TOOLS
    assert not offered, (
        f"with memories switched off the model was still offered {sorted(offered)}, so it "
        "could list and delete memories the person can no longer manage (#30228)"
    )


# --- narrow: #30309, the background review -------------------------------------


def _review_inputs(role="user"):
    return {
        "request": _request(),
        "user": SimpleNamespace(id="alice", role=role),
        "model": {"id": "m"},
        "metadata": {"features": {"memory": True}},
        "form_data": {},
        "assistant_message": {"role": "assistant", "content": "some answer"},
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ],
    }


@pytest.mark.asyncio
async def test_background_review_does_not_run_with_memories_switched_off(
    memory_module, config_store
):
    review = AsyncMock()
    with (
        _config(config_store, **{"memories.enable": False}),
        patch.object(memory_module, "_review_memory", review),
    ):
        await memory_module.review_memory_after_turn(**_review_inputs())

    assert review.called is False, (
        "with memories switched off the background review still drafted memory "
        "operations, spending a task-model call per interval turn on work the "
        "router then refuses at the write (#30309)"
    )


@pytest.mark.asyncio
async def test_background_review_does_not_run_for_a_user_barred_from_memories(
    memory_module, config_store
):
    review = AsyncMock()
    with (
        _config(config_store, **{"user.permissions": {"features": {"memories": False}}}),
        patch.object(memory_module, "_review_memory", review),
    ):
        await memory_module.review_memory_after_turn(**_review_inputs())

    assert review.called is False, (
        "the background review trusted the client-supplied memory feature flag over "
        "the features.memories permission, so a barred account still spent the task-model "
        "call before failing at the write (#30309)"
    )


@pytest.mark.asyncio
async def test_admin_barred_by_default_permissions_still_gets_the_review(
    memory_module, config_store
):
    """Nearby: admins bypass the permission lookup, so denying every feature must not stop them."""
    review = AsyncMock()
    with (
        _config(config_store, **{"user.permissions": {"features": {"memories": False}}}),
        patch.object(memory_module, "_review_memory", review),
    ):
        await memory_module.review_memory_after_turn(**_review_inputs(role="admin"))

    assert review.called is True


# --- broad: the two surfaces must agree with the router's own gate ---------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "memories_enabled,permitted",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
async def test_switch_and_permission_gate_every_chat_surface_the_same_way(
    tools_module,
    groups_model,
    user_model,
    config_store,
    memories_enabled,
    permitted,
):
    """Whatever the router refuses, the tool surface must refuse too, or the model reads
    and writes memories through the tool that the person cannot manage anymore."""
    user = _chat_user(user_model)
    extra_params = {"__user__": user.model_dump(), "__metadata__": {"chat_id": ""}}

    with (
        _config(
            config_store,
            **{
                "memories.enable": memories_enabled,
                "user.permissions": {"features": {"memories": permitted}},
            }
        ),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        tools = await tools_module.get_builtin_tools(
            _request(), extra_params, {"memory": True}, {"id": "m"}
        )

    assert bool(set(tools) & MEMORY_TOOLS) == (memories_enabled and permitted), (
        f"the memory tools were offered (switch={memories_enabled}, permitted={permitted}) "
        "where the memories router would have refused the request (#30228)"
    )


# --- nearby: the positive path and the earlier gates stay ------------------------


@pytest.mark.asyncio
async def test_permitted_memories_still_reach_the_prompt_and_the_tools(
    middleware_module, tools_module, groups_model, user_model, config_store
):
    """The fix must not lock out an ordinary permitted user."""
    handler = AsyncMock(side_effect=lambda request, form_data, user, model: form_data)
    form_data = {
        "model": "m",
        "messages": [{"role": "user", "content": "what do you remember"}],
        "features": {"memory": True},
    }
    metadata = {
        "chat_id": "",
        "params": {"function_calling": "legacy"},
        "features": {"memory": True},
    }
    user = _chat_user(user_model)
    extra_params = {"__user__": user.model_dump(), "__metadata__": {"chat_id": ""}}

    with (
        _config(config_store),
        patch.object(middleware_module, "add_memory_context", handler),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        await middleware_module.process_chat_payload(
            _request(), form_data, user, metadata, {"id": "m"}
        )
        tools = await tools_module.get_builtin_tools(
            _request(), extra_params, {"memory": True}, {"id": "m"}
        )

    assert handler.called is True
    assert MEMORY_TOOLS <= set(tools)


@pytest.mark.asyncio
async def test_narrow_system_context_switch_still_wins_over_the_permission(
    middleware_module, groups_model, user_model, config_store
):
    """Nearby: the pre-existing narrower switch keeps its veto on the injection path."""
    handler = AsyncMock(side_effect=lambda request, form_data, user, model: form_data)
    form_data = {
        "model": "m",
        "messages": [{"role": "user", "content": "what do you remember"}],
        "features": {"memory": True},
    }
    metadata = {
        "chat_id": "",
        "params": {"function_calling": "legacy"},
        "features": {"memory": True},
    }

    with (
        _config(config_store, **{"memories.system_context.enable": False}),
        patch.object(middleware_module, "add_memory_context", handler),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        await middleware_module.process_chat_payload(
            _request(), form_data, _chat_user(user_model), metadata, {"id": "m"}
        )

    assert handler.called is False


@pytest.mark.asyncio
async def test_review_still_runs_for_a_permitted_user(
    memory_module, config_store
):
    """Nearby: the interval and feature gates stay, and a permitted user's review still runs."""
    review = AsyncMock()
    with (
        _config(config_store),
        patch.object(memory_module, "_review_memory", review),
    ):
        await memory_module.review_memory_after_turn(**_review_inputs())

    assert review.called is True


@pytest.mark.asyncio
async def test_review_stays_off_between_interval_turns(
    memory_module, config_store
):
    """Nearby: the interval gate is unchanged, so an off-interval turn schedules nothing."""
    review = AsyncMock()
    inputs = _review_inputs()
    inputs["messages"] = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    with (
        _config(config_store),
        patch.object(memory_module, "_review_memory", review),
    ):
        await memory_module.review_memory_after_turn(**inputs)

    assert review.called is False
