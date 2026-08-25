"""Regression: permission and feature switches that were advisory on one code path.

open-webui 0.11.1 gathers several fixes where a switch an admin had set was
honoured on the main path and ignored on a second one:

* `934802e18` (#27668, `utils/middleware.py`): the legacy features block ran
  `add_memory_context` on any request whose client-supplied `features.memory`
  was set, with no `features.memories` permission check. The native
  function-calling path already gated it.
* `e17dfae72` / `646a568ae` (#27759, #27669, `utils/middleware.py`):
  `chat_image_generation_handler` was reachable without passing through the
  `/images` routes, so it generated images while `image_generation.enable` was
  off, and the legacy web-search branch ignored `web.search.enable`.
* `80d2f4154` (#27716) and `8fc5ffe26` (#27609, `routers/users.py`):
  `SharingPermissions.public_tools` / `public_notes` defaulted to True, so any
  unrelated permission save granted everyone public tool and note sharing, and
  `open_chats` had no field at all, so the toggle was dropped on save.
* `d9e23b90c` (#28366, `main.py`): the message-sending path persisted a new chat
  with a caller-supplied `folder_id` without checking write access on it.
* `7d392bedc` (#28631, `main.py`): the `channel:` branch only checked that the
  target message belonged to the channel, so channel write access let a model
  edit any member's message.
* `ad8c79f686` (#27766, `routers/users.py`): the settings save serialized the
  form with a plain `model_dump()`, so a client saving one setting wrote a fully
  defaulted document over everything the user had not touched.

Discriminates: passes on v0.11.1, fails on v0.11.0 (the memory context is added
for a denied user, image generation runs with the feature switched off, the
sharing defaults grant public sharing on upgrade, `open_chats` vanishes on save,
a chat is filed into a folder the caller cannot write, another member's channel
message is accepted for edit, and untouched settings are overwritten).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def users_router_module(owui_module):
    return owui_module("open_webui.routers.users")


@pytest.fixture(scope="session")
def config_store(owui_module):
    owui_module("open_webui.config")
    return owui_module("open_webui.models.config").Config


def _sharing_defaults() -> dict:
    from open_webui.config import DEFAULT_USER_PERMISSIONS

    return DEFAULT_USER_PERMISSIONS["sharing"]


# --- 26 / 102: what an admin saved is what the schema persists ----------------


def test_public_tool_sharing_is_not_granted_by_an_unrelated_save(users_router_module):
    """An upgraded instance has no `public_tools` key stored; re-serializing must
    not invent one that is on."""
    stored = {key: value for key, value in _sharing_defaults().items() if key != "public_tools"}

    saved = users_router_module.SharingPermissions(**stored).model_dump()

    assert saved["public_tools"] is False, (
        "saving any unrelated permission granted every user public tool sharing, "
        "because SharingPermissions.public_tools defaulted to True (#27716)"
    )


def test_public_note_sharing_is_not_granted_by_an_unrelated_save(users_router_module):
    stored = {key: value for key, value in _sharing_defaults().items() if key != "public_notes"}

    saved = users_router_module.SharingPermissions(**stored).model_dump()

    assert saved["public_notes"] is False, (
        "saving any unrelated permission granted every user public note sharing, "
        "because SharingPermissions.public_notes defaulted to True (#27716)"
    )


@pytest.mark.asyncio
async def test_open_sharing_reaches_the_permission_layer(
    users_router_module, access_control_module
):
    """`open_chats` had no schema field, so an admin enabling open sharing had the
    flag dropped and the toggle came back off."""
    saved = users_router_module.SharingPermissions(
        **{**_sharing_defaults(), "open_chats": True}
    ).model_dump()

    with patch.object(
        access_control_module.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])
    ):
        allowed = await access_control_module.has_permission(
            "alice", "sharing.open_chats", {"sharing": saved}
        )

    assert allowed is True, (
        "open sharing reads as disabled for a user even though the admin enabled it, "
        "because SharingPermissions dropped open_chats on save (#27609)"
    )


def test_disabled_sharing_toggles_stay_disabled(users_router_module):
    """Nearby: the schema must not flip anything on that was explicitly saved off."""
    stored = {key: False for key in _sharing_defaults()}

    saved = users_router_module.SharingPermissions(**stored).model_dump()

    assert not any(saved.values()), (
        f"enabled without being saved: {sorted(k for k, v in saved.items() if v)}"
    )


def test_enabled_sharing_toggles_stay_enabled(users_router_module):
    stored = {key: True for key in _sharing_defaults()}

    saved = users_router_module.SharingPermissions(**stored).model_dump()

    assert all(saved.values()), (
        f"dropped despite being saved: {sorted(k for k, v in saved.items() if not v)}"
    )


# --- 111: a partial settings save must not overwrite untouched settings -------


def _settings_user(role: str = "user"):
    return types.SimpleNamespace(id="alice", role=role, email="alice@example.com", name="Alice")


async def _saved_settings_document(users_module, form_data, user):
    """Drive the real settings endpoint and report the document it hands the store."""
    store = AsyncMock(return_value=types.SimpleNamespace(id=user.id, settings={}))
    with (
        patch.object(users_module.Users, "update_user_settings_by_id", store),
        patch.object(users_module, "publish_event", AsyncMock()),
        patch.object(users_module.Config, "get", AsyncMock(return_value=None)),
    ):
        await users_module.update_user_settings_by_session_user(
            types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace())),
            form_data,
            user,
            None,
        )
    return store.await_args.args[1]


@pytest.mark.asyncio
async def test_saving_one_setting_does_not_write_defaults_over_the_others(users_router_module):
    form_data = users_router_module.UserSettings(**{"notifications": {"webhook_url": ""}})

    document = await _saved_settings_document(
        users_router_module, form_data, _settings_user("admin")
    )

    assert "ui" not in document, (
        "a client that saved one setting sent a fully defaulted document, so every "
        "field the user had not touched was overwritten with its default (#27766)"
    )


@pytest.mark.asyncio
async def test_settings_the_client_did_send_are_still_written(users_router_module):
    """Nearby: exclude_unset must not swallow the fields the client actually set."""
    form_data = users_router_module.UserSettings(**{"ui": {"theme": "dark"}})

    document = await _saved_settings_document(
        users_router_module, form_data, _settings_user("admin")
    )

    assert document["ui"] == {"theme": "dark"}


@pytest.mark.asyncio
async def test_a_setting_explicitly_sent_as_its_default_is_still_written(users_router_module):
    """A value equal to the field default is set, not unset, so it must persist."""
    form_data = users_router_module.UserSettings(**{"ui": {}})

    document = await _saved_settings_document(
        users_router_module, form_data, _settings_user("admin")
    )

    assert document["ui"] == {}


# --- 24 / 73: the legacy features block must honour permissions and switches --

FEATURE_SWITCHES = {
    "memories.system_context.enable": True,
    "web.search.enable": True,
    "image_generation.enable": True,
    "image_generation.prompt.enable": False,
    "images.edit.enable": True,
}


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def groups_model(owui_module):
    return owui_module("open_webui.models.groups").Groups


@pytest.fixture(scope="session")
def user_model(owui_module):
    return owui_module("open_webui.models.users").UserModel


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


def _chat_request():
    app = types.SimpleNamespace(state=types.SimpleNamespace(MODELS={"m": {"id": "m"}}))
    return types.SimpleNamespace(
        app=app,
        cookies={},
        headers={},
        state=types.SimpleNamespace(direct=False, internal=False),
    )


@contextmanager
def _config(config_store, permitted_features=(), **switch_overrides):
    """Serve the config reads the production code makes instead of writing the
    real, process-wide config store."""
    overrides = {
        **FEATURE_SWITCHES,
        **switch_overrides,
        "user.permissions": {
            "features": {
                feature: feature in permitted_features
                for feature in ("memories", "web_search", "image_generation")
            }
        },
    }
    real_get, real_get_many = config_store.get, config_store.get_many

    async def get(key, default=None):
        return overrides[key] if key in overrides else await real_get(key, default)

    async def get_many(*keys):
        return {
            **await real_get_many(*keys),
            **{key: overrides[key] for key in keys if key in overrides},
        }

    with patch.object(config_store, "get", get), patch.object(config_store, "get_many", get_many):
        yield


async def _legacy_payload_activations(middleware_module, groups_model, user, features):
    """Run the real payload pipeline in the legacy format and report which
    privileged handlers it dispatched to."""
    handlers = {
        "memory": AsyncMock(side_effect=lambda request, form_data, user, model: form_data),
        "web_search": AsyncMock(
            side_effect=lambda request, form_data, extra_params, user: form_data
        ),
    }
    form_data = {
        "model": "m",
        "messages": [{"role": "user", "content": "what do you remember, and look it up"}],
        "features": dict(features),
    }
    metadata = {"chat_id": "", "params": {"function_calling": "legacy"}, "features": dict(features)}
    with (
        patch.object(middleware_module, "add_memory_context", handlers["memory"]),
        patch.object(middleware_module, "chat_web_search_handler", handlers["web_search"]),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        await middleware_module.process_chat_payload(
            _chat_request(), form_data, user, metadata, {"id": "m"}
        )
    return {name: handler.called for name, handler in handlers.items()}


@pytest.mark.asyncio
async def test_legacy_memory_context_denied_without_the_memories_permission(
    middleware_module, groups_model, user_model, config_store
):
    with _config(config_store):
        activations = await _legacy_payload_activations(
            middleware_module, groups_model, _chat_user(user_model), {"memory": True}
        )

    assert activations["memory"] is False, (
        "a user denied features.memories had their stored memories injected into the "
        "prompt anyway, just by setting the client-supplied memory flag (#27668)"
    )


@pytest.mark.asyncio
async def test_legacy_web_search_denied_while_web_search_is_switched_off(
    middleware_module, groups_model, user_model, config_store
):
    with _config(config_store, ("web_search",), **{"web.search.enable": False}):
        activations = await _legacy_payload_activations(
            middleware_module, groups_model, _chat_user(user_model), {"web_search": True}
        )

    assert activations["web_search"] is False, (
        "a session opened before the admin switched web search off still ran server-side "
        "searches, because the legacy branch never read web.search.enable (#27669)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "admin"])
async def test_permitted_memory_context_still_runs(
    middleware_module, groups_model, user_model, config_store, role
):
    """Nearby: the gate must not lock out a permitted user or an admin."""
    with _config(config_store, ("memories",)):
        activations = await _legacy_payload_activations(
            middleware_module, groups_model, _chat_user(user_model, role), {"memory": True}
        )

    assert activations["memory"] is True


@pytest.mark.asyncio
async def test_memory_context_stays_off_when_the_system_context_is_disabled(
    middleware_module, groups_model, user_model, config_store
):
    """Nearby: the pre-existing global switch still wins over the permission."""
    with _config(config_store, ("memories",), **{"memories.system_context.enable": False}):
        activations = await _legacy_payload_activations(
            middleware_module, groups_model, _chat_user(user_model), {"memory": True}
        )

    assert activations["memory"] is False


@pytest.mark.asyncio
async def test_permitted_web_search_still_runs_while_the_switch_is_on(
    middleware_module, groups_model, user_model, config_store
):
    with _config(config_store, ("web_search",)):
        activations = await _legacy_payload_activations(
            middleware_module, groups_model, _chat_user(user_model), {"web_search": True}
        )

    assert activations["web_search"] is True


# --- 73: the image handler enforces the switches the /images routes enforce ----


async def _run_image_handler(middleware_module, user, chat_id, files=None):
    """Drive the real handler and report what it did: the image calls it made, the
    statuses it emitted, and the system context it appended."""
    calls = {
        "generations": AsyncMock(return_value=[{"url": "/cache/image/generated.png"}]),
        "edits": AsyncMock(return_value=[{"url": "/cache/image/edited.png"}]),
    }
    statuses = []

    async def emitter(event):
        if event.get("type") == "status":
            statuses.append(event["data"].get("description"))

    message = {"role": "user", "content": "draw me a cat", "files": files or []}
    form_data = {"model": "m", "messages": [message]}
    extra_params = {
        "__metadata__": {"chat_id": chat_id, "message_id": "assistant-1"},
        "__event_emitter__": emitter,
    }
    stored_chat = types.SimpleNamespace(
        chat={"history": {"currentId": "u1", "messages": {"u1": {"id": "u1", **message}}}}
    )
    with (
        patch.object(middleware_module, "image_generations", calls["generations"]),
        patch.object(middleware_module, "image_edits", calls["edits"]),
        patch.object(
            middleware_module.Chats,
            "get_chat_by_id_and_user_id",
            AsyncMock(return_value=stored_chat),
        ),
    ):
        result = await middleware_module.chat_image_generation_handler(
            _chat_request(), form_data, extra_params, user
        )
    return {
        "generated": calls["generations"].called,
        "edited": calls["edits"].called,
        "statuses": statuses,
        "system_message": next(
            (m["content"] for m in result["messages"] if m["role"] == "system"), None
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["temporary:session-1", "saved-chat-1"])
async def test_image_handler_generates_nothing_while_generation_is_switched_off(
    middleware_module, user_model, config_store, chat_id
):
    with _config(config_store, **{"image_generation.enable": False}):
        outcome = await _run_image_handler(middleware_module, _chat_user(user_model), chat_id)

    assert outcome["generated"] is False, (
        "a chat session opened before the admin switched image generation off still "
        "produced images: the handler is reachable without the /images routes that "
        "enforce the switch (#27759)"
    )
    assert "Creating image" not in outcome["statuses"], (
        "the user was told an image was being created while image generation was off"
    )


@pytest.mark.asyncio
async def test_image_handler_still_generates_while_generation_is_on(
    middleware_module, user_model, config_store
):
    """Nearby: the switch check must not disable the feature it guards."""
    with _config(config_store):
        outcome = await _run_image_handler(
            middleware_module, _chat_user(user_model), "saved-chat-1"
        )

    assert outcome["generated"] is True
    assert "Creating image" in outcome["statuses"]


@pytest.mark.asyncio
async def test_image_editing_stays_available_when_only_editing_is_enabled(
    middleware_module, user_model, config_store
):
    """Nearby: editing has its own switch, so it must survive generation being off."""
    files = [{"type": "image", "url": "data:image/png;base64,AAAA"}]
    with _config(config_store, **{"image_generation.enable": False}):
        outcome = await _run_image_handler(
            middleware_module, _chat_user(user_model), "saved-chat-1", files=files
        )

    assert outcome["edited"] is True
    assert outcome["generated"] is False


@pytest.mark.asyncio
async def test_image_handler_leaves_the_payload_alone_without_an_event_emitter(
    middleware_module, user_model, config_store
):
    """Nearby: the pre-existing early return for a caller with no emitter."""
    with _config(config_store):
        result = await middleware_module.chat_image_generation_handler(
            _chat_request(),
            {"model": "m", "messages": [{"role": "user", "content": "draw me a cat"}]},
            {"__metadata__": {"chat_id": "saved-chat-1"}, "__event_emitter__": None},
            _chat_user(user_model),
        )

    assert [m["role"] for m in result["messages"]] == ["user"]


# --- 27 / 92: the message-sending path gates the folder and the target message -

# Raised by the stubbed payload processing so nothing past the gates actually runs.
# Whether it was raised at all is the signal; chat_completion only re-raises it as a
# 400 when the request carries no message id, so the call itself is what we record.
REACHED_THE_PIPELINE = "reached-the-pipeline"


@pytest.fixture(scope="module")
def main_module(owui_module):
    return owui_module("open_webui.main")


@pytest.fixture(scope="session")
def access_grants_model(owui_module):
    return owui_module("open_webui.models.access_grants").AccessGrants


def _folder(folder_id: str, owner: str):
    from open_webui.models.folders import FolderModel

    return FolderModel(
        id=folder_id,
        parent_id=None,
        user_id=owner,
        name=folder_id,
        created_at=0,
        updated_at=0,
    )


def _channel_message(message_id: str, author: str, channel_id: str):
    from open_webui.models.messages import MessageModel

    return MessageModel(
        id=message_id,
        user_id=author,
        channel_id=channel_id,
        content="hello",
        created_at=0,
        updated_at=0,
    )


async def _send_message(
    main, form_data, user, *, folders=None, messages=None, channel=None, has_access=False
):
    """Drive the real /api/chat/completions handler as far as the metadata gates,
    with every store and the model pipeline stubbed at the boundary."""
    import open_webui.models.folders as folders_module

    request = _chat_request()
    request.app.state.redis = None
    request.state.metadata = {}

    insert_new_chat = AsyncMock()
    process_chat_payload = AsyncMock(side_effect=Exception(REACHED_THE_PIPELINE))
    folder_lookup = AsyncMock(side_effect=lambda folder_id, db=None: (folders or {}).get(folder_id))
    message_lookup = AsyncMock(side_effect=lambda message_id: (messages or {}).get(message_id))

    with (
        patch.object(main.Models, "get_model_by_id", AsyncMock(return_value=None)),
        patch.object(main, "check_model_access", AsyncMock(return_value=None)),
        patch.object(main.Chats, "get_chat_by_id", AsyncMock(return_value=None)),
        patch.object(main.Chats, "get_chat_folder_id", AsyncMock(return_value=None)),
        patch.object(main.Chats, "insert_new_chat", insert_new_chat),
        patch.object(main.Channels, "get_channel_by_id", AsyncMock(return_value=channel)),
        patch.object(main.Channels, "is_user_channel_member", AsyncMock(return_value=True)),
        # Denied unless the caller asks for it, so a gate can never be satisfied by
        # the harness itself. Channel tests grant it to isolate the authorship check.
        patch.object(main.AccessGrants, "has_access", AsyncMock(return_value=has_access)),
        patch.object(folders_module.Folders, "get_folder_by_id", folder_lookup),
        patch.object(main.Messages, "get_message_by_id", message_lookup),
        patch.object(main, "publish_event", AsyncMock()),
        patch.object(main, "emit_chat_list_event", AsyncMock()),
        patch.object(main, "get_event_emitter", AsyncMock(return_value=None)),
        patch.object(main, "has_active_tasks", AsyncMock(return_value=False)),
        patch.object(main, "process_chat_payload", process_chat_payload),
    ):
        outcome = {"status": None, "detail": None}
        try:
            await main.chat_completion(request, form_data, user)
        except HTTPException as e:
            outcome = {"status": e.status_code, "detail": e.detail}
    return {
        **outcome,
        "filed": insert_new_chat.called,
        "reached": process_chat_payload.called,
    }


def _reached_the_pipeline(outcome) -> bool:
    return outcome["reached"] is True


@pytest.mark.asyncio
async def test_new_chat_is_not_filed_into_a_folder_the_sender_cannot_write(
    main_module, user_model, access_grants_model
):
    victim_folder = _folder("victim-folder", owner="bob")
    with patch.object(access_grants_model, "has_access", AsyncMock(return_value=False)):
        outcome = await _send_message(
            main_module,
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "parent_id": None,
                "folder_id": victim_folder.id,
            },
            _chat_user(user_model),
            folders={victim_folder.id: victim_folder},
        )

    assert outcome["status"] == 404, (
        "sending a message with another user's folder_id filed the new chat into that "
        f"folder with no write check at all (#28366): {outcome}"
    )
    assert outcome["filed"] is False


@pytest.mark.asyncio
async def test_new_chat_is_not_filed_into_a_folder_that_does_not_exist(main_module, user_model):
    """The same gate must treat an unknown folder like an inaccessible one."""
    outcome = await _send_message(
        main_module,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "parent_id": None,
            "folder_id": "no-such-folder",
        },
        _chat_user(user_model),
        folders={},
    )

    assert outcome["status"] == 404, outcome
    assert outcome["filed"] is False


@pytest.mark.asyncio
async def test_new_chat_is_filed_into_the_senders_own_folder(main_module, user_model):
    """Nearby: the owner still files their own chats."""
    own_folder = _folder("own-folder", owner="alice")
    outcome = await _send_message(
        main_module,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "parent_id": None,
            "folder_id": own_folder.id,
        },
        _chat_user(user_model),
        folders={own_folder.id: own_folder},
    )

    assert _reached_the_pipeline(outcome), outcome
    assert outcome["filed"] is True


@pytest.mark.asyncio
async def test_new_chat_with_no_folder_is_unaffected(main_module, user_model):
    outcome = await _send_message(
        main_module,
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "parent_id": None},
        _chat_user(user_model),
    )

    assert _reached_the_pipeline(outcome), outcome
    assert outcome["filed"] is True


@pytest.mark.asyncio
async def test_channel_write_access_does_not_allow_editing_another_members_message(
    main_module, user_model
):
    channel = types.SimpleNamespace(id="c1", type="channel", user_id="bob")
    theirs = _channel_message("bob-message", author="bob", channel_id="c1")

    outcome = await _send_message(
        main_module,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "rewrite this"}],
            "chat_id": "channel:c1",
            "id": theirs.id,
        },
        _chat_user(user_model),
        channel=channel,
        has_access=True,
        messages={theirs.id: theirs},
    )

    assert outcome["status"] == 403, (
        "write access on a channel let a member point the model at another member's "
        f"message and overwrite it (#28631): {outcome}"
    )


@pytest.mark.asyncio
async def test_editing_your_own_channel_message_is_still_allowed(main_module, user_model):
    channel = types.SimpleNamespace(id="c1", type="channel", user_id="bob")
    mine = _channel_message("alice-message", author="alice", channel_id="c1")

    outcome = await _send_message(
        main_module,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "rewrite this"}],
            "chat_id": "channel:c1",
            "id": mine.id,
        },
        _chat_user(user_model),
        channel=channel,
        has_access=True,
        messages={mine.id: mine},
    )

    assert _reached_the_pipeline(outcome), outcome


@pytest.mark.asyncio
async def test_an_admin_may_still_edit_any_channel_message(main_module, user_model):
    """Nearby: the authorship check is explicitly admin-exempt."""
    channel = types.SimpleNamespace(id="c1", type="channel", user_id="bob")
    theirs = _channel_message("bob-message", author="bob", channel_id="c1")

    outcome = await _send_message(
        main_module,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "rewrite this"}],
            "chat_id": "channel:c1",
            "id": theirs.id,
        },
        _chat_user(user_model, role="admin"),
        channel=channel,
        has_access=True,
        messages={theirs.id: theirs},
    )

    assert _reached_the_pipeline(outcome), outcome


@pytest.mark.asyncio
async def test_a_message_from_another_channel_is_still_refused(main_module, user_model):
    """Nearby: the pre-existing cross-channel check must survive the new clause."""
    channel = types.SimpleNamespace(id="c1", type="channel", user_id="alice")
    elsewhere = _channel_message("alice-message", author="alice", channel_id="c2")

    outcome = await _send_message(
        main_module,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "rewrite this"}],
            "chat_id": "channel:c1",
            "id": elsewhere.id,
        },
        _chat_user(user_model),
        channel=channel,
        has_access=True,
        messages={elsewhere.id: elsewhere},
    )

    assert outcome["status"] == 403, outcome
