"""Regression: the legacy chat-features block trusted client-supplied feature flags.

open-webui 0.11.0 fix `897d69a35` (#26703): `process_chat_payload` dispatched to
`chat_web_search_handler` / `chat_image_generation_handler` whenever the request body
carried `features.web_search` / `features.image_generation` and `params.function_calling`
was `legacy`. The native function-calling path (`get_builtin_tools`) already gated both on
the per-user `features.*` permission, so a user denied image generation or web search could
still trigger the billable server-side call by sending the legacy request format. The fix
puts the same admin-or-`has_permission` gate on both legacy branches.

Discriminates: passes on v0.11.0, fails on v0.10.2 (a denied user's legacy request still
reaches the image generation and web search handlers).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

# Chat features a request payload can switch on that dispatch a privileged, billable
# server-side call. Both request formats must reach the same decision for each.
BILLABLE_FEATURES = ("web_search", "image_generation")

# Builtin tools the native format exposes per feature, all permission-gated.
NATIVE_TOOLS_PER_FEATURE = {
    "web_search": {"search_web", "fetch_url"},
    "image_generation": {"generate_image", "edit_image"},
    "code_interpreter": {"execute_code"},
}

# Global feature switches, kept on so a denied request is only ever stopped by the
# permission check.
FEATURE_SWITCHES = {
    "web.search.enable": True,
    "image_generation.enable": True,
    "images.edit.enable": True,
    "code_interpreter.enable": True,
}

ALL_PAYLOAD_FEATURES = {
    "voice": True,
    "memory": True,
    "web_search": True,
    "image_generation": True,
    "code_interpreter": True,
}


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def builtin_tools_loader(owui_module):
    return owui_module("open_webui.utils.tools")


@pytest.fixture(scope="session")
def groups_model(owui_module):
    return owui_module("open_webui.models.groups").Groups


@pytest.fixture(scope="session")
def user_model(owui_module):
    return owui_module("open_webui.models.users").UserModel


@pytest.fixture(scope="session")
def config_store(owui_module):
    owui_module("open_webui.config")
    return owui_module("open_webui.models.config").Config


def _request():
    app = types.SimpleNamespace(state=types.SimpleNamespace(MODELS={"m": {"id": "m"}}))
    return types.SimpleNamespace(
        app=app,
        cookies={},
        headers={},
        state=types.SimpleNamespace(direct=False, internal=False),
    )


def _user(user_model, role="user"):
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


@contextmanager
def _permit(config_store, *permitted_features):
    """Grant exactly `permitted_features` by serving the config reads the production
    code makes, rather than writing the real, process-wide config store."""
    overrides = {
        **FEATURE_SWITCHES,
        "user.permissions": {
            "features": {
                feature: feature in permitted_features
                for feature in ("web_search", "image_generation", "code_interpreter", "memories")
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


async def _legacy_activations(
    middleware_module, groups_model, user, features, function_calling="legacy"
):
    """Run the real payload pipeline in the legacy format and report which
    privileged handlers it dispatched to."""
    handlers = {
        "web_search": AsyncMock(
            side_effect=lambda request, form_data, extra_params, user: form_data
        ),
        "image_generation": AsyncMock(
            side_effect=lambda request, form_data, extra_params, user: form_data
        ),
    }
    form_data = {
        "model": "m",
        "messages": [{"role": "user", "content": "draw me a cat and look it up"}],
        "features": dict(features),
    }
    metadata = {
        "chat_id": "",
        "params": {"function_calling": function_calling},
        "features": dict(features),
    }
    with (
        patch.object(middleware_module, "chat_web_search_handler", handlers["web_search"]),
        patch.object(
            middleware_module, "chat_image_generation_handler", handlers["image_generation"]
        ),
        patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        await middleware_module.process_chat_payload(
            _request(), form_data, user, metadata, {"id": "m"}
        )
    return {name: handler.called for name, handler in handlers.items()}


async def _native_tool_names(builtin_tools_loader, groups_model, user, features):
    """Run the real native-format builtin tool resolution and report the tool names offered."""
    extra_params = {
        "__user__": user.model_dump(),
        "__metadata__": {"chat_id": ""},
    }
    with patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])):
        tools = await builtin_tools_loader.get_builtin_tools(
            _request(), extra_params, dict(features), {"id": "m"}
        )
    return set(tools)


# --- Narrow: exactly the bug --------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_image_generation_ignored_for_denied_user(
    middleware_module, groups_model, user_model, config_store
):
    with _permit(config_store):
        activations = await _legacy_activations(
            middleware_module, groups_model, _user(user_model), {"image_generation": True}
        )
    assert activations["image_generation"] is False, (
        "a user denied features.image_generation triggered server-side image generation, "
        "and the provider bill for it, just by sending the legacy request format (#26703)"
    )


@pytest.mark.asyncio
async def test_legacy_web_search_ignored_for_denied_user(
    middleware_module, groups_model, user_model, config_store
):
    with _permit(config_store):
        activations = await _legacy_activations(
            middleware_module, groups_model, _user(user_model), {"web_search": True}
        )
    assert activations["web_search"] is False, (
        "a user denied features.web_search triggered a server-side web search, and the "
        "search provider bill for it, just by sending the legacy request format (#26703)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", BILLABLE_FEATURES)
async def test_native_format_already_refused_denied_user(
    builtin_tools_loader, groups_model, user_model, config_store, feature
):
    """The format that was already correct: no tool is offered, so nothing to call."""
    with _permit(config_store):
        tool_names = await _native_tool_names(
            builtin_tools_loader, groups_model, _user(user_model), {feature: True}
        )
    assert not (tool_names & NATIVE_TOOLS_PER_FEATURE[feature]), (
        f"the native request format offered {feature} tools to a user denied that feature (#26703)"
    )


# --- Broad: both formats must reach the same decision -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", BILLABLE_FEATURES)
@pytest.mark.parametrize("role,permitted", [("user", False), ("user", True), ("admin", False)])
async def test_request_formats_agree_on_every_billable_feature(
    middleware_module,
    builtin_tools_loader,
    groups_model,
    user_model,
    config_store,
    feature,
    role,
    permitted,
):
    user = _user(user_model, role=role)
    features = {feature: True}

    with _permit(config_store, *([feature] if permitted else [])):
        legacy_allowed = (
            await _legacy_activations(middleware_module, groups_model, user, features)
        )[feature]
        native_allowed = bool(
            (await _native_tool_names(builtin_tools_loader, groups_model, user, features))
            & NATIVE_TOOLS_PER_FEATURE[feature]
        )

    assert legacy_allowed == native_allowed, (
        f"the legacy and native request formats disagree on {feature} for a {role} "
        f"(permitted={permitted}): legacy={legacy_allowed}, native={native_allowed}. "
        "Whichever format is more permissive is a way around the permission (#26703)"
    )


@pytest.mark.asyncio
async def test_denied_user_enabling_every_feature_activates_nothing(
    middleware_module, builtin_tools_loader, groups_model, user_model, config_store
):
    """Catch-all: a payload that switches on every feature it can must still leave a
    fully denied user with no permission-gated capability in either format."""
    user = _user(user_model)

    with _permit(config_store):
        activations = await _legacy_activations(
            middleware_module, groups_model, user, ALL_PAYLOAD_FEATURES
        )
        tool_names = await _native_tool_names(
            builtin_tools_loader, groups_model, user, ALL_PAYLOAD_FEATURES
        )
    gated_tools = set().union(*NATIVE_TOOLS_PER_FEATURE.values())

    assert not any(activations.values()), (
        f"a fully denied user reached {[k for k, v in activations.items() if v]} by switching "
        "on every feature flag in one legacy payload (#26703)"
    )
    assert not (tool_names & gated_tools), (
        f"a fully denied user was offered the gated tools {sorted(tool_names & gated_tools)} "
        "by switching on every feature flag in one native payload (#26703)"
    )


# --- Nearby: behaviour that was already correct and must stay ------------------


@pytest.mark.asyncio
async def test_permitted_user_gets_both_features_in_both_formats(
    middleware_module, builtin_tools_loader, groups_model, user_model, config_store
):
    user = _user(user_model)
    features = {"web_search": True, "image_generation": True}

    with _permit(config_store, "web_search", "image_generation"):
        activations = await _legacy_activations(middleware_module, groups_model, user, features)
        tool_names = await _native_tool_names(builtin_tools_loader, groups_model, user, features)

    assert activations == {"web_search": True, "image_generation": True}
    assert {"search_web", "generate_image"} <= tool_names


@pytest.mark.asyncio
async def test_admin_gets_both_features_in_both_formats(
    middleware_module, builtin_tools_loader, groups_model, user_model, config_store
):
    """Admins bypass the permission lookup entirely, so denying every feature must
    not lock them out."""
    admin = _user(user_model, role="admin")
    features = {"web_search": True, "image_generation": True}

    with _permit(config_store):
        activations = await _legacy_activations(middleware_module, groups_model, admin, features)
        tool_names = await _native_tool_names(builtin_tools_loader, groups_model, admin, features)

    assert activations == {"web_search": True, "image_generation": True}
    assert {"search_web", "generate_image"} <= tool_names


@pytest.mark.asyncio
async def test_payload_enabling_nothing_is_unaffected(
    middleware_module, groups_model, user_model, config_store
):
    user = _user(user_model)

    with _permit(config_store, "web_search", "image_generation"):
        empty = await _legacy_activations(middleware_module, groups_model, user, {})
        switched_off = await _legacy_activations(
            middleware_module, groups_model, user, {"web_search": False, "image_generation": False}
        )

    assert empty == {"web_search": False, "image_generation": False}
    assert switched_off == {"web_search": False, "image_generation": False}


@pytest.mark.asyncio
async def test_native_function_calling_skips_the_legacy_handlers(
    middleware_module, groups_model, user_model, config_store
):
    """The forced RAG handlers belong to the legacy format only; a permitted user on
    native FC gets the tools instead, so the gate must not resurrect them."""
    with _permit(config_store, "web_search", "image_generation"):
        activations = await _legacy_activations(
            middleware_module,
            groups_model,
            _user(user_model),
            {"web_search": True, "image_generation": True},
            function_calling="native",
        )
    assert activations == {"web_search": False, "image_generation": False}
