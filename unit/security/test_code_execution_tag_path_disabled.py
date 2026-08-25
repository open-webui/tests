"""Regression: the code-interpreter tag path stayed live under native function calling.

open-webui 0.11.1 fix `ac85b0f2a` (#29024): `streaming_chat_response_handler` computed
`DETECT_CODE_INTERPRETER` from the feature flag and the five authorization checks only, so
tag detection ran in every tool-calling mode. Native (Agentic) mode never teaches the
`<code_interpreter>` format and exposes `execute_code` as a builtin tool instead, so a model
that emitted the tag in ordinary reply text - copied from a document, echoed back from user
input, quoted from a search result - had that code handed to the executor with no tool call.
The fix adds `params.function_calling == 'legacy'` as the first gate, matching the condition
that already decides whether the tag prompt is injected at all.

The gate lives inside a ~2000-line handler that cannot be driven in isolation, so the narrow
tests pull the real `DETECT_CODE_INTERPRETER` assignment (plus the three assignments it
reads) out of the shipped middleware with `ast` and evaluate it against a chosen metadata /
model / user, exercising the actual awaits. The nearby tests drive the real
`get_builtin_tools` and `process_chat_payload` to show the explicit tool call and the legacy
prompt injection are untouched.

Discriminates: passes on v0.11.1, fails on v0.11.0 (a reply containing the tag reaches the
executor on native function calling).
"""

from __future__ import annotations

import ast
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

# Assignments the gate reads, in the order the handler makes them.
GATE_INPUTS = ("features", "model_capabilities", "builtin_tools_meta")

# Tool-calling modes that are not the legacy XML-tag mode. Every one of them gets
# `execute_code` as a builtin tool, so none of them may run tags found in reply text.
NON_LEGACY_MODES = ["native", "default", "", None, "LEGACY"]


def _gate_statements(open_webui_backend: Path) -> list[ast.stmt]:
    """The real `DETECT_CODE_INTERPRETER` assignment and the ones feeding it."""
    source = (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DETECT_CODE_INTERPRETER"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, (
        f"expected exactly one `DETECT_CODE_INTERPRETER` assignment in middleware.py, found "
        f"{len(assignments)}; the extraction below no longer matches the shipped code (#29024)"
    )

    block = None
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list) and any(item is assignments[0] for item in value):
                block = value
    assert block is not None, "could not locate the statement block holding the gate (#29024)"

    index = next(i for i, item in enumerate(block) if item is assignments[0])
    start = index
    while start > 0:
        previous = block[start - 1]
        if not (
            isinstance(previous, ast.Assign)
            and len(previous.targets) == 1
            and isinstance(previous.targets[0], ast.Name)
            and previous.targets[0].id in GATE_INPUTS
        ):
            break
        start -= 1
    statements = block[start : index + 1]
    assert len(statements) == len(GATE_INPUTS) + 1, (
        f"expected the gate to be preceded by {GATE_INPUTS}, got "
        f"{[ast.unparse(s).split(' =')[0] for s in statements]} (#29024)"
    )
    return statements


def _gate_evaluator(open_webui_backend: Path):
    """Compile the extracted statements into an awaitable `gate(namespace) -> bool`."""
    statements = _gate_statements(open_webui_backend)
    function = ast.AsyncFunctionDef(
        name="_gate",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[*statements, ast.Return(ast.Name(id="DETECT_CODE_INTERPRETER", ctx=ast.Load()))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    async def evaluate(namespace: dict):
        scope = dict(namespace)
        exec(compile(module, "middleware.py", "exec"), scope)
        return await scope["_gate"]()

    return evaluate


async def _tag_detection_enabled(
    open_webui_backend: Path,
    *,
    function_calling="legacy",
    feature_flag=True,
    builtin_tool_enabled=True,
    globally_enabled=True,
    model_capable=True,
    role="user",
    permitted=True,
):
    """Would a `<code_interpreter>` block in ordinary reply text be executed?"""

    class FakeConfig:
        @staticmethod
        async def get(key, default=None):
            if key == "code_interpreter.enable":
                return globally_enabled
            if key == "user.permissions":
                return {"features": {"code_interpreter": permitted}}
            raise AssertionError(f"unexpected config read in the gate: {key}")

    async def has_permission(user_id, key, permissions):
        assert key == "features.code_interpreter"
        return bool(permissions.get("features", {}).get("code_interpreter"))

    namespace = {
        "metadata": {
            "params": {"function_calling": function_calling},
            "features": {"code_interpreter": feature_flag},
        },
        "model": {
            "info": {
                "meta": {
                    "capabilities": {"code_interpreter": model_capable},
                    "builtinTools": {"code_interpreter": builtin_tool_enabled},
                }
            }
        },
        "user": types.SimpleNamespace(id="alice", role=role),
        "Config": FakeConfig,
        "has_permission": has_permission,
    }
    return bool(await _gate_evaluator(open_webui_backend)(namespace))


# --- Harness for the paths that must keep working ------------------------------


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
def _code_interpreter_allowed(config_store, enabled=True):
    """Serve the config reads the production code makes, without writing the real store."""
    overrides = {
        "code_interpreter.enable": enabled,
        "code_interpreter.engine": "pyodide",
        "code_interpreter.prompt_template": "",
        "user.permissions": {"features": {"code_interpreter": True}},
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


async def _builtin_tool_names(builtin_tools_loader, groups_model, user, function_calling):
    extra_params = {
        "__user__": user.model_dump(),
        "__metadata__": {"chat_id": "", "params": {"function_calling": function_calling}},
    }
    with patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])):
        tools = await builtin_tools_loader.get_builtin_tools(
            _request(), extra_params, {"code_interpreter": True}, {"id": "m"}
        )
    return set(tools)


async def _tag_prompt_injected(middleware_module, groups_model, user, function_calling):
    """Does the payload pipeline teach the model the `<code_interpreter>` tag format?"""
    features = {"code_interpreter": True}
    form_data = {
        "model": "m",
        "messages": [{"role": "user", "content": "what is 2 + 2?"}],
        "features": dict(features),
    }
    metadata = {
        "chat_id": "",
        "params": {"function_calling": function_calling},
        "features": dict(features),
    }
    with patch.object(groups_model, "get_groups_by_member_id", AsyncMock(return_value=[])):
        payload, _metadata, _events = await middleware_module.process_chat_payload(
            _request(), form_data, user, metadata, {"id": "m"}
        )
    rendered = " ".join(str(message.get("content", "")) for message in payload["messages"])
    return "<code_interpreter" in rendered


# --- Narrow: exactly the bug --------------------------------------------------


@pytest.mark.asyncio
async def test_native_function_calling_does_not_execute_a_tag_in_the_reply(open_webui_backend):
    assert await _tag_detection_enabled(open_webui_backend, function_calling="native") is False, (
        "on native function calling a <code_interpreter> block appearing in ordinary reply "
        "text is still sent to the executor, so code runs without the model ever calling "
        "execute_code (#29024)"
    )


@pytest.mark.asyncio
async def test_legacy_function_calling_still_executes_a_tag_in_the_reply(open_webui_backend):
    """Legacy mode is where the tag format is taught, and must keep working."""
    assert await _tag_detection_enabled(open_webui_backend, function_calling="legacy") is True


@pytest.mark.asyncio
async def test_unset_function_calling_does_not_execute_a_tag_in_the_reply(open_webui_backend):
    """The default install sends no `function_calling` param at all."""
    assert await _tag_detection_enabled(open_webui_backend, function_calling=None) is False, (
        "a chat with no explicit tool-calling mode executed a <code_interpreter> block found "
        "in reply text; the tag prompt is never injected in that mode (#29024)"
    )


# --- Broad: the tag path must exist exactly where the tag format is taught -----


@pytest.mark.asyncio
@pytest.mark.parametrize("function_calling", NON_LEGACY_MODES, ids=lambda mode: repr(mode))
async def test_no_non_legacy_mode_executes_a_tag_in_the_reply(open_webui_backend, function_calling):
    assert (
        await _tag_detection_enabled(open_webui_backend, function_calling=function_calling) is False
    ), (
        f"tool-calling mode {function_calling!r} is not the legacy tag mode, yet a "
        "<code_interpreter> block in reply text still reaches the executor (#29024)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("function_calling", ["legacy", *NON_LEGACY_MODES])
async def test_tag_execution_follows_the_tag_prompt_injection(
    open_webui_backend, middleware_module, groups_model, user_model, config_store, function_calling
):
    """The two halves of the tag path: whether the model is taught the format, and whether
    the format is acted on. A mode that acts on tags it never taught runs code the model
    was never asked to run."""
    with _code_interpreter_allowed(config_store):
        injected = await _tag_prompt_injected(
            middleware_module, groups_model, _user(user_model, role="admin"), function_calling
        )
    detected = await _tag_detection_enabled(
        open_webui_backend, function_calling=function_calling, role="admin"
    )
    assert injected == detected, (
        f"in mode {function_calling!r} the tag prompt injection ({injected}) and the tag "
        f"execution ({detected}) disagree; the executor acts on a format the request never "
        "taught the model (#29024)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate",
    ["feature_flag", "builtin_tool_enabled", "globally_enabled", "model_capable", "permitted"],
)
async def test_every_authorization_gate_still_disables_the_tag_path(open_webui_backend, gate):
    """The mode gate was added in front of five existing checks; none of them may have been
    dropped in the process."""
    assert (
        await _tag_detection_enabled(open_webui_backend, function_calling="legacy", **{gate: False})
        is False
    ), f"the tag path ran with {gate} switched off (#29024)"


# --- Nearby: the explicit tool call, which is how code is meant to run ---------


@pytest.mark.asyncio
@pytest.mark.parametrize("function_calling", ["legacy", "native", None])
async def test_execute_code_tool_is_offered_in_every_mode(
    builtin_tools_loader, groups_model, user_model, config_store, function_calling
):
    with _code_interpreter_allowed(config_store):
        tool_names = await _builtin_tool_names(
            builtin_tools_loader, groups_model, _user(user_model), function_calling
        )
    assert "execute_code" in tool_names, (
        f"the explicit execute_code tool disappeared in mode {function_calling!r}; gating the "
        "tag path must not take the supported way to run code with it (#29024)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("function_calling", ["legacy", "native"])
async def test_execute_code_tool_still_honours_the_global_switch(
    builtin_tools_loader, groups_model, user_model, config_store, function_calling
):
    with _code_interpreter_allowed(config_store, enabled=False):
        tool_names = await _builtin_tool_names(
            builtin_tools_loader, groups_model, _user(user_model), function_calling
        )
    assert "execute_code" not in tool_names, (
        f"execute_code was offered in mode {function_calling!r} with the code interpreter "
        "switched off globally (#29024)"
    )


@pytest.mark.asyncio
async def test_legacy_mode_still_teaches_the_tag_format(
    middleware_module, groups_model, user_model, config_store
):
    with _code_interpreter_allowed(config_store):
        injected = await _tag_prompt_injected(
            middleware_module, groups_model, _user(user_model, role="admin"), "legacy"
        )
    assert injected, "legacy mode stopped injecting the <code_interpreter> prompt (#29024)"


def test_tag_reply_is_the_shape_the_parser_looks_for(open_webui_backend):
    """These tests are only meaningful if the shipped tag prompt still uses that shape."""
    source = (open_webui_backend / "open_webui" / "config.py").read_text(encoding="utf-8")
    assert '<code_interpreter type="code" lang="python">' in source, (
        "the shipped code-interpreter prompt no longer uses the tag shape these tests "
        "assume; re-check what the reply parser matches (#29024)"
    )
