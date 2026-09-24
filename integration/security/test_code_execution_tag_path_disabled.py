"""Regression: a `<code_interpreter>` tag in the reply ran even under native function calling.

open-webui 0.11.1 fix `ac85b0f2a` (#29024): the streaming handler decided whether to act on
`<code_interpreter>` tags from the feature flag and five authorization checks only, so tag
detection ran in every tool-calling mode. Native mode never teaches the tag format and offers
`execute_code` as a tool instead, so code a model merely quoted in its reply (from a document,
a search result, the user's own message) was sent to the executor. The fix gates detection on
`params.function_calling == 'legacy'`, the condition that already decides whether the tag
format is taught.

The code interpreter runs on its Jupyter engine pointed at the listener, so every execution
attempt shows up there as a kernel request.

Twin of unit/security/test_code_execution_tag_path_disabled.py.

Discriminates: passes on dev bbfa876af; with the `function_calling == 'legacy'` gate dropped
the native, unset, empty, `default` and `LEGACY` replies reach the kernel. Each gate test goes
red when its own check is taken out of the gate.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CODE_EXECUTION_CONFIG = ("/api/v1/configs/code_execution", "/api/v1/configs/code_execution")
TAG_FORMAT = '<code_interpreter type="code" lang="python">'
TAG_REPLY = f'The document contains this snippet:\n{TAG_FORMAT}print("tag ran")</code_interpreter>'
KERNELS = "/api/kernels"

UNSET = None  # what the web client sends by default
NON_LEGACY_MODES = ["native", UNSET, "", "default", "LEGACY"]
GATES = ["feature_flag", "global_switch", "user_permission", "model_capability", "builtin_tool"]


def _mode_id(function_calling: str | None) -> str:
    return {UNSET: "unset", "": "empty"}.get(function_calling, function_calling)


def _set_code_execution(admin, **changes) -> None:
    with admin.client() as client:
        current = client.get(CODE_EXECUTION_CONFIG[0]).json()
        client.post(CODE_EXECUTION_CONFIG[1], json={**current, **changes}).raise_for_status()


@pytest.fixture
def jupyter(admin, preserve, listener):
    """The code interpreter on, its Jupyter engine served by the listener."""
    preserve(CODE_EXECUTION_CONFIG)
    _set_code_execution(
        admin,
        ENABLE_CODE_INTERPRETER=True,
        CODE_INTERPRETER_ENGINE="jupyter",
        CODE_INTERPRETER_JUPYTER_URL=listener.base_url,
        CODE_INTERPRETER_JUPYTER_AUTH="",
    )
    return listener


@pytest.fixture
def custom_model(admin):
    """`custom_model(meta)` is a model over the mock, readable by everyone, with that meta."""
    created: list[str] = []
    grant = {"principal_type": "user", "principal_id": "*", "permission": "read"}

    def create(meta: dict) -> str:
        model_id = f"code-gate-{uuid.uuid4().hex[:8]}"
        form = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": model_id,
            "meta": meta,
            "params": {},
            "access_grants": [grant],
        }
        with admin.client() as client:
            client.post("/api/v1/models/create", json=form).raise_for_status()
            created.append(model_id)
            client.get("/api/models").raise_for_status()  # registers it for chat
        return model_id

    yield create
    with admin.client() as client:
        for model_id in created:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _ask_with_code_interpreter(actor, function_calling, *, feature=True, model=MOCK_MODEL_ID):
    params = {} if function_calling is UNSET else {"params": {"function_calling": function_calling}}
    with actor.client() as client:
        ask(client, "run it", model=model, features={"code_interpreter": feature}, **params)


def _tag_format_taught(upstream) -> bool:
    first_request = upstream.chat_requests()[0]
    return any(TAG_FORMAT in str(entry.get("content")) for entry in first_request["messages"])


def _offered_tools(upstream) -> set[str]:
    first_request = upstream.chat_requests()[0]
    return {tool["function"]["name"] for tool in first_request.get("tools", [])}


# Narrow


@pytest.mark.parametrize("function_calling", ["native", UNSET], ids=_mode_id)
def test_a_tag_in_the_reply_does_not_run_under_native_function_calling(
    function_calling, jupyter, make_user, upstream
):
    upstream.queue(reply.text(TAG_REPLY))
    _ask_with_code_interpreter(make_user(), function_calling)

    assert jupyter.requests_to(KERNELS) == [], (
        f"with function calling {_mode_id(function_calling)!r} a <code_interpreter> block "
        "quoted in the reply was sent to the executor, though the model never called "
        "execute_code (#29024)"
    )


# Broad


@pytest.mark.parametrize("function_calling", ["legacy", *NON_LEGACY_MODES], ids=_mode_id)
def test_a_tag_runs_only_in_the_mode_that_teaches_the_tag_format(
    function_calling, jupyter, make_user, upstream
):
    upstream.queue(reply.text(TAG_REPLY))
    _ask_with_code_interpreter(make_user(), function_calling)

    taught = _tag_format_taught(upstream)
    ran = bool(jupyter.requests_to(KERNELS))
    assert ran == taught, (
        f"with function calling {_mode_id(function_calling)!r} the tag format was taught: "
        f"{taught}, a tag in the reply ran: {ran}; the executor may only act on a format the "
        "request taught the model (#29024)"
    )


# Nearby


def test_legacy_mode_still_teaches_and_runs_the_tag(jupyter, make_user, upstream):
    upstream.queue(reply.text(TAG_REPLY))
    _ask_with_code_interpreter(make_user(), "legacy")

    assert _tag_format_taught(upstream), "legacy mode stopped teaching the tag format"
    assert jupyter.requests_to(KERNELS), "legacy mode stopped running the tag it taught"


@pytest.mark.parametrize("gate", GATES)
def test_every_authorization_gate_still_stops_the_tag(
    gate, jupyter, admin, preserve, make_user, custom_model, upstream
):
    """The mode gate went in front of five existing checks; none may have been dropped."""
    feature, model = True, MOCK_MODEL_ID
    if gate == "feature_flag":
        feature = False
    elif gate == "global_switch":
        _set_code_execution(admin, ENABLE_CODE_INTERPRETER=False)
    elif gate == "user_permission":
        preserve("permissions")
        with admin.client() as client:
            permissions = client.get("/api/v1/users/default/permissions").json()
            permissions["features"]["code_interpreter"] = False
            client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()
    elif gate == "model_capability":
        model = custom_model({"capabilities": {"code_interpreter": False}})
    else:
        model = custom_model({"builtinTools": {"code_interpreter": False}})

    upstream.queue(reply.text(TAG_REPLY))
    _ask_with_code_interpreter(make_user(), "legacy", feature=feature, model=model)

    assert jupyter.requests_to(KERNELS) == [], f"the tag ran with the {gate} gate closed"


@pytest.mark.parametrize("function_calling", ["native", UNSET], ids=_mode_id)
def test_an_explicit_execute_code_call_still_runs_under_native_function_calling(
    function_calling, jupyter, make_user, upstream
):
    upstream.queue(
        reply.tool_call("execute_code", {"code": 'print("tool ran")'}), reply.text("done")
    )
    _ask_with_code_interpreter(make_user(), function_calling)

    assert "execute_code" in _offered_tools(upstream), (
        f"execute_code is no longer offered with function calling {_mode_id(function_calling)!r}"
    )
    assert jupyter.requests_to(KERNELS), "the model's execute_code call never ran"


def test_execute_code_is_not_offered_with_the_interpreter_switched_off(
    jupyter, admin, make_user, upstream
):
    _set_code_execution(admin, ENABLE_CODE_INTERPRETER=False)
    _ask_with_code_interpreter(make_user(), "native")

    assert "execute_code" not in _offered_tools(upstream)
