"""Pins tool creation from chat (v0.13.0, upstream 91241306e).

The `create_tool` builtin lets the model save a Python tool for the account it chats with. It is
offered to an admin and to a user with the workspace tools permission, and the tool it saves is
private: owned by that account, shared with nobody, listed in its own workspace and hidden from
everyone else. A user without the permission is not offered it, and calling it anyway saves
nothing. The model editor's Tool Creation switch (`meta.builtinTools.tools`) takes it off the
tools a model is offered and leaves the other builtins alone.

Discriminates: passes on v0.13.0 and dev a01b2ed55; the creation and switch tests fail on v0.12.5
(the builtin does not exist, so it is never offered). In backend copies of v0.13.0, dropping the
permission checks of `get_builtin_tools` and `create_tool` turned the no-permission test red
(offered), and with the create route's check dropped too the call saved a tool; `create_tool`
saving an everyone-reads grant turned the admin's private tool test red; and ignoring
`builtinTools.tools` turned the switch test red.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Iterator

import pytest

from harness.access import make_group
from harness.python_tools import EVERYONE_READS
from harness.tool_calls import offered_tools, run_tool
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOOL_BUILDER = {"workspace": {"tools": True}}
SOURCE = '''class Tools:
    def shout(self, text: str) -> str:
        """Say the text back in capitals.

        :param text: what to say
        """
        return text.upper()
'''


def tool_builder(admin, make_user):
    account = make_user()
    make_group(admin, [account], TOOL_BUILDER)
    return account


@contextlib.contextmanager
def created_from_chat(account, upstream) -> Iterator[tuple[str, dict]]:
    """Have the model call `create_tool` for `account`; yields what it was told and the request."""
    tag = uuid.uuid4().hex[:8]
    request = {
        "id": f"shouter_{tag}",
        "name": f"Shouter {tag}",
        "description": f"Says things loudly {tag}",
        "content": SOURCE,
    }
    with account.client() as client:
        result = run_tool(client, upstream, "create_tool", request)
    try:
        yield result, request
    finally:
        with account.client() as client:
            client.delete(f"/api/v1/tools/id/{request['id']}/delete")


def offered(actor, upstream, **options) -> set[str]:
    with actor.client() as client:
        return offered_tools(client, upstream, **options)


def listed_ids(actor) -> set[str]:
    with actor.client() as client:
        listed = client.get("/api/v1/tools/list")
    assert listed.status_code == 200, listed.text
    return {tool["id"] for tool in listed.json()}


@pytest.mark.parametrize("role", ["admin", "user with the tools permission"])
def test_a_tool_created_from_chat_is_private_to_its_creator(role, admin, make_user, upstream):
    creator = make_user(role="admin") if role == "admin" else tool_builder(admin, make_user)
    other_builder = tool_builder(admin, make_user)
    tools = offered(creator, upstream)
    assert "create_tool" in tools, f"create_tool is not offered: {sorted(tools)}"

    with created_from_chat(creator, upstream) as (result, request):
        tool_id = request["id"]
        expected = {"id": tool_id, "url": f"/workspace/tools/edit?id={tool_id}"}
        assert json.loads(result) == expected, result
        with creator.client() as client:
            stored = client.get(f"/api/v1/tools/id/{tool_id}")
        assert stored.status_code == 200, stored.text
        tool = stored.json()
        assert tool["user_id"] == creator.id, f"the tool belongs to {tool['user_id']}"
        assert tool["access_grants"] == [], f"the new tool is shared: {tool['access_grants']}"
        assert (tool["name"], tool["meta"]["description"]) == (
            request["name"],
            request["description"],
        )
        assert [spec["name"] for spec in tool["specs"]] == ["shout"], tool["specs"]
        assert tool_id in listed_ids(creator), "the tool is missing from its creator's workspace"

        assert tool_id not in listed_ids(other_builder), "another account's workspace lists it"
        with other_builder.client() as client:
            opened = client.get(f"/api/v1/tools/id/{tool_id}")
        assert opened.status_code == 401, f"another account opened it: {opened.status_code}"


def test_without_the_tools_permission_no_tool_is_created_from_chat(admin, make_user, upstream):
    account = make_user()
    tools = offered(account, upstream)
    assert "get_current_timestamp" in tools, sorted(tools)
    assert "create_tool" not in tools, "create_tool is offered without the permission"

    with created_from_chat(account, upstream) as (result, request):
        with admin.client() as client:
            stored = client.get(f"/api/v1/tools/id/{request['id']}")
    assert stored.status_code == 404, f"a tool was created ({result}): {stored.text}"


@contextlib.contextmanager
def model_with(admin, builtin_tools: dict | None) -> Iterator[str]:
    """A native function calling model with these builtin tool switches; yields its id."""
    model_id = f"toolmaker-{uuid.uuid4().hex[:8]}"
    meta: dict = {"capabilities": {}}
    if builtin_tools is not None:
        meta["builtinTools"] = builtin_tools
    with admin.client() as client:
        made = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": meta,
                "params": {"function_calling": "native"},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert made.status_code == 200, made.text
        try:
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_the_tool_creation_switch_takes_create_tool_off_the_model(admin, upstream):
    with model_with(admin, None) as default_model:
        by_default = offered(admin, upstream, model=default_model)
    with model_with(admin, {"tools": False}) as switched_off:
        with_switch_off = offered(admin, upstream, model=switched_off)

    assert "create_tool" in by_default, sorted(by_default)
    assert "create_tool" not in with_switch_off, sorted(with_switch_off)
    assert "get_current_timestamp" in with_switch_off, sorted(with_switch_off)
