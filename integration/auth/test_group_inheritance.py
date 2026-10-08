"""Journey: members of a subgroup receive what every group above them was given, and no more.

Three nested groups hold one member each. Whatever an admin shares with the top group (a model
preset, a knowledge base, a tool, a prompt, a note, a skill, a channel or a calendar) is read by
the leaf group's member and the middle group's member as well as the top group's own, and is
refused to an outsider. What is shared with the leaf group only is read by the leaf's member and
by nobody above it. A permission switched on in the top group reaches the leaf's member in the
permissions the web client reads and in the session, while a permission only the leaf holds stays
out of the top's. A chat with a preset shared with the top group is answered for the leaf's
member. The admin's group preview resolves through the ancestors: the leaf's preview lists what
the top group was given and reports the inherited and effective permissions, and the top's
preview leaves out what only the leaf holds.

Discriminates: in a backend copy, making `user_group_memberships` ignore `include_inherited`
turns the leaf and middle rows of every kind red (and the permission and chat rows), dropping
the ancestor walk from the group preview (the group's own id as the only one) turns the two
preview tests red. The reverse rows pass on both and guard against a grant that climbs upward.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.access import grant, make_group
from harness.channel_quotes import enable_channels
from harness.group_tree import build_chain
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_SOURCE = '''class Tools:
    def ping(self) -> str:
        """Answer pong."""
        return "pong"
'''
CREATE_STATUS = 200


def _created(response):
    assert response.status_code == CREATE_STATUS, response.text
    return response.json()["id"]


def _suffix() -> str:
    return uuid.uuid4().hex[:8]


def _create_model(client, grants):
    body = {
        "id": f"preset-{_suffix()}",
        "base_model_id": MOCK_MODEL_ID,
        "name": "Inherited preset",
        "meta": {},
        "params": {},
        "access_grants": grants,
    }
    return _created(client.post("/api/v1/models/create", json=body))


def _create_knowledge(client, grants):
    body = {"name": f"Inherited {_suffix()}", "description": "", "access_grants": grants}
    return _created(client.post("/api/v1/knowledge/create", json=body))


def _create_tool(client, grants):
    body = {
        "id": f"tool_{_suffix()}",
        "name": "Ping",
        "content": TOOL_SOURCE,
        "meta": {"description": "inherited"},
        "access_grants": grants,
    }
    return _created(client.post("/api/v1/tools/create", json=body))


def _create_prompt(client, grants):
    body = {
        "command": f"inherited-{_suffix()}",
        "name": "Inherited",
        "content": "text",
        "access_grants": grants,
    }
    return _created(client.post("/api/v1/prompts/create", json=body))


def _create_note(client, grants):
    body = {"title": f"Inherited {_suffix()}", "data": {"content": {}}, "access_grants": grants}
    return _created(client.post("/api/v1/notes/create", json=body))


def _create_skill(client, grants):
    body = {
        "id": f"skill-{_suffix()}",
        "name": f"Inherited {_suffix()}",
        "content": "Write release notes.",
        "access_grants": grants,
    }
    return _created(client.post("/api/v1/skills/create", json=body))


def _create_channel(client, grants):
    body = {"name": f"inherited-{_suffix()}", "type": None, "access_grants": grants}
    return _created(client.post("/api/v1/channels/create", json=body))


def _create_calendar(client, grants):
    body = {"name": f"Inherited {_suffix()}", "access_grants": grants}
    return _created(client.post("/api/v1/calendars/create", json=body))


def _reads(path: str):
    def sees(client, resource_id: str) -> bool:
        return client.get(path.format(id=resource_id)).status_code == 200

    return sees


def _listed_in(path: str, key: str | None):
    def sees(client, resource_id: str) -> bool:
        listed = client.get(path)
        assert listed.status_code == 200, listed.text
        items = listed.json()[key] if key else listed.json()
        return resource_id in {item["id"] for item in items}

    return sees


# kind: how an admin creates one shared with `grants`, and whether an account's client sees it
KINDS = {
    "model": (_create_model, _listed_in("/api/models", "data")),
    "knowledge": (_create_knowledge, _reads("/api/v1/knowledge/{id}")),
    "tool": (_create_tool, _reads("/api/v1/tools/id/{id}")),
    "prompt": (_create_prompt, _reads("/api/v1/prompts/id/{id}")),
    "note": (_create_note, _reads("/api/v1/notes/{id}")),
    "skill": (_create_skill, _reads("/api/v1/skills/id/{id}")),
    "channel": (_create_channel, _listed_in("/api/v1/channels/", None)),
    "calendar": (_create_calendar, _reads("/api/v1/calendars/{id}")),
}


def _skill_ids(client) -> set[str]:
    listed = client.get("/api/v1/skills/")
    assert listed.status_code == 200, listed.text
    return {skill["id"] for skill in listed.json()}


@pytest.fixture(autouse=True)
def no_skill_left_behind(admin):
    # An admin's active skill is listed in every later admin chat on the shared instance.
    with admin.client() as client:
        before = _skill_ids(client)
    yield
    with admin.client() as client:
        for skill_id in _skill_ids(client) - before:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")


@pytest.fixture
def chain(admin, make_user, preserve):
    preserve("admin_config")
    enable_channels(admin)
    tree = build_chain(admin, make_user)
    yield tree
    with admin.client() as client:
        for group_id in (tree.leaf, tree.middle, tree.top):
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _shared_with(admin, kind: str, group_id: str) -> str:
    create, _ = KINDS[kind]
    with admin.client() as client:
        return create(client, [grant("group", group_id, "read")])


def _sees(actor, kind: str, resource_id: str) -> bool:
    _, sees = KINDS[kind]
    with actor.client() as client:
        return sees(client, resource_id)


@pytest.mark.parametrize("kind", KINDS)
def test_what_the_top_group_is_given_reaches_every_member_below(admin, make_user, chain, kind):
    resource_id = _shared_with(admin, kind, chain.top)

    reached = {
        "top": _sees(chain.top_member, kind, resource_id),
        "middle": _sees(chain.middle_member, kind, resource_id),
        "leaf": _sees(chain.leaf_member, kind, resource_id),
        "outsider": _sees(make_user(), kind, resource_id),
    }

    assert reached == {"top": True, "middle": True, "leaf": True, "outsider": False}


@pytest.mark.parametrize("kind", KINDS)
def test_what_only_the_leaf_group_is_given_stays_out_of_the_groups_above(admin, chain, kind):
    resource_id = _shared_with(admin, kind, chain.leaf)

    reached = {
        "top": _sees(chain.top_member, kind, resource_id),
        "middle": _sees(chain.middle_member, kind, resource_id),
        "leaf": _sees(chain.leaf_member, kind, resource_id),
    }

    assert reached == {"top": False, "middle": False, "leaf": True}


def _refresh_models(admin) -> None:
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()


def _chat_with(actor, model_id: str):
    body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    with actor.client() as client:
        return client.post("/api/chat/completions", json=body)


def test_a_preset_shared_with_the_top_group_answers_the_leafs_member(admin, chain, upstream):
    model_id = _shared_with(admin, "model", chain.top)
    _refresh_models(admin)
    upstream.queue(reply.text("answered through the parent grant"))

    answered = _chat_with(chain.leaf_member, model_id)

    assert answered.status_code == 200, answered.text
    assert "answered through the parent grant" in answered.text


def test_a_preset_shared_with_the_leaf_group_refuses_the_top_groups_member(admin, chain, upstream):
    model_id = _shared_with(admin, "model", chain.leaf)
    _refresh_models(admin)

    refused = _chat_with(chain.top_member, model_id)

    assert refused.status_code in (400, 401, 403, 404), (
        f"the top group's member chatted with the leaf's preset: {refused.status_code}"
    )
    assert upstream.chat_requests() == []


def _permissions(actor) -> dict:
    with actor.client() as client:
        listed = client.get("/api/v1/users/permissions")
        session = client.get("/api/v1/auths/")
    assert listed.status_code == 200, listed.text
    assert session.status_code == 200, session.text
    return {"listed": listed.json(), "session": session.json().get("permissions", {})}


def test_a_permission_the_top_group_grants_reaches_the_leafs_member(admin, make_user):
    top_member, leaf_member, outsider = make_user(), make_user(), make_user()
    top = make_group(admin, [top_member], {"workspace": {"models": True}})
    leaf = make_group(admin, [leaf_member], parent_id=top)

    try:
        for who, expected in ((top_member, True), (leaf_member, True), (outsider, False)):
            seen = _permissions(who)
            assert seen["listed"]["workspace"]["models"] is expected, who.id
            assert seen["session"]["workspace"]["models"] is expected, who.id
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/groups/id/{leaf}/delete")
            client.delete(f"/api/v1/groups/id/{top}/delete")


def test_a_permission_only_the_leaf_group_grants_stays_out_of_the_top(admin, make_user):
    top_member, leaf_member = make_user(), make_user()
    top = make_group(admin, [top_member])
    leaf = make_group(admin, [leaf_member], {"workspace": {"tools": True}}, parent_id=top)

    try:
        assert _permissions(leaf_member)["listed"]["workspace"]["tools"] is True
        assert _permissions(top_member)["listed"]["workspace"]["tools"] is False
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/groups/id/{leaf}/delete")
            client.delete(f"/api/v1/groups/id/{top}/delete")


def _preview(admin, group_id: str) -> dict:
    with admin.client() as client:
        previewed = client.get(f"/api/v1/groups/id/{group_id}/preview")
    assert previewed.status_code == 200, previewed.text
    return previewed.json()


def test_the_leafs_preview_lists_what_the_top_group_was_given(admin, chain):
    on_top = {kind: _shared_with(admin, kind, chain.top) for kind in ("model", "knowledge", "tool")}
    on_leaf = {
        kind: _shared_with(admin, kind, chain.leaf) for kind in ("model", "knowledge", "tool")
    }

    leaf_preview = _preview(admin, chain.leaf)
    top_preview = _preview(admin, chain.top)

    for section, kind in (("models", "model"), ("knowledge", "knowledge"), ("tools", "tool")):
        in_leaf = {item["id"] for item in leaf_preview[section]["items"]}
        in_top = {item["id"] for item in top_preview[section]["items"]}
        assert {on_top[kind], on_leaf[kind]} <= in_leaf, f"the leaf's preview misses {section}"
        assert on_top[kind] in in_top
        assert on_leaf[kind] not in in_top, f"the top's preview lists the leaf's {kind}"


def test_the_preview_reports_inherited_and_effective_permissions(admin, make_user):
    top = make_group(admin, [], {"workspace": {"models": True}})
    leaf = make_group(admin, [], {"workspace": {"tools": True}}, parent_id=top)

    try:
        leaf_preview = _preview(admin, leaf)
        top_preview = _preview(admin, top)
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/groups/id/{leaf}/delete")
            client.delete(f"/api/v1/groups/id/{top}/delete")

    assert leaf_preview["permissions"] == {"workspace": {"tools": True}}
    assert leaf_preview["inherited_permissions"]["workspace"]["models"] is True
    assert leaf_preview["inherited_permissions"]["workspace"]["tools"] is False
    assert leaf_preview["effective_permissions"]["workspace"]["models"] is True
    assert leaf_preview["effective_permissions"]["workspace"]["tools"] is True
    assert top_preview["effective_permissions"]["workspace"]["tools"] is False
