"""Regression: read access to a tool handed out the tool's Python source.

open-webui 0.11.0, fix `c05de13b4` (PR #27005): `GET /api/v1/tools/id/{id}` answered with
`ToolAccessResponse(**tool.model_dump(), ...)`. The response model has no `content` field but
allows extra keys, so the source came back to any caller with a mere read grant, including every
user when a tool is shared with `*`. Tool source routinely carries API keys and internal URLs. The
fix drops `content` for callers without write access. The listings had the same mouth:
`Tools.get_tools(defer_content=True)` ignored the flag until `310ae9130` (PR #27387) made it a
column select, so `GET /tools/` and `GET /tools/list` shipped every tool's source.

Twin of unit/security/test_tool_source_exposure.py.

Discriminates: passes on bbfa876af; fails with c05de13b4 reverted (the read-grant user and the
admin without the access-control bypass get the source by id) and with 310ae9130's column select
reverted (both listings carry the source); the other tests pass on both.
"""

from __future__ import annotations

import uuid

import pytest

from harness.actors import admin_of, create_user

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SECRET = "sk-live-do-not-leak"
SOURCE = f'''API_KEY = "{SECRET}"


class Tools:
    def get_weather(self, city: str) -> str:
        """Look up the weather in a city.

        :param city: the city to look up
        """
        return city
'''
# shares one boot with test_admin_shared_chat_access
RESTRICTED_ADMIN_ENV = {"ENABLE_ADMIN_CHAT_ACCESS": "false", "BYPASS_ADMIN_ACCESS_CONTROL": "false"}


def _grant(owner, tool_id: str, *grants: tuple[str, str]) -> None:
    access_grants = [
        {"principal_type": "user", "principal_id": principal_id, "permission": permission}
        for principal_id, permission in grants
    ]
    with owner.client() as client:
        updated = client.post(
            f"/api/v1/tools/id/{tool_id}/access/update", json={"access_grants": access_grants}
        )
    assert updated.status_code == 200, updated.text


def _create_tool(owner) -> str:
    tool_id = f"weather_{uuid.uuid4().hex[:8]}"
    form = {
        "id": tool_id,
        "name": "Weather",
        "content": SOURCE,
        "meta": {"description": "Looks up the weather"},
    }
    with owner.client() as client:
        created = client.post("/api/v1/tools/create", json=form)
    assert created.status_code == 200, created.text
    return tool_id


def _delete_tool(owner, tool_id: str) -> None:
    with owner.client() as client:
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")


@pytest.fixture
def tool_id(admin):
    """An admin's tool whose source embeds a secret."""
    created = _create_tool(admin)
    yield created
    _delete_tool(admin, created)


def _tool_by_id(actor, tool_id: str) -> dict:
    with actor.client() as client:
        found = client.get(f"/api/v1/tools/id/{tool_id}")
    assert found.status_code == 200, found.text
    return found.json()


# narrow: a read grant is not a licence to read the source


def test_read_grant_does_not_reveal_the_source(admin, make_user, tool_id):
    reader = make_user()
    _grant(admin, tool_id, (reader.id, "read"))

    tool = _tool_by_id(reader, tool_id)

    assert tool["write_access"] is False
    assert SECRET not in str(tool), "a read-only caller got the tool's source (#27005)"


def test_admin_without_the_bypass_does_not_get_another_admins_source(instance_with):
    restricted = instance_with(RESTRICTED_ADMIN_ENV)
    owner = admin_of(restricted)
    other_admin = create_user(restricted, role="admin")
    tool = _create_tool(owner)
    try:
        found = _tool_by_id(other_admin, tool)
    finally:
        _delete_tool(owner, tool)

    assert found["write_access"] is False
    assert SECRET not in str(found), "the admin role alone unlocked the tool's source (#27005)"


# broad: no path a read-only caller has hands out the source


@pytest.mark.parametrize("listing", ["/api/v1/tools/", "/api/v1/tools/list"])
def test_listings_do_not_carry_the_source(admin, make_user, tool_id, listing):
    reader = make_user()
    _grant(admin, tool_id, (reader.id, "read"))

    with reader.client() as client:
        listed = client.get(listing)

    assert listed.status_code == 200, listed.text
    entries = [entry for entry in listed.json() if entry["id"] == tool_id]
    assert len(entries) == 1, f"the shared tool is missing from {listing}"
    assert SECRET not in listed.text, f"{listing} shipped the tool's source (#27005)"


def test_export_refuses_a_user_without_the_export_permission(admin, make_user, tool_id):
    reader = make_user()
    _grant(admin, tool_id, (reader.id, "read"))

    with reader.client() as client:
        exported = client.get("/api/v1/tools/export")

    assert exported.status_code == 401, exported.text


# nearby: write access still gets the source, read access still gets the rest


def test_write_grant_reveals_the_source(admin, make_user, tool_id):
    writer = make_user()
    _grant(admin, tool_id, (writer.id, "read"), (writer.id, "write"))

    tool = _tool_by_id(writer, tool_id)

    assert tool["write_access"] is True
    assert tool["content"] == SOURCE


def test_owner_gets_the_source(admin, tool_id):
    assert _tool_by_id(admin, tool_id)["content"] == SOURCE


def test_another_admin_with_the_bypass_gets_the_source(make_user, tool_id):
    tool = _tool_by_id(make_user(role="admin"), tool_id)

    assert tool["write_access"] is True
    assert tool["content"] == SOURCE


def test_read_grant_still_gets_the_descriptive_fields(admin, make_user, tool_id):
    reader = make_user()
    _grant(admin, tool_id, (reader.id, "read"))

    tool = _tool_by_id(reader, tool_id)

    assert tool["name"] == "Weather"
    assert tool["meta"]["description"] == "Looks up the weather"
    assert [spec["name"] for spec in tool["specs"]] == ["get_weather"]


def test_user_without_a_grant_is_refused(make_user, tool_id):
    with make_user().client() as client:
        found = client.get(f"/api/v1/tools/id/{tool_id}")

    assert found.status_code == 401


def test_unknown_tool_is_not_found(user):
    with user.client() as client:
        found = client.get("/api/v1/tools/id/no_such_tool")

    assert found.status_code == 404
