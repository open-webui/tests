"""Journey: what the group info and the admin's group access preview show, and to whom.

The access list of a shared resource names every group it is shared with, including groups the
viewer is not in, so any signed-in account resolves a group id to its name, description and
member count; the members and the group's permissions stay with the admin, who reads the full
group. The admin's Preview Group Access, documented as admin-only, lists the models, knowledge
bases and tools the group can read through its own grants or a public one, leaving out deactivated
models and anything shared only elsewhere, next to the totals and the group's permissions.

No route reaches the group search in the groups model; the groups list is the only search there
is, and it is covered by test_groups.

Discriminates: in a backend copy, returning the full group from the info route turns the info
test red (a non-member sees the permissions), and building the preview from every grant instead
of the group's turns the preview test red (the other group's knowledge base is listed).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import grant, make_group
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = {"workspace": {"models": True}}
TOOL_SOURCE = '''class Tools:
    def ping(self) -> str:
        """Answer pong."""
        return "pong"
'''


@pytest.fixture
def team(admin, make_user):
    """A group with one member and a workspace permission, and that member."""
    member = make_user()
    group_id = make_group(admin, [member], PERMISSIONS)
    yield group_id, member
    with admin.client() as client:
        client.delete(f"/api/v1/groups/id/{group_id}/delete")


def info(actor, group_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"/api/v1/groups/id/{group_id}/info")


def test_any_signed_in_account_resolves_a_group_to_its_name(admin, make_user, team):
    group_id, member = team
    with admin.client() as client:
        full = client.get(f"/api/v1/groups/id/{group_id}").json()

    answers = {"member": info(member, group_id), "non-member": info(make_user(), group_id)}

    for who, answer in answers.items():
        assert answer.status_code == 200, f"{who}: {answer.text}"
        shown = answer.json()
        assert (shown["name"], shown["description"], shown["member_count"]) == (
            full["name"],
            full["description"],
            1,
        ), who
        assert "permissions" not in shown and "user_ids" not in shown, f"{who} sees too much"
    assert full["permissions"]["workspace"]["models"] is True


def test_the_full_group_and_its_members_stay_with_the_admin(make_user, team):
    group_id, member = team
    with member.client() as client:
        answers = [
            client.get(f"/api/v1/groups/id/{group_id}").status_code,
            client.get(f"/api/v1/groups/id/{group_id}/export").status_code,
            client.post(f"/api/v1/groups/id/{group_id}/users").status_code,
        ]
    assert answers == [401, 401, 401]


def test_a_pending_account_and_an_unknown_group_get_no_info(make_user, team):
    group_id, member = team
    assert info(make_user(role="pending"), group_id).status_code == 401
    assert info(member, str(uuid.uuid4())).status_code == 401


# --------------------------------------------------------------------------- preview


def create_model(client: httpx.Client, access_grants: list[dict]) -> str:
    model_id = f"preview-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": model_id,
            "meta": {},
            "params": {},
            "access_grants": access_grants,
        },
    )
    assert created.status_code == 200, created.text
    return model_id


def create_knowledge(client: httpx.Client, access_grants: list[dict]) -> str:
    created = client.post(
        "/api/v1/knowledge/create",
        json={
            "name": f"kb {uuid.uuid4().hex[:6]}",
            "description": "",
            "access_grants": access_grants,
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def create_tool(client: httpx.Client, access_grants: list[dict]) -> str:
    tool_id = f"preview_{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/tools/create",
        json={
            "id": tool_id,
            "name": tool_id,
            "content": TOOL_SOURCE,
            "meta": {"description": "preview"},
            "access_grants": access_grants,
        },
    )
    assert created.status_code == 200, created.text
    return tool_id


@pytest.fixture
def shared_resources(admin, make_user, team):
    """Resources shared with the team, with everyone and with another group only."""
    group_id, _ = team
    other_group = make_group(admin, [make_user()])
    to_team, to_other = [grant("group", group_id, "read")], [grant("group", other_group, "read")]
    with admin.client() as client:
        made = {
            "model": create_model(client, to_team),
            "inactive_model": create_model(client, to_team),
            "knowledge": create_knowledge(client, to_team),
            "public_knowledge": create_knowledge(client, [EVERYONE_READS]),
            "other_knowledge": create_knowledge(client, to_other),
            "tool": create_tool(client, to_team),
            "other_tool": create_tool(client, to_other),
        }
        toggled = client.post("/api/v1/models/model/toggle", params={"id": made["inactive_model"]})
        assert toggled.json()["is_active"] is False, toggled.text
        yield made
        for model_key in ("model", "inactive_model"):
            client.post("/api/v1/models/model/delete", json={"id": made[model_key]})
        for knowledge_key in ("knowledge", "public_knowledge", "other_knowledge"):
            client.delete(f"/api/v1/knowledge/{made[knowledge_key]}/delete")
        for tool_key in ("tool", "other_tool"):
            client.delete(f"/api/v1/tools/id/{made[tool_key]}/delete")
        client.delete(f"/api/v1/groups/id/{other_group}/delete")


def test_the_preview_lists_what_the_group_can_read(admin, team, shared_resources):
    group_id, _ = team
    made = shared_resources

    with admin.client() as client:
        answer = client.get(f"/api/v1/groups/id/{group_id}/preview")

    assert answer.status_code == 200, answer.text
    preview = answer.json()
    listed = {
        kind: {item["id"] for item in preview[kind]["items"]}
        for kind in ("models", "knowledge", "tools")
    }
    assert preview["group"]["id"] == group_id
    assert made["model"] in listed["models"]
    assert made["inactive_model"] not in listed["models"], "a deactivated model was listed"
    assert {made["knowledge"], made["public_knowledge"]} <= listed["knowledge"]
    assert made["other_knowledge"] not in listed["knowledge"], "another group's share was listed"
    assert made["tool"] in listed["tools"] and made["other_tool"] not in listed["tools"]
    for kind in ("models", "knowledge", "tools"):
        assert preview[kind]["total"] >= len(listed[kind])
    assert preview["permissions"]["workspace"]["models"] is True


def test_the_preview_is_the_admins_alone(team):
    group_id, member = team
    with member.client() as client:
        assert client.get(f"/api/v1/groups/id/{group_id}/preview").status_code == 401


def test_the_preview_of_an_unknown_group_is_404(admin):
    with admin.client() as client:
        assert client.get(f"/api/v1/groups/id/{uuid.uuid4()}/preview").status_code == 404
