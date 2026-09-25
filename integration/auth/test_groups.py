"""Journey: groups as the admin panel manages them, and what membership grants.

An admin creates, renames and deletes a group. Adding someone to it gives them what the group
holds: a knowledge base shared with the group becomes readable and a permission the group grants
shows in their effective permissions, which is what the web client reads to decide what to
offer. Removing them from the group, or deleting the group, takes both away again.

Discriminates: fails with the member filter dropped from `get_groups_by_member_id` (an outsider
reads the group's knowledge base and gets its permission).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness.knowledge_bases import knowledge_base

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

GROUP_PERMISSIONS = {"workspace": {"models": True}}


@pytest.fixture
def group(admin) -> Iterator[dict]:
    with admin.client() as client:
        created = client.post(
            "/api/v1/groups/create",
            json={
                "name": f"team-{uuid.uuid4().hex[:8]}",
                "description": "a team",
                "permissions": GROUP_PERMISSIONS,
            },
        )
        assert created.status_code == 200, created.text
        yield created.json()
        client.delete(f"/api/v1/groups/id/{created.json()['id']}/delete")


@pytest.fixture
def shared_knowledge(admin, group) -> Iterator[str]:
    grant = {"principal_type": "group", "principal_id": group["id"], "permission": "read"}
    with admin.client() as client, knowledge_base(client, "Team notes", [grant]) as knowledge_id:
        yield knowledge_id


def change_members(admin, group_id: str, action: str, *members) -> None:
    with admin.client() as client:
        changed = client.post(
            f"/api/v1/groups/id/{group_id}/users/{action}",
            json={"user_ids": [member.id for member in members]},
        )
    assert changed.status_code == 200, changed.text


def what_membership_gives(account, knowledge_id: str) -> tuple[bool, bool]:
    """Whether the account reads the shared knowledge base, and has the group's permission."""
    with account.client() as client:
        reads = client.get(f"/api/v1/knowledge/{knowledge_id}").status_code == 200
        permissions = client.get("/api/v1/users/permissions").json()
    return reads, permissions["workspace"]["models"]


def test_a_group_is_created_renamed_and_deleted(admin, group):
    with admin.client() as client:
        listed = [entry["id"] for entry in client.get("/api/v1/groups/").json()]
        renamed = client.post(
            f"/api/v1/groups/id/{group['id']}/update",
            json={"name": f"{group['name']}-renamed", "description": "renamed"},
        )
        stored = client.get(f"/api/v1/groups/id/{group['id']}").json()
        deleted = client.delete(f"/api/v1/groups/id/{group['id']}/delete")
        listed_after = [entry["id"] for entry in client.get("/api/v1/groups/").json()]
        read_after = client.get(f"/api/v1/groups/id/{group['id']}")

    assert group["id"] in listed
    assert renamed.status_code == 200, renamed.text
    assert (stored["name"], stored["description"]) == (f"{group['name']}-renamed", "renamed")
    assert deleted.json() is True
    assert group["id"] not in listed_after
    assert read_after.status_code in (401, 404)


def test_joining_grants_the_shared_resource_and_the_permission(
    admin, make_user, group, shared_knowledge
):
    member, outsider = make_user(), make_user()
    assert what_membership_gives(member, shared_knowledge) == (False, False)

    change_members(admin, group["id"], "add", member)

    assert what_membership_gives(member, shared_knowledge) == (True, True)
    assert what_membership_gives(outsider, shared_knowledge) == (False, False)
    with member.client() as client:
        assert group["id"] in [entry["id"] for entry in client.get("/api/v1/groups/").json()]


def test_leaving_the_group_takes_both_away(admin, make_user, group, shared_knowledge):
    member = make_user()
    change_members(admin, group["id"], "add", member)

    change_members(admin, group["id"], "remove", member)

    assert what_membership_gives(member, shared_knowledge) == (False, False)


def test_deleting_the_group_takes_both_away(admin, make_user, group, shared_knowledge):
    member = make_user()
    change_members(admin, group["id"], "add", member)

    with admin.client() as client:
        assert client.delete(f"/api/v1/groups/id/{group['id']}/delete").json() is True

    assert what_membership_gives(member, shared_knowledge) == (False, False)
    with admin.client() as client:
        groups = client.get(f"/api/v1/users/{member.id}").json()["groups"]
    assert groups == []
