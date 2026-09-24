"""User listing and channel membership regressions fixed in 0.11.1, seen through the API.

* The DM lookup counted `channel_member` rows, so the leftover membership of a deleted account
  still counted towards the exact-set match: messaging someone from a conversation that had
  included a since-deleted account opened a second conversation (`a41faa3c22`, #28257).
* SCIM read the generic user queries, so a directory sync listed (and could then change)
  accounts created with a password inside Open WebUI; SCIM now sees only accounts carrying an
  `oauth` or `scim` payload. The same commit moved the user list's ordering into its own `sort`
  argument: stuffing `direction` into the filter made it truthy, so the newest-first fallback
  never ran and the admin list came back in storage order (`fb4f476316`).
* Read grants for users and for groups became two filters that AND together, so a channel
  shared with a person and a group they are not in listed nobody, and never its owner
  (`e3e82b1471`, #28289, #28288).

Twin of unit/models/test_users_and_channels.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the DM count back on
`channel_member`, the SCIM queries unfiltered, `direction` back in the filter, the two grant
filters restored): a second DM is opened, SCIM finds the local account, the list is oldest
first and the channel lists nobody.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from harness.actors import create_user
from harness.channel_quotes import enable_channels
from harness.scim import SCIM_ENV, provision, scim_client

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def _read_grant(principal_type: str, principal_id: str) -> dict:
    return {"principal_type": principal_type, "principal_id": principal_id, "permission": "read"}


def _dm_with_user(client: httpx.Client, user_id: str) -> str:
    opened = client.get(f"/api/v1/channels/users/{user_id}")
    assert opened.status_code == 200, opened.text
    return opened.json()["id"]


# ---------------------------------------------------------------- the DM lookup


def test_a_deleted_member_does_not_hide_the_existing_conversation(channels_on, admin, make_user):
    alice, bob, carol = make_user(), make_user(), make_user()
    with alice.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={"name": "", "type": "dm", "user_ids": [bob.id, carol.id]},
        )
        assert created.status_code == 200, created.text
        with admin.client() as admin_client:
            admin_client.delete(f"/api/v1/users/{carol.id}").raise_for_status()

        reopened = _dm_with_user(client, bob.id)

    assert reopened == created.json()["id"], (
        "the deleted account's leftover membership hid the conversation, so a second one was opened"
    )


def test_messaging_the_same_person_twice_reuses_the_conversation(channels_on, make_user):
    alice, bob = make_user(), make_user()
    with alice.client() as client:
        first = _dm_with_user(client, bob.id)
        again = _dm_with_user(client, bob.id)
    with bob.client() as client:
        from_bob = _dm_with_user(client, alice.id)

    assert first == again == from_bob


def test_a_three_way_conversation_is_not_the_two_way_one(channels_on, make_user):
    alice, bob, carol = make_user(), make_user(), make_user()
    with alice.client() as client:
        group_dm = client.post(
            "/api/v1/channels/create",
            json={"name": "", "type": "dm", "user_ids": [bob.id, carol.id]},
        )
        assert group_dm.status_code == 200, group_dm.text

        assert _dm_with_user(client, bob.id) != group_dm.json()["id"]


# ---------------------------------------------------------------- SCIM sees provisioned accounts


@pytest.fixture(scope="module")
def scim_instance(instance_with):
    return instance_with(SCIM_ENV)


@pytest.mark.slow
def test_scim_cannot_find_or_read_a_local_password_account(scim_instance):
    local = create_user(scim_instance)

    with scim_client(scim_instance) as directory:
        listed = directory.get("/Users", params={"filter": f'userName eq "{local.email}"'})
        read = directory.get(f"/Users/{local.id}")

    assert listed.status_code == 200, listed.text
    assert (listed.json()["totalResults"], listed.json()["Resources"]) == (0, []), (
        "a directory sync can see an account created with a password inside Open WebUI"
    )
    assert read.status_code == 404, read.text


@pytest.mark.slow
def test_scim_still_finds_the_accounts_it_provisioned(scim_instance):
    with scim_client(scim_instance) as directory:
        user = provision(directory)
        listed = directory.get("/Users", params={"filter": f'userName eq "{user["userName"]}"'})
        read = directory.get(f"/Users/{user['id']}")

    assert [resource["id"] for resource in listed.json()["Resources"]] == [user["id"]]
    assert read.status_code == 200, read.text


# ---------------------------------------------------------------- the admin user list


def test_the_user_list_is_newest_first_without_a_search(admin, make_user):
    older = make_user()
    time.sleep(1.1)  # created_at has one-second resolution
    newer = make_user()

    with admin.client() as client:
        listed = client.get("/api/v1/users/")
    assert listed.status_code == 200, listed.text
    users = listed.json()["users"]

    created = [user["created_at"] for user in users]
    assert created == sorted(created, reverse=True), "the user list is not newest first"
    ids = [user["id"] for user in users]
    assert newer.id in ids and older.id in ids, "the newest accounts are not on the first page"
    assert ids.index(newer.id) < ids.index(older.id)


@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_a_name_sort_with_a_search_still_applies(admin, make_user, direction):
    marker = uuid.uuid4().hex[:8]
    last, first = make_user(name=f"zzz {marker}"), make_user(name=f"aaa {marker}")

    with admin.client() as client:
        listed = client.get(
            "/api/v1/users/", params={"query": marker, "order_by": "name", "direction": direction}
        )
    ids = [user["id"] for user in listed.json()["users"]]

    assert ids == ([first.id, last.id] if direction == "asc" else [last.id, first.id])


# ---------------------------------------------------------------- who a channel's members are


def _group_with(admin, *members) -> str:
    with admin.client() as client:
        group = client.post(
            "/api/v1/groups/create",
            json={"name": f"group-{uuid.uuid4().hex[:8]}", "description": ""},
        )
        assert group.status_code == 200, group.text
        group_id = group.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add",
            json={"user_ids": [member.id for member in members]},
        )
        assert added.status_code == 200, added.text
    return group_id


def _members_of(admin, *grants: dict, query: str | None = None) -> dict:
    with admin.client() as client:
        channel = client.post(
            "/api/v1/channels/create",
            json={"name": f"shared-{uuid.uuid4().hex[:8]}", "access_grants": list(grants)},
        )
        assert channel.status_code == 200, channel.text
        members = client.get(
            f"/api/v1/channels/{channel.json()['id']}/members",
            params={"query": query} if query else None,
        )
    assert members.status_code == 200, members.text
    return members.json()


def test_a_channel_shared_with_a_person_and_a_group_lists_both_and_its_owner(
    channels_on, admin, make_user
):
    invited, group_member = make_user(), make_user()
    group_id = _group_with(admin, group_member)

    members = _members_of(admin, _read_grant("user", invited.id), _read_grant("group", group_id))

    assert {user["id"] for user in members["users"]} == {admin.id, invited.id, group_member.id}, (
        "the member list dropped the owner or intersected the person and group grants"
    )
    assert members["total"] == 3


@pytest.mark.parametrize("grant_kinds", [("user",), ("group",)])
def test_the_owner_is_listed_whatever_the_grants(channels_on, admin, make_user, grant_kinds):
    invited, group_member = make_user(), make_user()
    group_id = _group_with(admin, group_member)
    grants = {"user": _read_grant("user", invited.id), "group": _read_grant("group", group_id)}

    members = _members_of(admin, *(grants[kind] for kind in grant_kinds))
    listed = {user["id"] for user in members["users"]}

    expected = {"user": invited.id, "group": group_member.id}
    assert listed == {admin.id, *(expected[kind] for kind in grant_kinds)}
    assert members["total"] == len(listed)


def test_a_public_channel_lists_everyone_but_pending_accounts(channels_on, admin, make_user):
    marker = uuid.uuid4().hex[:8]
    active = make_user(name=f"Active {marker}")
    pending = make_user(role="pending", name=f"Pending {marker}")

    members = _members_of(admin, _read_grant("user", "*"), query=marker)

    assert {user["id"] for user in members["users"]} == {active.id}
    assert pending.id not in {user["id"] for user in members["users"]}
