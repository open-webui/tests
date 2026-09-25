"""Journey: a directory syncing groups over SCIM, and removing the people it provisioned.

SCIM answers only the configured bearer token: no header, another token, or an Open WebUI
session of a user or even an admin is refused before anything is read or written. A group the
directory creates shows in the admin panel with its members; PATCH adds and removes members, PUT
replaces the name and the whole member list and DELETE removes the group, each mirrored in
`/api/v1/groups`. Deleting a provisioned person removes the account and its group memberships,
and the same address can be provisioned again.

Discriminates: fails with the `hmac.compare_digest` token check in `get_scim_auth` removed (a
wrong token and any Open WebUI session read and write the directory).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.actors import create_user
from harness.scim import PATCH_SCHEMA, SCIM_ENV, USER_SCHEMA, provision, scim_client

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"


@pytest.fixture(scope="module")
def scim_instance(instance_with):
    return instance_with(SCIM_ENV)


@pytest.fixture(scope="module")
def directory(scim_instance):
    with scim_client(scim_instance) as client:
        yield client


def members(*resources: dict) -> list[dict]:
    return [{"value": resource["id"]} for resource in resources]


def create_group(directory, *people: dict) -> dict:
    created = directory.post(
        "/Groups",
        json={
            "schemas": [GROUP_SCHEMA],
            "displayName": f"directory-team-{uuid.uuid4().hex[:8]}",
            "members": members(*people),
        },
    )
    assert created.status_code == 201, created.text
    return created.json()


def patch_group(directory, group_id: str, *operations: dict) -> httpx.Response:
    return directory.patch(
        f"/Groups/{group_id}", json={"schemas": [PATCH_SCHEMA], "Operations": list(operations)}
    )


def admin_panel_group(scim_instance, group_id: str) -> tuple[str, set[str]] | None:
    """The group's name and member ids as the admin panel sees them, or None when it is gone."""
    with scim_instance.client() as admin:
        listed = {group["id"]: group for group in admin.get("/api/v1/groups/").json()}
        if group_id not in listed:
            return None
        exported = admin.get(f"/api/v1/groups/id/{group_id}/export").json()
    return listed[group_id]["name"], set(exported["user_ids"])


def headers_for(scim_instance, credential: str) -> dict[str, str]:
    if credential == "none":
        return {}
    if credential == "basic-scheme":
        return {"Authorization": "Basic c2NpbTpzY2lt"}
    if credential == "wrong-token":
        token = "scim-test-token-wrong"
    elif credential == "admin-session":
        token = scim_instance.admin_token
    else:
        token = create_user(scim_instance).token
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- who may speak SCIM


@pytest.mark.parametrize(
    "credential", ["none", "wrong-token", "basic-scheme", "user-session", "admin-session"]
)
def test_only_the_scim_token_is_accepted(scim_instance, credential):
    base_url = f"{scim_instance.base_url}/api/v1/scim/v2"
    headers = headers_for(scim_instance, credential)
    group_name = f"refused-{uuid.uuid4().hex[:8]}"

    with httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as client:
        listed = client.get("/Users")
        created = client.post(
            "/Groups", json={"schemas": [GROUP_SCHEMA], "displayName": group_name}
        )
        provisioned = client.post(
            "/Users",
            json={
                "schemas": [USER_SCHEMA],
                "userName": f"{group_name}@example.com",
                "displayName": "Refused Person",
                "active": True,
            },
        )

    assert (listed.status_code, created.status_code, provisioned.status_code) == (401, 401, 401)
    with scim_instance.client() as admin:
        names = [group["name"] for group in admin.get("/api/v1/groups/").json()]
        accounts = admin.get("/api/v1/users/", params={"query": group_name}).json()["users"]
    assert group_name not in names
    assert accounts == []


# ---------------------------------------------------------------- the group lifecycle


def test_a_created_group_shows_in_the_admin_panel_with_its_members(scim_instance, directory):
    alice, bob = provision(directory), provision(directory)

    group = create_group(directory, alice, bob)

    assert {member["value"] for member in group["members"]} == {alice["id"], bob["id"]}
    assert admin_panel_group(scim_instance, group["id"]) == (
        group["displayName"],
        {alice["id"], bob["id"]},
    )


def test_patch_adds_and_removes_members(scim_instance, directory):
    alice, bob = provision(directory), provision(directory)
    group = create_group(directory, alice)

    added = patch_group(
        directory, group["id"], {"op": "add", "path": "members", "value": members(bob)}
    )
    assert added.status_code == 200, added.text
    assert admin_panel_group(scim_instance, group["id"])[1] == {alice["id"], bob["id"]}

    removed = patch_group(
        directory, group["id"], {"op": "remove", "path": f'members[value eq "{alice["id"]}"]'}
    )
    assert removed.status_code == 200, removed.text
    assert admin_panel_group(scim_instance, group["id"])[1] == {bob["id"]}


def test_patch_replaces_the_name_and_the_member_list(scim_instance, directory):
    alice, bob = provision(directory), provision(directory)
    group = create_group(directory, alice)
    new_name = f"{group['displayName']}-patched"

    replaced = patch_group(
        directory,
        group["id"],
        {"op": "replace", "path": "displayName", "value": new_name},
        {"op": "replace", "path": "members", "value": members(bob)},
    )

    assert replaced.status_code == 200, replaced.text
    assert admin_panel_group(scim_instance, group["id"]) == (new_name, {bob["id"]})


def test_put_replaces_the_name_and_the_whole_member_list(scim_instance, directory):
    alice, bob, carol = provision(directory), provision(directory), provision(directory)
    group = create_group(directory, alice, bob)
    new_name = f"{group['displayName']}-renamed"

    replaced = directory.put(
        f"/Groups/{group['id']}",
        json={"schemas": [GROUP_SCHEMA], "displayName": new_name, "members": members(carol)},
    )

    assert replaced.status_code == 200, replaced.text
    assert admin_panel_group(scim_instance, group["id"]) == (new_name, {carol["id"]})
    assert directory.get(f"/Groups/{group['id']}").json()["displayName"] == new_name


def test_delete_removes_the_group(scim_instance, directory):
    alice = provision(directory)
    group = create_group(directory, alice)

    deleted = directory.delete(f"/Groups/{group['id']}")

    assert deleted.status_code == 204, deleted.text
    assert admin_panel_group(scim_instance, group["id"]) is None
    assert directory.get(f"/Groups/{group['id']}").status_code == 404
    with scim_instance.client() as admin:
        assert admin.get(f"/api/v1/users/{alice['id']}").json()["groups"] == []


# ---------------------------------------------------------------- removing a provisioned person


def test_deleting_a_provisioned_person_removes_the_account_and_its_memberships(
    scim_instance, directory
):
    alice, bob = provision(directory), provision(directory)
    group = create_group(directory, alice, bob)

    deleted = directory.delete(f"/Users/{alice['id']}")

    assert deleted.status_code == 204, deleted.text
    assert directory.get(f"/Users/{alice['id']}").status_code == 404
    with scim_instance.client() as admin:
        assert admin.get(f"/api/v1/users/{alice['id']}").status_code == 400
    assert admin_panel_group(scim_instance, group["id"])[1] == {bob["id"]}
    group_members = directory.get(f"/Groups/{group['id']}").json()["members"]
    assert [member["value"] for member in group_members] == [bob["id"]]


def test_a_deleted_address_can_be_provisioned_again(directory):
    person = provision(directory)
    assert directory.delete(f"/Users/{person['id']}").status_code == 204

    again = directory.post(
        "/Users",
        json={
            "schemas": [USER_SCHEMA],
            "userName": person["userName"],
            "externalId": f"{person['externalId']}-again",
            "displayName": "Returning Person",
            "emails": [{"value": person["userName"], "primary": True}],
            "active": True,
        },
    )

    assert again.status_code == 201, again.text
    assert again.json()["id"] != person["id"]


def test_deleting_someone_unknown_is_a_404(directory):
    assert directory.delete(f"/Users/{uuid.uuid4()}").status_code == 404
