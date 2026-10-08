"""Regression: a folder its owner had shared never came back in the owner's shared folders.

Commit ce18eca34 lists folders shared by the user next to those shared with them: the owner
finds a folder they shared, and its sub-folders, under their own name and with write access,
however little the grant gives the others, which is what lets the owner's sidebar mark the folder
as shared. A folder the owner keeps to themselves, or stops sharing, stays out of the list, and a
member the folder is shared with still gets only what the grant gives.

Discriminates: passes on dev 87a937459. In a backend copy with ce18eca34 reverted the two tests
for the owner's shared folder and its sub-folder go red (the owner's list leaves them out).
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import make_group

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

FOLDER_SHARING = {"sharing": {"folders": True}}


@pytest.fixture
def crew(make_user, admin):
    """An owner and a member in one group allowed to share folders, and that group's id."""
    owner, member = make_user(), make_user()
    return owner, member, make_group(admin, [owner, member], FOLDER_SHARING)


def create_folder(owner, parent_id: str | None = None) -> str:
    form = {"name": f"Harbour {uuid.uuid4().hex[:6]}"}
    if parent_id:
        form["parent_id"] = parent_id
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json=form)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def share(owner, folder_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        updated = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
    assert updated.status_code == 200, updated.text


def read_grant(group_id: str) -> list[dict]:
    return [{"principal_type": "group", "principal_id": group_id, "permission": "read"}]


def shared_list(actor) -> dict[str, dict]:
    with actor.client() as client:
        listed = client.get("/api/v1/folders/shared")
    assert listed.status_code == 200, listed.text
    return {folder["id"]: folder for folder in listed.json()}


def test_the_owner_finds_a_folder_they_shared_under_their_name_with_write_access(crew):
    owner, _, group_id = crew
    folder_id = create_folder(owner)
    share(owner, folder_id, read_grant(group_id))

    listed = shared_list(owner)

    assert folder_id in listed, "the owner's shared folder is missing from their shared folders"
    assert listed[folder_id]["user_id"] == owner.id
    assert listed[folder_id]["owner_name"] == owner.name
    assert listed[folder_id]["permission"] == "write"


def test_the_owner_finds_the_sub_folders_of_a_folder_they_shared(crew):
    owner, _, group_id = crew
    folder_id = create_folder(owner)
    child_id = create_folder(owner, parent_id=folder_id)
    share(owner, folder_id, read_grant(group_id))

    listed = shared_list(owner)

    assert child_id in listed, "the sub-folder of the owner's shared folder is missing"
    assert listed[child_id]["parent_id"] == folder_id
    assert listed[child_id]["permission"] == "write"


def test_a_folder_the_owner_keeps_to_themselves_is_not_listed(crew):
    owner, member, group_id = crew
    kept = create_folder(owner)
    shared = create_folder(owner)
    share(owner, shared, read_grant(group_id))

    assert kept not in shared_list(owner)
    assert kept not in shared_list(member)


def test_a_folder_the_owner_stops_sharing_leaves_the_list(crew):
    owner, member, group_id = crew
    folder_id = create_folder(owner)
    share(owner, folder_id, read_grant(group_id))
    share(owner, folder_id, [])

    assert folder_id not in shared_list(owner)
    assert folder_id not in shared_list(member)


def test_a_member_still_gets_only_what_the_grant_gives(crew):
    owner, member, group_id = crew
    folder_id = create_folder(owner)
    child_id = create_folder(owner, parent_id=folder_id)
    share(owner, folder_id, read_grant(group_id))

    listed = shared_list(member)

    assert listed[folder_id]["permission"] == "read"
    assert listed[folder_id]["owner_name"] == owner.name
    assert listed[child_id]["permission"] == "read"
