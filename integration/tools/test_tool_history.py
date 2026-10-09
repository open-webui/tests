"""Journey: a workspace tool's version history, as the tool editor keeps and restores it.

Creating a tool records its first version. Every save that changes its name, description or
source adds a version with the message typed beside Save and makes it Production; a save that
changes nothing adds none. Comparing an older version to Production shows the source change as
a diff and the renamed fields side by side. Setting an older version as Production puts its
source back, and that source is what the tool runs from then on. The Production version cannot
be deleted; an older one can. Only the owner, a writer and the admin see or touch the versions;
a reader and a stranger are refused and nothing changes. Bringing back older source is a source
change, so a writer without the workspace tools permission may restore a renamed version but
not one with different code.

Discriminates: passes on dev 206bf9723 (3 of 3). In backend copies: creating a tool without
storing its first version turns `test_creating_a_tool_records_its_first_version` red (and the
Set as Production rows of the access test, which then have no older version); a save that
changes the tool adding no version turns `test_a_save_adds_a_version_and_makes_it_production`
and `test_comparing_an_old_version_to_production_shows_what_changed` red; Set as Production
keeping the live source and the history routes asking for `read` instead of `write` turn
`test_setting_an_old_version_as_production_runs_its_source_again` and the reader rows of
`test_only_accounts_that_may_change_the_tool_reach_its_versions` red; dropping the code change
check when a version is set as Production turns only
`test_a_writer_without_the_tools_permission_cannot_restore_older_source` red (the writer gets
200 and the old source is back); every save adding a version and restoring never being allowed
to change code turn `test_a_save_that_changes_nothing_adds_no_version` and
`test_a_writer_with_the_tools_permission_restores_older_source` red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import Shareable, attempts, cast, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_BUILDER = {"workspace": {"tools": True}}
SOURCE = '''class Tools:
    def shout(self, text: str) -> str:
        """Say the text back.

        :param text: what to say
        """
        return text
'''
NEW_SOURCE = SOURCE.replace("def shout", "def whisper").replace(
    "return text", "return text.lower()"
)


def _tool_form(tool_id: str, name: str = "Echo", content: str = SOURCE, **fields) -> dict:
    return {
        "id": tool_id,
        "name": name,
        "content": content,
        "meta": {"description": "version history"},
        **fields,
    }


TOOL = Shareable(
    create_path="/api/v1/tools/create",
    create_body=lambda: _tool_form(f"history_{uuid.uuid4().hex[:8]}"),
    access_path="/api/v1/tools/id/{id}/access/update",
    owner_permissions=TOOL_BUILDER,
)


@pytest.fixture
def toolsmith(admin, make_user) -> Actor:
    account = make_user()
    make_group(admin, [account], TOOL_BUILDER)
    return account


def _create(client: httpx.Client) -> dict:
    created = client.post(
        "/api/v1/tools/create", json=_tool_form(f"history_{uuid.uuid4().hex[:8]}")
    )
    assert created.status_code == 200, created.text
    return created.json()


def _save(client: httpx.Client, tool_id: str, **fields) -> dict:
    saved = client.post(f"/api/v1/tools/id/{tool_id}/update", json=_tool_form(tool_id, **fields))
    assert saved.status_code == 200, saved.text
    return saved.json()


def _versions(client: httpx.Client, tool_id: str) -> list[dict]:
    listed = client.get(f"/api/v1/tools/id/{tool_id}/history")
    assert listed.status_code == 200, listed.text
    return listed.json()


def _tool(client: httpx.Client, tool_id: str) -> dict:
    read = client.get(f"/api/v1/tools/id/{tool_id}")
    assert read.status_code == 200, read.text
    return read.json()


def _spec_names(tool: dict) -> list[str]:
    return [spec["name"] for spec in tool["specs"]]


def test_creating_a_tool_records_its_first_version(toolsmith):
    with toolsmith.client() as client:
        tool = _create(client)
        versions = _versions(client, tool["id"])
        first = client.get(f"/api/v1/tools/id/{tool['id']}/history/{tool['version_id']}").json()

    assert [version["id"] for version in versions] == [tool["version_id"]]
    assert versions[0]["user"] == {"name": toolsmith.name}
    assert first["snapshot"]["content"] == SOURCE


def test_a_save_adds_a_version_and_makes_it_production(toolsmith):
    with toolsmith.client() as client:
        tool = _create(client)
        saved = _save(client, tool["id"], content=NEW_SOURCE, commit_message="whisper instead")
        versions = _versions(client, tool["id"])
        latest = client.get(f"/api/v1/tools/id/{tool['id']}/history/{saved['version_id']}")

    assert saved["version_id"] != tool["version_id"]
    assert {version["id"] for version in versions} == {tool["version_id"], saved["version_id"]}
    assert latest.status_code == 200, latest.text
    assert latest.json()["commit_message"] == "whisper instead"
    assert latest.json()["parent_id"] == tool["version_id"]
    assert latest.json()["snapshot"]["content"] == NEW_SOURCE


def test_a_save_that_changes_nothing_adds_no_version(toolsmith):
    with toolsmith.client() as client:
        tool = _create(client)
        saved = _save(client, tool["id"], commit_message="nothing changed")
        versions = _versions(client, tool["id"])

    assert [version["id"] for version in versions] == [tool["version_id"]]
    assert saved["version_id"] == tool["version_id"]


def test_comparing_an_old_version_to_production_shows_what_changed(toolsmith):
    with toolsmith.client() as client:
        tool = _create(client)
        saved = _save(client, tool["id"], name="Whisper", content=NEW_SOURCE)
        compared = client.get(
            f"/api/v1/tools/id/{tool['id']}/history/diff",
            params={"from_id": tool["version_id"], "to_id": saved["version_id"]},
        )

    assert compared.status_code == 200, compared.text
    diff = compared.json()
    removed = [line for line in diff["content_diff"].splitlines() if line.startswith("-    ")]
    added = [line for line in diff["content_diff"].splitlines() if line.startswith("+    ")]
    assert "-    def shout(self, text: str) -> str:" in removed
    assert "+    def whisper(self, text: str) -> str:" in added
    assert diff["metadata"]["name"] == {"before": "Echo", "after": "Whisper"}
    assert "meta" not in diff["metadata"]


def test_setting_an_old_version_as_production_runs_its_source_again(toolsmith):
    with toolsmith.client() as client:
        tool = _create(client)
        _save(client, tool["id"], name="Whisper", content=NEW_SOURCE)
        assert _spec_names(_tool(client, tool["id"])) == ["whisper"]
        restored = client.post(
            f"/api/v1/tools/id/{tool['id']}/update/version",
            json={"version_id": tool["version_id"]},
        )
        assert restored.status_code == 200, restored.text
        current = _tool(client, tool["id"])
        versions = _versions(client, tool["id"])

    assert (current["name"], current["content"]) == ("Echo", SOURCE)
    assert _spec_names(current) == ["shout"]
    assert current["version_id"] == tool["version_id"]
    assert len(versions) == 2


def test_the_production_version_cannot_be_deleted(toolsmith):
    with toolsmith.client() as client:
        tool = _create(client)
        saved = _save(client, tool["id"], content=NEW_SOURCE)
        refused = client.delete(f"/api/v1/tools/id/{tool['id']}/history/{saved['version_id']}")
        deleted = client.delete(f"/api/v1/tools/id/{tool['id']}/history/{tool['version_id']}")
        remaining = [version["id"] for version in _versions(client, tool["id"])]

    assert refused.status_code == 400, refused.text
    assert deleted.status_code == 200, deleted.text
    assert remaining == [saved["version_id"]]


def _renamed_version(owner: Actor, tool_id: str) -> dict:
    """A second save that only renames, so the first version is an older one with the same code."""
    with owner.client() as client:
        first = _tool(client, tool_id)["version_id"]
        _save(client, tool_id, name="Renamed")
    return {"history_id": first}


def _owners_view(owner_client: httpx.Client, fields: dict) -> tuple:
    tool = _tool(owner_client, fields["id"])
    versions = sorted(version["id"] for version in _versions(owner_client, fields["id"]))
    return tool["name"], tool["content"], tool["version_id"], versions


REFUSED, ALLOWED = 401, 200
MAY_CHANGE = {
    "owner": ALLOWED,
    "stranger": REFUSED,
    "reader": REFUSED,
    "writer": ALLOWED,
    "admin": ALLOWED,
}

# method, path, body, whether an allowed request changes the owner's view
ROUTES = [
    ("GET", "/api/v1/tools/id/{id}/history", None, False),
    ("GET", "/api/v1/tools/id/{id}/history/{history_id}", None, False),
    (
        "GET",
        "/api/v1/tools/id/{id}/history/diff?from_id={history_id}&to_id={history_id}",
        None,
        False,
    ),
    ("DELETE", "/api/v1/tools/id/{id}/history/{history_id}", None, True),
    (
        "POST",
        "/api/v1/tools/id/{id}/update/version",
        lambda actor, fields: {"version_id": fields["history_id"]},
        True,
    ),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, changes",
    ROUTES,
    ids=["list", "open", "compare", "delete", "set as production"],
)
def test_only_accounts_that_may_change_the_tool_reach_its_versions(
    admin, make_user, via, method, path, body, changes
):
    accounts = cast(TOOL, admin, make_user, via=via)

    answered = attempts(
        accounts, method, path, body=body, setup=_renamed_version, look=_owners_view
    )

    assert {role: attempt.status for role, attempt in answered.items()} == MAY_CHANGE
    for role, attempt in answered.items():
        if MAY_CHANGE[role] == REFUSED or not changes:
            assert attempt.after == attempt.before, role
        else:
            assert attempt.after != attempt.before, role


def _shared_for_writing(owner: Actor, writer: Actor) -> dict:
    """The owner's tool with a second version in new code, shared with the writer for writing."""
    with owner.client() as client:
        tool = _create(client)
        _save(client, tool["id"], content=NEW_SOURCE)
        shared = client.post(
            f"/api/v1/tools/id/{tool['id']}/access/update",
            json={
                "access_grants": [
                    {"principal_type": "user", "principal_id": writer.id, "permission": "read"},
                    {"principal_type": "user", "principal_id": writer.id, "permission": "write"},
                ]
            },
        )
    assert shared.status_code == 200, shared.text
    return tool


def test_a_writer_without_the_tools_permission_cannot_restore_older_source(toolsmith, make_user):
    writer = make_user()
    tool = _shared_for_writing(toolsmith, writer)

    with writer.client() as client:
        refused = client.post(
            f"/api/v1/tools/id/{tool['id']}/update/version",
            json={"version_id": tool["version_id"]},
        )
    with toolsmith.client() as client:
        current = _tool(client, tool["id"])

    assert refused.status_code == 401, refused.text
    assert current["content"] == NEW_SOURCE
    assert _spec_names(current) == ["whisper"]


def test_a_writer_with_the_tools_permission_restores_older_source(admin, toolsmith, make_user):
    writer = make_user()
    make_group(admin, [writer], TOOL_BUILDER)
    tool = _shared_for_writing(toolsmith, writer)

    with writer.client() as client:
        restored = client.post(
            f"/api/v1/tools/id/{tool['id']}/update/version",
            json={"version_id": tool["version_id"]},
        )
    with toolsmith.client() as client:
        current = _tool(client, tool["id"])

    assert restored.status_code == 200, restored.text
    assert current["content"] == SOURCE
    assert _spec_names(current) == ["shout"]
