"""Journey: who may list, read, change and look back on the files of a shared skill.

A user with the workspace skills permission writes a skill with a supporting file, changes the
file once (two versions), and shares the skill with a reader (read) and a writer (read and
write), directly or through a group. Listing the files, downloading one, reading the history, an
old version and the diffs between versions is open to everyone who may read the skill; adding,
renaming, deleting and uploading files, setting an old version as production and deleting an
old version need write. A stranger gets
none of it, and no refused answer carries the file's text. After every refused write the owner's
skill, its files and its history are as they were. A public skill opens the reads to everyone
and the writes to nobody; a reader removed from the group that held the grant loses it all; a
pending account is refused even with a grant of its own; and with admin access to workspace
content switched off the admin is a stranger too. A version id of a skill the caller cannot see
opens nothing through a skill the caller can, and a skill made under a deleted skill's id starts
with no versions of the old one.

Discriminates: passes on dev 178de3666. In a backend copy, making the skills router's shared access
check always pass turned the read rows, the public skill and the dropped reader red (the stranger
got 200 and the file's text); dropping the grant lookup from that check turned the reader rows and
the reader and writer tests red; asking it for `read` whatever the permission turned the set
production rows red; dropping the write check from the update handler turned the file edit rows
red; letting `get_verified_user` pass a pending account turned the pending test red; letting every
admin through regardless of the admin access setting turned the admin test red; serving the current
version whatever version is asked turned the reader's version test red; looking a history entry up
by its id alone turned the foreign version test red; and keeping the history of a deleted skill
turned the reused id test red. Retargeted for 24ee1cb16, which replaced restoring a version with
setting it as production and added deleting a version: on dev 206bf9723 a backend copy whose
version switch and version delete ask for `read` turned the set production and delete rows red, and
one whose version switch keeps the current version turned the writer's set production test red.
"""

from __future__ import annotations

import uuid
from typing import Callable

import httpx
import pytest

from harness.access import ROLES, Cast, Shareable, cast, create_shared, grant, make_group
from harness.actors import Actor, admin_of, create_user
from harness.skill_files import read

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SKILLS = "/api/v1/skills"
FIRST_TEXT = "first checklist: pack the rope"
SECOND_TEXT = "second checklist: pack the rope and the lamp"


def _skill() -> dict:
    suffix = uuid.uuid4().hex[:8]
    return {
        "id": f"field-kit-{suffix}",
        "name": f"Field kit {suffix}",
        "description": "Pack the field kit.",
        "files": [
            {"path": "SKILL.md", "content": "Pack the field kit from the checklist."},
            {"path": "references/checklist.md", "content": FIRST_TEXT},
        ],
    }


SKILL = Shareable(
    create_path=f"{SKILLS}/create",
    create_body=_skill,
    access_path=f"{SKILLS}/id/{{id}}/access/update",
    owner_permissions={"workspace": {"skills": True}, "sharing": {"public_skills": True}},
)


def _second_version(owner: Actor, skill_id: str) -> dict:
    """Changes the checklist once; the ids of both versions."""
    with owner.client() as client:
        current = client.get(f"{SKILLS}/id/{skill_id}").json()
        changed = client.post(
            f"{SKILLS}/id/{skill_id}/update",
            json={
                "id": skill_id,
                "name": current["name"],
                "expected_version_id": current["version_id"],
                "operations": [
                    {"op": "put", "path": "references/checklist.md", "content": SECOND_TEXT}
                ],
            },
        )
    assert changed.status_code == 200, changed.text
    return {"first": current["version_id"], "current": changed.json()["version_id"]}


def _file_edit(*operations: dict) -> Callable[[Actor, dict], dict]:
    def body(actor: Actor, fields: dict) -> dict:
        return {
            "id": fields["id"],
            "name": fields["name"],
            "expected_version_id": fields["current"],
            "operations": list(operations),
        }

    return body


def _set_production(actor: Actor, fields: dict) -> dict:
    return {"version_id": fields["first"], "expected_version_id": fields["current"]}


def _owner_view(client: httpx.Client, fields: dict) -> list[tuple[int, bytes]]:
    paths = [
        f"{SKILLS}/id/{fields['id']}",
        f"{SKILLS}/id/{fields['id']}/files",
        f"{SKILLS}/id/{fields['id']}/history",
    ]
    return [(answer.status_code, answer.content) for answer in map(client.get, paths)]


READ = {"owner", "reader", "writer", "admin"}
WRITE = {"owner", "writer", "admin"}
HISTORY = f"{SKILLS}/id/{{id}}/history"
CHECKLIST = "path=references/checklist.md"

# method, path, body, who gets through, the route's refusal code
MATRIX = [
    ("GET", f"{SKILLS}/id/{{id}}/files", None, READ, 403),
    ("GET", f"{SKILLS}/id/{{id}}/files?version_id={{first}}", None, READ, 403),
    (
        "GET",
        f"{SKILLS}/id/{{id}}/files/content?version_id={{first}}&{CHECKLIST}",
        None,
        READ,
        403,
    ),
    ("GET", HISTORY, None, READ, 403),
    ("GET", f"{HISTORY}/{{first}}", None, READ, 403),
    ("GET", f"{HISTORY}/diff?from_id={{first}}&to_id={{current}}", None, READ, 403),
    (
        "GET",
        f"{HISTORY}/diff/file?from_id={{first}}&to_id={{current}}&{CHECKLIST}",
        None,
        READ,
        403,
    ),
    ("POST", f"{SKILLS}/id/{{id}}/update/version", _set_production, WRITE, 403),
    ("DELETE", f"{HISTORY}/{{first}}", None, WRITE, 403),
    (
        "POST",
        f"{SKILLS}/id/{{id}}/update",
        _file_edit({"op": "put", "path": "references/new.md", "content": "added"}),
        WRITE,
        401,
    ),
    (
        "POST",
        f"{SKILLS}/id/{{id}}/update",
        _file_edit(
            {"op": "move", "path": "references/checklist.md", "destination": "notes/list.md"}
        ),
        WRITE,
        401,
    ),
    (
        "POST",
        f"{SKILLS}/id/{{id}}/update",
        _file_edit({"op": "delete", "path": "references"}),
        WRITE,
        401,
    ),
    (
        "POST",
        f"{SKILLS}/id/{{id}}/update",
        _file_edit(
            {
                "op": "put",
                "path": "assets/logo.png",
                "content": "iVBORw0KGgo=",
                "encoding": "base64",
            }
        ),
        WRITE,
        401,
    ),
]
ROW_IDS = [
    "list files",
    "list an old version",
    "download a file",
    "history",
    "an old version",
    "diff",
    "file diff",
    "set production",
    "delete a version",
    "add a file",
    "rename",
    "delete a folder",
    "upload",
]


def _send(actor: Actor, method: str, path: str, body, fields: dict) -> httpx.Response:
    payload = body(actor, fields) if callable(body) else body
    with actor.client() as client:
        return client.request(method, path.format(**fields), json=payload)


def _fresh(accounts: Cast) -> dict:
    skill_id = create_shared(accounts)
    fields = {"id": skill_id, **_second_version(accounts.owner, skill_id)}
    with accounts.owner.client() as client:
        fields["name"] = client.get(f"{SKILLS}/id/{skill_id}").json()["name"]
    return fields


def _check_row(accounts: Cast, actors: dict[str, Actor], method, path, body, allowed, refused):
    """Each actor's status on a fresh skill; a refusal carries no file text and changes nothing."""
    answered = {}
    for role, actor in actors.items():
        fields = _fresh(accounts)
        with accounts.owner.client() as owner_client:
            before = _owner_view(owner_client, fields)
        response = _send(actor, method, path, body, fields)
        with accounts.owner.client() as owner_client:
            after = _owner_view(owner_client, fields)
        answered[role] = response.status_code
        if role not in allowed:
            assert "checklist" not in response.text, f"{role} was answered {response.text}"
            assert after == before, f"a refused {role} changed the owner's skill"
    return answered


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize("method, path, body, allowed, refused", MATRIX, ids=ROW_IDS)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, allowed, refused, via, admin, make_user
):
    accounts = cast(SKILL, admin, make_user, via=via)
    actors = {role: accounts.actor(role) for role in ROLES}

    answered = _check_row(accounts, actors, method, path, body, allowed, refused)

    assert answered == {role: 200 if role in allowed else refused for role in ROLES}, (
        f"{method} {path} shared via {via}"
    )


def test_a_reader_sees_the_right_text_in_each_version(admin, make_user):
    accounts = cast(SKILL, admin, make_user)
    fields = _fresh(accounts)
    base = f"{SKILLS}/id/{fields['id']}"

    with accounts.reader.client() as client:
        old = client.get(
            f"{base}/files/content",
            params={"version_id": fields["first"], "path": "references/checklist.md"},
        )
        new = client.get(
            f"{base}/files/content",
            params={"version_id": fields["current"], "path": "references/checklist.md"},
        )
        diff = client.get(
            f"{base}/history/diff/file",
            params={
                "from_id": fields["first"],
                "to_id": fields["current"],
                "path": "references/checklist.md",
            },
        ).json()
        history = client.get(f"{base}/history").json()

    assert (old.text, new.text) == (FIRST_TEXT, SECOND_TEXT)
    assert f"-{FIRST_TEXT}" in diff["diff"] and f"+{SECOND_TEXT}" in diff["diff"], diff
    assert {entry["id"]: entry["parent_id"] for entry in history} == {
        fields["first"]: None,
        fields["current"]: fields["first"],
    }


def test_a_writers_switch_to_an_old_version_brings_its_files_back(admin, make_user):
    accounts = cast(SKILL, admin, make_user)
    fields = _fresh(accounts)
    base = f"{SKILLS}/id/{fields['id']}"

    with accounts.writer.client() as client:
        switched = client.post(f"{base}/update/version", json=_set_production(None, fields))
    assert switched.status_code == 200, switched.text
    live = read(accounts.owner, fields["id"], "references/checklist.md")
    with accounts.owner.client() as client:
        history = client.get(f"{base}/history").json()
        skill = client.get(f"{base}").json()

    assert live == FIRST_TEXT.encode()
    assert skill["version_id"] == fields["first"]
    assert {entry["id"] for entry in history} == {fields["first"], fields["current"]}, history
    assert len(skill["access_grants"]) == 3, "a version switch changed who the skill is shared with"


def test_a_public_skill_is_readable_by_anyone_and_writable_by_nobody(admin, make_user):
    accounts = cast(SKILL, admin, make_user)
    public = [grant("user", "*", "read")]
    passer_by = make_user()

    answered = {}
    for row, row_id in zip(MATRIX, ROW_IDS):
        method, path, body, allowed, refused = row
        with accounts.owner.client() as client:
            created = client.post(SKILL.create_path, json=_skill())
            skill_id = created.json()["id"]
            client.post(SKILL.access_path.format(id=skill_id), json={"access_grants": public})
            fields = {"id": skill_id, **_second_version(accounts.owner, skill_id)}
            fields["name"] = client.get(f"{SKILLS}/id/{skill_id}").json()["name"]
            before = _owner_view(client, fields)
        answer = _send(passer_by, method, path, body, fields)
        with accounts.owner.client() as client:
            after = _owner_view(client, fields)
        answered[row_id] = answer.status_code
        if "owner" in allowed and "reader" not in allowed:
            assert after == before, f"a passer-by's {row_id} changed a public skill"

    assert answered == {
        row_id: 200 if "reader" in row[3] else row[4] for row, row_id in zip(MATRIX, ROW_IDS)
    }


def test_a_reader_dropped_from_the_group_loses_the_files(admin, make_user):
    owner, reader = make_user(), make_user()
    make_group(admin, [owner], SKILL.owner_permissions)
    group_id = make_group(admin, [reader])
    with owner.client() as client:
        skill_id = client.post(SKILL.create_path, json=_skill()).json()["id"]
        shared = client.post(
            SKILL.access_path.format(id=skill_id),
            json={"access_grants": [grant("group", group_id, "read")]},
        )
    assert shared.status_code == 200, shared.text
    fields = {"id": skill_id, **_second_version(owner, skill_id), "name": ""}
    reads = [row for row in MATRIX if row[0] == "GET"]

    before = {path: _send(reader, "GET", path, None, fields).status_code for _, path, *_ in reads}
    with admin.client() as client:
        removed = client.post(
            f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [reader.id]}
        )
    assert removed.status_code == 200, removed.text
    after = {path: _send(reader, "GET", path, None, fields) for _, path, *_ in reads}

    assert set(before.values()) == {200}
    assert {path: answer.status_code for path, answer in after.items()} == {
        path: 403 for path in after
    }
    assert not any("checklist" in answer.text for answer in after.values())


def test_a_pending_account_is_refused_even_with_a_grant(admin, make_user):
    owner, pending = make_user(), make_user(role="pending")
    make_group(admin, [owner], SKILL.owner_permissions)
    with owner.client() as client:
        skill_id = client.post(SKILL.create_path, json=_skill()).json()["id"]
        client.post(
            SKILL.access_path.format(id=skill_id),
            json={
                "access_grants": [
                    grant("user", pending.id, "read"),
                    grant("user", pending.id, "write"),
                ]
            },
        ).raise_for_status()
        fields = {"id": skill_id, **_second_version(owner, skill_id)}
        fields["name"] = client.get(f"{SKILLS}/id/{skill_id}").json()["name"]
        before = _owner_view(client, fields)

    answers = {row_id: _send(pending, *row[:3], fields) for row, row_id in zip(MATRIX, ROW_IDS)}

    with owner.client() as client:
        assert _owner_view(client, fields) == before
    assert {row_id: answer.status_code for row_id, answer in answers.items()} == {
        row_id: 401 for row_id in ROW_IDS
    }
    assert not any("checklist" in answer.text for answer in answers.values())


def test_a_version_of_another_skill_opens_nothing(admin, make_user):
    accounts = cast(SKILL, admin, make_user)
    readable = _fresh(accounts)
    with accounts.owner.client() as client:
        private_id = client.post(SKILL.create_path, json=_skill()).json()["id"]
        private = _second_version(accounts.owner, private_id)
    foreign = private["first"]
    base = f"{SKILLS}/id/{readable['id']}"
    checklist = {"path": "references/checklist.md"}

    with accounts.reader.client() as client:
        reads = [
            client.get(f"{base}/files", params={"version_id": foreign}),
            client.get(f"{base}/files/content", params={"version_id": foreign, **checklist}),
            client.get(f"{base}/history/{foreign}"),
            client.get(f"{base}/history/diff", params={"from_id": foreign, "to_id": foreign}),
            client.get(
                f"{base}/history/diff/file",
                params={"from_id": foreign, "to_id": readable["current"], **checklist},
            ),
        ]
    with accounts.writer.client() as client:
        writes = [
            client.post(
                f"{base}/update/version",
                json={"version_id": foreign, "expected_version_id": readable["current"]},
            ),
            client.delete(f"{base}/history/{foreign}"),
        ]

    assert [answer.status_code for answer in [*reads, *writes]] == [404] * 7
    assert not any(FIRST_TEXT in answer.text for answer in [*reads, *writes])
    with accounts.owner.client() as client:
        assert client.get(f"{base}").json()["version_id"] == readable["current"]
        private_history = client.get(f"{SKILLS}/id/{private_id}/history").json()
    assert foreign in {entry["id"] for entry in private_history}


@pytest.mark.slow
def test_without_admin_access_to_workspace_content_the_admin_is_a_stranger(instance_with):
    closed = instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "False"})
    admin = admin_of(closed)
    accounts = cast(SKILL, admin, lambda: create_user(closed))
    actors = {"admin": admin, "reader": accounts.reader}

    answered = {}
    for row, row_id in zip(MATRIX, ROW_IDS):
        answered[row_id] = _check_row(accounts, actors, *row)

    assert answered == {
        row_id: {"admin": row[4], "reader": 200 if "reader" in row[3] else row[4]}
        for row, row_id in zip(MATRIX, ROW_IDS)
    }


def test_a_skill_made_under_a_deleted_skills_id_inherits_no_versions(admin, make_user):
    owner, newcomer = make_user(), make_user()
    make_group(admin, [owner, newcomer], SKILL.owner_permissions)
    body = _skill()
    with owner.client() as client:
        client.post(SKILL.create_path, json=body).raise_for_status()
    old = _second_version(owner, body["id"])
    with owner.client() as client:
        assert client.delete(f"{SKILLS}/id/{body['id']}/delete").json() is True

    with newcomer.client() as client:
        remade = client.post(SKILL.create_path, json={**_skill(), "id": body["id"]})
        assert remade.status_code == 200, remade.text
        history = client.get(f"{SKILLS}/id/{body['id']}/history").json()
        old_files = client.get(
            f"{SKILLS}/id/{body['id']}/files/content",
            params={"version_id": old["current"], "path": "references/checklist.md"},
        )

    assert [entry["id"] for entry in history] == [remade.json()["version_id"]]
    assert old_files.status_code == 404 and SECOND_TEXT not in old_files.text
