"""Journey: what the model's skill tools can read and change in a chat, for each account.

In a chat the model may load a skill (`view_skill`), read one of its supporting files
(`read_skill_file`), save file changes (`update_skill_files`) and write a new skill
(`create_skill`), always as the account it is chatting with. It reads a skill's files only when
that account may read the skill, directly, through a group or because the skill is public, and
saves changes only with write access. A stranger's chat gets an error and none of the skill's
text, neither in the tool result nor anywhere else in what the model is sent, and a reader
dropped from the group that held the grant loses access in the next chat. A switched-off skill
is still open to those who may edit it. Only accounts with the workspace skills permission are
offered `create_skill`, and what it writes is private. A path that steps out of the skill or
into another one finds nothing, and the skill list the model sees names only skills the
account may read.

Discriminates: passes on dev 178de3666. In a backend copy, making the skills router's shared
access check always pass turned the stranger rows, `view_skill` and the dropped reader red (the
stranger's model got the file); dropping its grant lookup turned the public skill and the path
tests red; asking it for `read` whatever the permission turned the reader's
`update_skill_files` row and the switched-off skill test red; dropping the switched-off check
from `read_skill_file` turned the switched-off skill test red; letting `read_skill_file` follow
a path into another skill turned the path test red; offering `create_skill` to every account
turned the offer test red; listing every skill in the chat whatever the account may read
turned the skill list test red; and letting every admin through regardless of the admin access
setting turned the admin test red.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import pytest

from harness import upstream as reply
from harness.access import grant, make_group
from harness.actors import Actor, admin_of, create_user
from harness.chat import ask
from harness.tool_calls import offered_tools, run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SKILLS = "/api/v1/skills"
CHECKLIST = "references/checklist.md"


@dataclass
class Shared:
    owner: Actor
    reader: Actor
    writer: Actor
    stranger: Actor
    skill_id: str
    secret: str


def _create(owner: Actor, secret: str, grants: list[dict] | None = None) -> str:
    suffix = uuid.uuid4().hex[:8]
    body = {
        "id": f"harbour-{suffix}",
        "name": f"Harbour rules {suffix}",
        "description": f"Harbour rules {suffix}.",
        "files": [
            {"path": "SKILL.md", "content": f"Follow the harbour rules {suffix}."},
            {"path": CHECKLIST, "content": secret},
        ],
    }
    with owner.client() as client:
        created = client.post(f"{SKILLS}/create", json=body)
        assert created.status_code == 200, created.text
        if grants:
            shared = client.post(
                f"{SKILLS}/id/{body['id']}/access/update", json={"access_grants": grants}
            )
            assert shared.status_code == 200, shared.text
    return body["id"]


def _owner(admin: Actor, account: Actor) -> Actor:
    permissions = {"workspace": {"skills": True}, "sharing": {"public_skills": True}}
    make_group(admin, [account], permissions)
    return account


def _shared(admin: Actor, make_user, via: str = "user") -> Shared:
    owner = _owner(admin, make_user())
    reader, writer, stranger = make_user(), make_user(), make_user()
    if via == "group":
        reader_id, writer_id = make_group(admin, [reader]), make_group(admin, [writer])
    else:
        reader_id, writer_id = reader.id, writer.id
    secret = f"moor at berth {uuid.uuid4().hex[:8]}"
    skill_id = _create(
        owner,
        secret,
        [
            grant(via, reader_id, "read"),
            grant(via, writer_id, "read"),
            grant(via, writer_id, "write"),
        ],
    )
    return Shared(owner, reader, writer, stranger, skill_id, secret)


def _call(actor: Actor, upstream, tool: str, **arguments) -> tuple[dict, str]:
    """The tool's result, and everything the model was sent after the call."""
    with actor.client() as client:
        result = run_tool(client, upstream, tool, arguments)
    return json.loads(result), json.dumps(upstream.chat_requests()[-1])


def _version(owner: Actor, skill_id: str) -> str:
    with owner.client() as client:
        return client.get(f"{SKILLS}/id/{skill_id}").json()["version_id"]


def _system_prompt(actor: Actor, upstream, content: str) -> str:
    """The system prompt the model was sent for one message of `actor`'s."""
    upstream.queue(reply.text("done"))
    with actor.client() as client:
        ask(client, content)
    request = upstream.chat_requests()[-1]
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


@pytest.mark.parametrize("via", ["user", "group"])
def test_a_skill_file_reaches_the_model_only_for_those_who_may_read_it(
    via, admin, make_user, upstream
):
    shared = _shared(admin, make_user, via)
    actors = {
        "owner": shared.owner,
        "reader": shared.reader,
        "writer": shared.writer,
        "admin": admin,
        "stranger": shared.stranger,
    }

    answered = {}
    for role, actor in actors.items():
        result, sent = _call(actor, upstream, "read_skill_file", id=shared.skill_id, path=CHECKLIST)
        answered[role] = result.get("content", result.get("error"))
        if role == "stranger":
            assert shared.secret not in sent, "the stranger's model was sent the file"

    assert answered == {
        "owner": shared.secret,
        "reader": shared.secret,
        "writer": shared.secret,
        "admin": shared.secret,
        "stranger": "Access denied",
    }


def test_view_skill_lists_the_files_only_for_those_who_may_read_it(admin, make_user, upstream):
    shared = _shared(admin, make_user)

    loaded, _ = _call(shared.reader, upstream, "view_skill", id=shared.skill_id)
    refused, sent = _call(shared.stranger, upstream, "view_skill", id=shared.skill_id)

    assert [entry["path"] for entry in loaded["files"]] == ["SKILL.md", CHECKLIST], loaded
    assert loaded["version_id"] == _version(shared.owner, shared.skill_id)
    assert refused == {"error": "Access denied"}
    assert CHECKLIST not in sent and "Follow the harbour rules" not in sent


def test_a_public_skill_file_reaches_anyone(admin, make_user, upstream):
    owner = _owner(admin, make_user())
    secret = f"public berth {uuid.uuid4().hex[:8]}"
    skill_id = _create(owner, secret, [grant("user", "*", "read")])

    result, _ = _call(make_user(), upstream, "read_skill_file", id=skill_id, path=CHECKLIST)

    assert result.get("content") == secret, result


def test_a_reader_dropped_from_the_group_can_no_longer_read_the_file(admin, make_user, upstream):
    owner, reader = _owner(admin, make_user()), make_user()
    group_id = make_group(admin, [reader])
    secret = f"group berth {uuid.uuid4().hex[:8]}"
    skill_id = _create(owner, secret, [grant("group", group_id, "read")])

    before, _ = _call(reader, upstream, "read_skill_file", id=skill_id, path=CHECKLIST)
    with admin.client() as client:
        client.post(
            f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [reader.id]}
        ).raise_for_status()
    after, sent = _call(reader, upstream, "read_skill_file", id=skill_id, path=CHECKLIST)

    assert before.get("content") == secret, before
    assert after == {"error": "Access denied"} and secret not in sent


def test_a_switched_off_skill_stays_open_to_those_who_may_edit_it(admin, make_user, upstream):
    shared = _shared(admin, make_user)
    with shared.owner.client() as client:
        toggled = client.post(f"{SKILLS}/id/{shared.skill_id}/toggle")
    assert toggled.status_code == 200 and toggled.json()["is_active"] is False, toggled.text

    by_reader, sent = _call(
        shared.reader, upstream, "read_skill_file", id=shared.skill_id, path=CHECKLIST
    )
    by_writer, _ = _call(
        shared.writer, upstream, "read_skill_file", id=shared.skill_id, path=CHECKLIST
    )

    assert by_reader == {"error": "Access denied"} and shared.secret not in sent
    assert by_writer.get("content") == shared.secret, by_writer


def test_only_write_access_saves_file_changes(admin, make_user, upstream):
    shared = _shared(admin, make_user)
    operations = [{"op": "put", "path": "references/tides.md", "content": "high tide at noon"}]
    before = _version(shared.owner, shared.skill_id)

    refused = {
        role: _call(
            actor, upstream, "update_skill_files", id=shared.skill_id, operations=operations
        )[0]
        for role, actor in {"reader": shared.reader, "stranger": shared.stranger}.items()
    }
    unchanged = _version(shared.owner, shared.skill_id)
    saved, _ = _call(
        shared.writer, upstream, "update_skill_files", id=shared.skill_id, operations=operations
    )

    assert refused == {"reader": {"error": "Access denied"}, "stranger": {"error": "Access denied"}}
    assert unchanged == before
    assert saved["version_id"] != before, saved
    with shared.owner.client() as client:
        tides = client.get(
            f"{SKILLS}/id/{shared.skill_id}/files/content",
            params={"version_id": saved["version_id"], "path": "references/tides.md"},
        )
    assert tides.text == "high tide at noon"


def test_create_skill_is_offered_only_with_the_skills_permission(admin, make_user, upstream):
    without, author = make_user(), _owner(admin, make_user())

    with without.client() as client:
        offered_without = offered_tools(client, upstream)
    with author.client() as client:
        offered_author = offered_tools(client, upstream)
    suffix = uuid.uuid4().hex[:8]
    created, _ = _call(
        author,
        upstream,
        "create_skill",
        id=f"tides-{suffix}",
        name=f"Tides {suffix}",
        content="---\nname: tides\ndescription: Read the tide table.\n---\nRead it.",
        files=[{"path": "references/table.md", "content": "noon"}],
    )

    assert "read_skill_file" in offered_without and "create_skill" not in offered_without
    assert "create_skill" in offered_author
    with author.client() as client:
        stored = client.get(f"{SKILLS}/id/{created['id']}").json()
    assert stored["user_id"] == author.id and stored["access_grants"] == [], stored
    with without.client() as client:
        assert client.get(f"{SKILLS}/id/{created['id']}").status_code == 401


def test_a_path_out_of_the_skill_finds_nothing(admin, make_user, upstream):
    shared = _shared(admin, make_user)
    private_secret = f"private berth {uuid.uuid4().hex[:8]}"
    private_id = _create(shared.owner, private_secret)
    paths = [
        f"../{private_id}/{CHECKLIST}",
        f"/{private_id}/{CHECKLIST}",
        f"{private_id}/{CHECKLIST}",
        f"references/../../{private_id}/{CHECKLIST}",
        f"/{CHECKLIST}",
        "references//checklist.md",
        "./references/checklist.md",
    ]

    for path in paths:
        result, sent = _call(
            shared.reader, upstream, "read_skill_file", id=shared.skill_id, path=path
        )
        assert result == {"error": "File not found"}, f"{path}: {result}"
        assert private_secret not in sent and shared.secret not in sent, path


def test_the_model_sees_only_the_skills_the_account_may_read(admin, make_user, upstream):
    shared = _shared(admin, make_user)
    private_id = _create(shared.owner, "not for strangers")

    reader_view = _system_prompt(shared.reader, upstream, "which skills are there?")
    stranger_view = _system_prompt(shared.stranger, upstream, "which skills are there?")
    stranger_mention = _system_prompt(shared.stranger, upstream, f"<${private_id}|rules> what now?")

    assert shared.skill_id in reader_view and private_id not in reader_view
    assert shared.skill_id not in stranger_view and private_id not in stranger_view
    names = [
        f"Harbour rules {skill_id.split('-', 1)[1]}" for skill_id in (shared.skill_id, private_id)
    ]
    assert not any(name in stranger_view + stranger_mention for name in names), names
    assert "not for strangers" not in stranger_mention


@pytest.mark.slow
def test_without_admin_access_to_workspace_content_the_admins_model_reads_nothing(instance_with):
    closed = instance_with({"BYPASS_ADMIN_ACCESS_CONTROL": "False"})
    admin = admin_of(closed)
    owner = _owner(admin, create_user(closed))
    secret = f"closed berth {uuid.uuid4().hex[:8]}"
    skill_id = _create(owner, secret)

    result, sent = _call(admin, closed.upstream, "read_skill_file", id=skill_id, path=CHECKLIST)
    own, _ = _call(owner, closed.upstream, "read_skill_file", id=skill_id, path=CHECKLIST)

    assert result == {"error": "Access denied"} and secret not in sent
    assert own.get("content") == secret, own
