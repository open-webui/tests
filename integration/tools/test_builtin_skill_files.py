"""Journey: the model reads a skill's files, edits them and writes a new skill in a chat.

With the Skills builtin tool on, a chat offers `view_skill`, `read_skill_file` and
`update_skill_files`, and `create_skill` to an account allowed to write skills. `view_skill`
returns the skill's SKILL.md with the list of its files and the version it read;
`read_skill_file` returns one file in pages of `max_chars` with the offset of the next page, and
for a binary file only its size and a link to it in the skill editor. A file read after
`view_skill` in the same turn comes from the version `view_skill` loaded, even if the skill was
saved again in between. `update_skill_files` saves put, move and delete operations as a new
version with the commit message and keeps the files it did not touch; it refuses to remove
SKILL.md and leaves the skill as it was. `create_skill` writes a private skill with SKILL.md and
its supporting files, taking the description from the front matter. `/skills:create` in a chat
with earlier messages asks the model to save the skill with `create_skill`, with no terminal
selected.

Discriminates: passes on dev 178de3666. Each of these edits to a backend copy turns its test red:
a chat offered `view_skill` alone (offered tools), `view_skill` without the file list (view),
`read_skill_file` ignoring `offset` (paging), pasting a binary file as base64 (binary), reading a
missing file as empty (missing file), reading the current version instead of the one `view_skill`
loaded (pinning), file operations that start from SKILL.md alone (update), a SKILL.md delete that
is skipped without a word (SKILL.md), `create_skill` dropping the supporting files (create) and
skill authoring that still needs a terminal (`/skills:create`).
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply
from harness.skill_files import (
    create,
    current,
    delete_skills_of,
    files,
    history,
    new_id,
    read,
    save,
    skill_md,
)
from harness.tool_calls import offered_tools, run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    delete_skills_of(account)


SKILL_TOOLS = {"view_skill", "read_skill_file", "update_skill_files", "create_skill"}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _call(actor, upstream, tool: str, **arguments) -> dict:
    with actor.client() as client:
        return json.loads(run_tool(client, upstream, tool, arguments))


def _knots(owner) -> dict:
    skill_id = new_id("knots")
    skill = create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Use the knot book."),
            "references/bowline.md": "Rabbit out of the hole, round the tree, back down.\n",
            "references/hitch.md": "Two turns and a hitch.\n",
        },
        skill_id=skill_id,
    )
    saved = save(
        owner,
        skill,
        [
            {
                "op": "put",
                "path": "assets/knot.png",
                "content": base64.b64encode(PNG).decode(),
                "encoding": "base64",
            }
        ],
    )
    assert saved.status_code == 200, saved.text
    return saved.json()


def test_a_chat_offers_the_skill_file_tools(owner, upstream):
    with owner.client() as client:
        offered = offered_tools(client, upstream)

    assert SKILL_TOOLS <= offered, sorted(offered)


def test_view_skill_returns_the_instructions_and_the_file_list(owner, upstream):
    skill = _knots(owner)

    viewed = _call(owner, upstream, "view_skill", id=skill["id"])

    assert viewed["id"] == skill["id"]
    assert viewed["version_id"] == skill["version_id"]
    assert viewed["content"] == skill_md(skill["id"], "Use the knot book.")
    assert {entry["path"]: entry["size"] for entry in viewed["files"]} == files(owner, skill["id"])


def test_read_skill_file_pages_through_a_file(owner, upstream):
    skill = _knots(owner)
    text = "Rabbit out of the hole, round the tree, back down.\n"

    first = _call(
        owner,
        upstream,
        "read_skill_file",
        id=skill["id"],
        path="references/bowline.md",
        max_chars=20,
    )
    rest = _call(
        owner,
        upstream,
        "read_skill_file",
        id=skill["id"],
        path="references/bowline.md",
        offset=first["next_offset"],
        max_chars=1000,
    )

    assert first["content"] == text[:20]
    assert first["next_offset"] == 20
    assert rest["content"] == text[20:]
    assert rest["next_offset"] is None


def test_a_binary_file_is_described_and_linked_not_pasted(owner, upstream):
    skill = _knots(owner)

    described = _call(owner, upstream, "read_skill_file", id=skill["id"], path="assets/knot.png")

    assert "content" not in described, described
    assert described["size"] == len(PNG)
    assert described["url"].startswith(f"/workspace/skills/edit?id={skill['id']}")
    assert "path=assets%2Fknot.png" in described["url"]


def test_reading_a_missing_file_says_so(owner, upstream):
    skill = _knots(owner)

    missing = _call(owner, upstream, "read_skill_file", id=skill["id"], path="references/sheet.md")

    assert missing == {"error": "File not found"}


def test_a_file_read_after_view_skill_comes_from_the_version_it_loaded(owner, upstream):
    skill = _knots(owner)
    upstream.queue(
        reply.tool_call("view_skill", {"id": skill["id"]}, call_id="call_view"),
        # held back so the skill can be saved again between the two tool calls
        reply.tool_call(
            "read_skill_file",
            {"id": skill["id"], "path": "references/hitch.md"},
            call_id="call_read",
            delay=4,
        ),
        reply.text("done"),
    )

    with owner.client() as client:
        turn = send_message(client, "how do I tie a hitch?")
        deadline = time.monotonic() + 30
        while len(upstream.chat_requests()) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert len(upstream.chat_requests()) >= 2, "view_skill never ran"
        resaved = save(
            owner, skill, [{"op": "put", "path": "references/hitch.md", "content": "Changed.\n"}]
        )
        assert resaved.status_code == 200, resaved.text
        wait_for_reply(client, turn)

    sent_back = upstream.chat_requests()[-1]["messages"]
    results = {
        entry["tool_call_id"]: json.loads(entry["content"])
        for entry in sent_back
        if entry["role"] == "tool"
    }
    assert results["call_read"]["version_id"] == skill["version_id"]
    assert results["call_read"]["content"] == "Two turns and a hitch.\n"


def test_update_skill_files_saves_a_version_and_keeps_untouched_files(owner, upstream):
    skill = _knots(owner)

    updated = _call(
        owner,
        upstream,
        "update_skill_files",
        id=skill["id"],
        operations=[
            {"op": "put", "path": "references/hitch.md", "content": "Three turns and a hitch.\n"},
            {"op": "move", "path": "references/bowline.md", "destination": "knots/bowline.md"},
        ],
        commit_message="Tighter hitch",
    )

    assert updated.get("version_id") not in (None, skill["version_id"]), updated
    assert current(owner, skill["id"])["version_id"] == updated["version_id"]
    assert set(files(owner, skill["id"])) == {
        "SKILL.md",
        "assets/knot.png",
        "knots/bowline.md",
        "references/hitch.md",
    }
    assert read(owner, skill["id"], "references/hitch.md") == b"Three turns and a hitch.\n"
    assert read(owner, skill["id"], "assets/knot.png") == PNG
    latest = next(
        entry for entry in history(owner, skill["id"]) if entry["id"] == updated["version_id"]
    )
    assert latest["commit_message"] == "Tighter hitch"
    assert latest["user_id"] == owner.id


def test_update_skill_files_will_not_remove_skill_md(owner, upstream):
    skill = _knots(owner)

    refused = _call(
        owner,
        upstream,
        "update_skill_files",
        id=skill["id"],
        operations=[{"op": "delete", "path": "SKILL.md"}],
    )

    assert "SKILL.md" in refused.get("error", ""), refused
    assert current(owner, skill["id"])["version_id"] == skill["version_id"]
    assert "SKILL.md" in files(owner, skill["id"])


def test_create_skill_writes_a_private_skill_with_its_files(owner, upstream):
    skill_id = new_id("sails")
    content = skill_md(skill_id, "Reef before the squall.", description="Handling sails in wind.")

    created = _call(
        owner,
        upstream,
        "create_skill",
        id=skill_id,
        name=f"Sails {skill_id}",
        content=content,
        files=[{"path": "references/reefing.md", "content": "Ease the main first.\n"}],
        commit_message="First draft",
    )

    assert created["id"] == skill_id, created
    assert created["url"] == f"/workspace/skills/edit?id={skill_id}"
    stored = current(owner, skill_id)
    assert stored["description"] == "Handling sails in wind."
    assert stored["access_grants"] == []
    assert read(owner, skill_id, "SKILL.md").decode() == content
    assert read(owner, skill_id, "references/reefing.md") == b"Ease the main first.\n"
    assert history(owner, skill_id)[0]["commit_message"] == "First draft"


def test_skills_create_asks_the_model_to_save_with_create_skill_without_a_terminal(owner, upstream):
    upstream.queue(reply.text("Sure."), reply.text("Saved."))
    with owner.client() as client:
        first, _ = ask(client, "We just reefed the main in a squall.")
        history_so_far = [
            {"role": "user", "content": "We just reefed the main in a squall."},
            {"role": "assistant", "content": "Sure."},
        ]
        ask(
            client,
            "/skills:create a skill for reefing",
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=history_so_far,
        )

    last_user = [
        entry for entry in upstream.chat_requests()[-1]["messages"] if entry["role"] == "user"
    ][-1]
    prompt = str(last_user["content"])
    assert "create_skill" in prompt, prompt
    assert "a skill for reefing" in prompt
    assert "Open Terminal" not in prompt, prompt
