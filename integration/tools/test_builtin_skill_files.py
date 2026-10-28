"""Journey: the model reads, searches and edits a skill's files and writes a new skill in a chat.

With the Skills builtin tool on, a chat that resolves one of the account's skills offers
`view_skill`, `read_skill_file`, `grep_skill_files` and `update_skill_files`, and `create_skill`
to an account allowed to write skills. `view_skill` returns the skill's SKILL.md with the list of
its files; `read_skill_file` returns one file in pages of `max_chars` with the offset of the next
page, and for a binary file only its size and a link to it in the skill editor. A file read after
`view_skill` in the same turn comes from the version `view_skill` loaded, even if the skill was
saved again in between. `update_skill_files` saves put, move and delete operations as a new
version with the commit message and keeps the files it did not touch; it refuses to remove
SKILL.md and leaves the skill as it was. `create_skill` writes a private skill with SKILL.md and
its supporting files, taking the description from the front matter. `/skills:create` in a chat
with earlier messages asks the model to save the skill with `create_skill`, with no terminal
selected.

A SKILL.md longer than the content limit (the admin's `SKILL_CONTENT_MAX_CHARS`, 20,000 when
unset) reaches the model as a preview of the preview limit (`SKILL_PREVIEW_MAX_CHARS`, 5,000
when unset), whether it is `$`-mentioned or loaded with `view_skill`, followed by "Skill
instructions truncated", the skill id and the `next_offset` to page on from; a shorter one arrives
whole. On a model with the Skills builtin tool off a mentioned skill is cut at 100,000 characters
with no offset, since the model cannot page. `read_skill_file` returns at most the content limit
per call whatever `max_chars` asks for. `grep_skill_files` returns excerpts with the path and
offset of each match, ignores case unless told not to, searches only `path` when given, returns
at most `max_results` excerpts and lists the rest by location, and skips binary files.

Discriminates: passes on dev 178de3666. Each of these edits to a backend copy turns its test red:
a chat offered `view_skill` alone (offered tools), `view_skill` without the file list (view),
`read_skill_file` ignoring `offset` (paging), pasting a binary file as base64 (binary), reading a
missing file as empty (missing file), reading the current version instead of the one `view_skill`
loaded (pinning), file operations that start from SKILL.md alone (update), a SKILL.md delete that
is skipped without a word (SKILL.md), `create_skill` dropping the supporting files (create) and
skill authoring that still needs a terminal (`/skills:create`). Retargeted for c7193760b, where the
tools no longer show the model a `version_id`: pinning is read from the file text the model gets
and the saved version from the skill's history; passes on dev c7193760b and the pinning edit
still turns it red. Retargeted for 5032e834e, which offers the file tools only in a chat that
resolves a skill: the offered tools are read with one of the owner's skills in place, and
`grep_skill_files` joins them; passes on dev 5032e834e. The large-skill and search tests pass on
dev 948432b43, and in a backend copy: loading a large skill at the content limit instead of the
preview turned the preview and admin limits tests red; leaving the skill id and offset out of the
truncation note turned the preview test red, and cutting a skill at the content limit with the
Skills tool off turned the 100,000 test red; a page uncapped by the content limit turned the page
and admin limits tests red; ignoring the admin's limits turned the admin limits test red; a search
that always ignores case and returns 30 excerpts and reads binary files turned the case, cap and
binary tests red; and a search that ignores `path` turned the path test red.
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
import uuid
from typing import Iterator

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
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    delete_skills_of(account)


SKILL_TOOLS = {
    "view_skill",
    "read_skill_file",
    "grep_skill_files",
    "update_skill_files",
    "create_skill",
}
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
    _knots(owner)
    with owner.client() as client:
        offered = offered_tools(client, upstream)

    assert SKILL_TOOLS <= offered, sorted(offered)


def test_view_skill_returns_the_instructions_and_the_file_list(owner, upstream):
    skill = _knots(owner)

    viewed = _call(owner, upstream, "view_skill", id=skill["id"])

    assert viewed["id"] == skill["id"]
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
    assert read(owner, skill["id"], "references/hitch.md") == b"Changed.\n"
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

    assert updated.get("id") == skill["id"], updated
    saved_version = current(owner, skill["id"])["version_id"]
    assert saved_version != skill["version_id"]
    assert set(files(owner, skill["id"])) == {
        "SKILL.md",
        "assets/knot.png",
        "knots/bowline.md",
        "references/hitch.md",
    }
    assert read(owner, skill["id"], "references/hitch.md") == b"Three turns and a hitch.\n"
    assert read(owner, skill["id"], "assets/knot.png") == PNG
    latest = next(entry for entry in history(owner, skill["id"]) if entry["id"] == saved_version)
    assert latest["parent_id"] == skill["version_id"]
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


# Large skills load as a preview (0.12.1); the limits became admin settings on dev.

ADMIN_CONFIG = "/api/v1/auths/admin/config"
TRUNCATED = "Skill instructions truncated"


@pytest.fixture
def skill_limits(admin, preserve):
    """`skill_limits(content, preview)` saves the admin's skill limits; None leaves one unset."""
    preserve("admin_config")

    def save(content: int | None = None, preview: int | None = None) -> None:
        limits = {"SKILL_CONTENT_MAX_CHARS": content, "SKILL_PREVIEW_MAX_CHARS": preview}
        with admin.client() as client:
            current_config = client.get(ADMIN_CONFIG)
            assert current_config.status_code == 200, current_config.text
            saved = client.post(ADMIN_CONFIG, json={**current_config.json(), **limits})
        assert saved.status_code == 200, saved.text
        assert {key: saved.json().get(key) for key in limits} == limits, saved.json()

    return save


def _instructions(skill_id: str, length: int) -> str:
    """A SKILL.md of exactly `length` characters whose every slice is unique."""
    steps = "".join(f"Step {number:06d}: keep the line taut.\n" for number in range(length // 20))
    text = (skill_md(skill_id, "") + steps)[:length]
    assert len(text) == length
    return text


def _large_skill(owner, length: int, **extra_files: str) -> tuple[dict, str]:
    skill_id = new_id("rigging")
    text = _instructions(skill_id, length)
    return create(owner, {"SKILL.md": text, **extra_files}, skill_id=skill_id), text


def _injected(actor, upstream, skill: dict, model: str = MOCK_MODEL_ID) -> str:
    """The `<skill>` block the provider got for a message that mentions `skill` with `$`."""
    tag = uuid.uuid4().hex[:8]
    upstream.queue(reply.text("done", match=reply.answering(tag)))
    with actor.client() as client:
        ask(client, f"<${skill['id']}|{skill['name']}> how do I rig it? ({tag})", model=model)
    request = next(entry for entry in upstream.chat_requests() if reply.answering(tag)(entry))
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    start = system.find(f'<skill id="{skill["id"]}"')
    assert start >= 0, f"the mentioned skill was not loaded: {system[:2000]}"
    return system[start : system.find("</skill>", start)]


@contextlib.contextmanager
def _model_without_skill_tools(owner) -> Iterator[str]:
    """A native function calling model with its Skills builtin tool switched off."""
    model_id = new_id("no-skill-tools")
    with owner.client() as client:
        made = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {"capabilities": {}, "builtinTools": {"skills": False}},
                "params": {"function_calling": "native"},
                "access_grants": [],
            },
        )
        assert made.status_code == 200, made.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        try:
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_skill_over_the_content_limit_reaches_the_model_as_a_preview(
    owner, upstream, skill_limits
):
    skill_limits()
    skill, text = _large_skill(owner, 20_001)

    block = _injected(owner, upstream, skill)
    viewed = _call(owner, upstream, "view_skill", id=skill["id"])
    rest = _call(
        owner,
        upstream,
        "read_skill_file",
        id=skill["id"],
        path="SKILL.md",
        offset=5_000,
        max_chars=20_000,
    )

    assert text[:5_000] in block, f"the preview is missing: {block[:300]}"
    assert text[:5_001] not in block, "the model got more than the 5,000 character preview"
    note = block.split(text[:5_000], 1)[1]
    assert TRUNCATED in note and skill["id"] in note, note
    assert re.search(r"next_offset\W*5000\b", note), note
    assert (viewed["content"], viewed["next_offset"]) == (text[:5_000], 5_000), viewed
    assert (rest["content"], rest["next_offset"]) == (text[5_000:], None), rest["next_offset"]


def test_a_skill_within_the_content_limit_arrives_whole(owner, upstream, skill_limits):
    skill_limits()
    skill, text = _large_skill(owner, 20_000)

    block = _injected(owner, upstream, skill)

    assert text in block, "a 20,000 character skill did not arrive whole"
    assert TRUNCATED not in block, block[-300:]


def test_with_the_skills_tool_off_a_mentioned_skill_is_cut_at_100000_characters(
    owner, upstream, skill_limits
):
    skill_limits()
    skill, text = _large_skill(owner, 100_001)

    with _model_without_skill_tools(owner) as model:
        block = _injected(owner, upstream, skill, model=model)

    assert text[:100_000] in block, "the skill was cut before 100,000 characters"
    assert text not in block, "the skill was not cut at 100,000 characters"
    note = block.split(text[:100_000], 1)[1]
    assert TRUNCATED in note, note
    # without the skill tools the model cannot page on, so it is not told where to
    assert "next_offset" not in note, note


def test_read_skill_file_returns_at_most_the_content_limit_per_page(owner, upstream, skill_limits):
    skill_limits()
    table = "".join(f"Tide {number:05d} at noon.\n" for number in range(1_500))
    skill = create(
        owner,
        {"SKILL.md": skill_md("tides", "Read the table."), "references/table.md": table},
        skill_id=new_id("tides"),
    )

    page = _call(
        owner,
        upstream,
        "read_skill_file",
        id=skill["id"],
        path="references/table.md",
        max_chars=50_000,
    )

    assert len(table) > 20_000
    assert page["content"] == table[:20_000], len(page["content"])
    assert page["next_offset"] == 20_000


def test_the_admin_skill_limits_set_the_preview_and_the_page_size(owner, upstream, skill_limits):
    skill_limits(content=3_000, preview=800)
    skill, text = _large_skill(owner, 3_001)

    block = _injected(owner, upstream, skill)
    page = _call(
        owner, upstream, "read_skill_file", id=skill["id"], path="SKILL.md", max_chars=10_000
    )

    assert text[:800] in block and text[:801] not in block, "the preview ignored the setting"
    assert re.search(r"next_offset\W*800\b", block.split(text[:800], 1)[1]), block[-300:]
    assert (page["content"], page["next_offset"]) == (text[:3_000], 3_000), page["next_offset"]


# grep_skill_files

FILLER = "slack line " * 150


def _grep_skill(owner) -> tuple[dict, dict[str, str]]:
    skill_id = new_id("lines")
    texts = {
        "SKILL.md": skill_md(skill_id, "Start with the BOWLINE."),
        "references/bowline.md": "Rabbit out of the hole.\nBowline on a bight.\n",
        "references/hitch.md": "A bowline is not a hitch.\n",
        "references/reefing.md": "".join(f"{FILLER}reef point {n}.\n" for n in range(5)),
    }
    skill = create(owner, texts, skill_id=skill_id)
    return skill, texts


def _grep(owner, upstream, skill: dict, pattern: str, **options) -> dict:
    return _call(owner, upstream, "grep_skill_files", id=skill["id"], pattern=pattern, **options)


def test_grep_skill_files_finds_matches_with_their_path_and_offset_ignoring_case(owner, upstream):
    skill, texts = _grep_skill(owner)

    found = _grep(owner, upstream, skill, "bowline")
    exact = _grep(owner, upstream, skill, "bowline", case_insensitive=False)

    assert found["total_matches"] == 3, found
    assert {match["path"] for match in found["matches"]} == {
        "SKILL.md",
        "references/bowline.md",
        "references/hitch.md",
    }, found
    for match in found["matches"]:
        located = texts[match["path"]][match["offset"] : match["offset"] + match["match_length"]]
        assert located.lower() == "bowline", match
        assert located in match["excerpt"], match
    assert exact["total_matches"] == 1, exact
    assert [(match["path"], match["offset"]) for match in exact["matches"]] == [
        ("references/hitch.md", texts["references/hitch.md"].index("bowline"))
    ]


def test_grep_skill_files_searches_only_the_given_path(owner, upstream):
    skill, _ = _grep_skill(owner)

    found = _grep(owner, upstream, skill, "bowline", path="references/hitch.md")

    assert found["total_matches"] == 1, found
    assert [match["path"] for match in found["matches"]] == ["references/hitch.md"], found


def test_grep_skill_files_returns_at_most_max_results_excerpts(owner, upstream):
    skill, texts = _grep_skill(owner)
    reefing = texts["references/reefing.md"]

    found = _grep(owner, upstream, skill, "reef point", max_results=2)

    assert found["total_matches"] == 5, found
    assert len(found["matches"]) == 2 and found["truncated"] is True, found
    shown = [match["offset"] for match in found["matches"]]
    omitted = [location["offset"] for location in found["omitted_matches"]]
    expected = [match.start() for match in re.finditer("reef point", reefing)]
    assert sorted(shown + omitted) == expected, found


def test_grep_skill_files_skips_binary_files(owner, upstream):
    blob = b"\x00knot knot knot knot"
    # a run of the file's stored base64 text, so a search that read it would match
    pattern = base64.b64encode(blob).decode()[2:14]
    assert pattern.isalnum()
    skill_id = new_id("blob")
    skill = create(
        owner,
        {"SKILL.md": skill_md(skill_id, "Tie it."), "references/notes.md": f"Code {pattern}.\n"},
        skill_id=skill_id,
    )
    saved = save(
        owner,
        skill,
        [
            {
                "op": "put",
                "path": "assets/blob.bin",
                "content": base64.b64encode(blob).decode(),
                "encoding": "base64",
            }
        ],
    )
    assert saved.status_code == 200, saved.text

    found = _grep(owner, upstream, saved.json(), pattern)
    only_binary = _grep(owner, upstream, saved.json(), pattern, path="assets/blob.bin")

    assert found["total_matches"] == 1, found
    assert [match["path"] for match in found["matches"]] == ["references/notes.md"], found
    assert "matches" not in only_binary and "error" in only_binary, only_binary
