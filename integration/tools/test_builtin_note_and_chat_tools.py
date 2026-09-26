"""Journey: the model searches, reads, writes and edits notes and reads earlier chats.

The notes tools are offered while notes are switched on and the account may use them: a note
the model writes is private to the account, `search_notes` and `view_note` see only notes the
account owns or was granted, and `replace_note_content` needs write access and applies whole
replacements or offset ranges, refusing a range whose expected text no longer matches. The chat
tools search and read the account's own earlier chats and never the one being answered.

The scripted model calls each tool and the test reads the result it was sent back, and the
stored note or chat.

Discriminates: in a backend copy, `view_note` skipping its grant check turned the stranger test
red; `_has_write_access_to_note` asking for read in place of write turned the reader edit test
red; dropping the expected-text check turned the mismatch test red; applying ranges front to
back turned the two-range test red; `search_chats` keeping the current chat turned the exclusion
test red; `view_chat` not reversing the walk turned the order test red; and the notes gate
ignoring the permission turned the barred-account test red.
"""

from __future__ import annotations

import json
import time

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.tool_calls import offered_tools, run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

NOTE_TOOLS = {"search_notes", "view_note", "write_note", "replace_note_content"}
CHAT_TOOLS = {"search_chats", "view_chat"}
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
NOTE_BODY = "Heron sightings: two by the reed bed, one on the jetty."


def call(actor, upstream, tool: str, **arguments):
    """The JSON a builtin tool returned when the model called it for `actor`."""
    with actor.client() as client:
        return json.loads(run_tool(client, upstream, tool, arguments))


def create_note(actor, title: str, body: str, access_grants: list[dict] | None = None) -> str:
    with actor.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": title,
                "data": {"content": {"md": body}},
                "access_grants": access_grants or [],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def stored_note(actor, note_id: str) -> dict:
    with actor.client() as client:
        fetched = client.get(f"/api/v1/notes/{note_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def grant(actor, permission: str) -> dict:
    return {"principal_type": "user", "principal_id": actor.id, "permission": permission}


# --- notes --------------------------------------------------------------------------------------


def test_a_written_note_is_stored_privately_for_its_author(make_user, upstream):
    author, stranger = make_user(), make_user()

    result = call(author, upstream, "write_note", title="Birding", content=NOTE_BODY)

    assert result["status"] == "success"
    stored = stored_note(author, result["id"])
    assert (stored["title"], stored["data"]["content"]["md"]) == ("Birding", NOTE_BODY)
    with stranger.client() as client:
        assert client.get(f"/api/v1/notes/{result['id']}").status_code == 403


def test_search_finds_own_and_shared_notes_with_a_snippet(make_user, upstream):
    owner, reader, stranger = make_user(), make_user(), make_user()
    padding = "Weather was grey all morning and the path was muddy. " * 3
    shared_id = create_note(owner, "Lake log", padding + NOTE_BODY, [grant(reader, "read")])
    private_id = create_note(owner, "Private log", NOTE_BODY)
    own_id = create_note(reader, "My herons", "A heron fished at dawn.")

    found = call(reader, upstream, "search_notes", query="heron")

    by_id = {note["id"]: note for note in found}
    assert set(by_id) == {shared_id, own_id}, f"expected the shared and own notes: {found}"
    assert private_id not in by_id
    snippet = by_id[shared_id]["snippet"]
    assert snippet.startswith("...") and "Heron sightings" in snippet, snippet
    assert call(stranger, upstream, "search_notes", query="heron") == []


def test_search_honours_the_update_window_and_the_count(make_user, upstream):
    owner = make_user()
    for index in range(3):
        create_note(owner, f"Heron {index}", f"heron entry {index}")
    now = int(time.time())

    assert len(call(owner, upstream, "search_notes", query="heron", count=2)) == 2
    assert call(owner, upstream, "search_notes", query="heron", end_timestamp=now - 3600) == []
    later = call(owner, upstream, "search_notes", query="heron", start_timestamp=now - 3600)
    assert len(later) == 3


def test_view_note_reads_own_and_granted_notes_only(make_user, upstream):
    owner, reader, stranger = make_user(), make_user(), make_user()
    note_id = create_note(owner, "Lake log", NOTE_BODY, [grant(reader, "read")])

    viewed = call(reader, upstream, "view_note", note_id=note_id)
    refused = call(stranger, upstream, "view_note", note_id=note_id)

    assert (viewed["title"], viewed["content"]) == ("Lake log", NOTE_BODY)
    assert refused == {"error": "Access denied"}, f"a stranger read someone's note: {refused}"
    assert call(owner, upstream, "view_note", note_id="no-such-note") == {"error": "Note not found"}


def test_replacing_the_whole_note_keeps_the_title_unless_given(make_user, upstream):
    owner = make_user()
    note_id = create_note(owner, "Lake log", NOTE_BODY)

    result = call(owner, upstream, "replace_note_content", note_id=note_id, content="rewritten")
    call(owner, upstream, "replace_note_content", note_id=note_id, content="v2", title="Renamed")

    assert result["applied_operation_count"] == 0
    stored = stored_note(owner, note_id)
    assert (stored["title"], stored["data"]["content"]["md"]) == ("Renamed", "v2")


def edit(start: int, end: int, content: str, **extra) -> dict:
    return {"action": "replace_range", "start": start, "end": end, "content": content, **extra}


def test_two_ranges_are_applied_against_the_original_offsets(make_user, upstream):
    owner = make_user()
    note_id = create_note(owner, "Counts", "one heron, two geese")
    operations = [edit(0, 3, "three", expected="one"), edit(11, 14, "four")]

    result = call(owner, upstream, "replace_note_content", note_id=note_id, operations=operations)

    assert result["applied_operation_count"] == 2, result
    assert stored_note(owner, note_id)["data"]["content"]["md"] == "three heron, four geese"


BAD_EDITS = {
    "expected_mismatch": [edit(0, 3, "x", expected="two")],
    "overlapping_operations": [edit(0, 5, "a"), edit(4, 8, "b")],
    "range_out_of_bounds": [edit(0, 999, "x")],
    "invalid_operations": [{"action": "replace", "content": "a"}, edit(0, 1, "b")],
    "invalid_action": [{"action": "append", "content": "x"}],
}


@pytest.mark.parametrize("code", BAD_EDITS)
def test_a_bad_edit_is_refused_and_the_note_is_unchanged(make_user, upstream, code):
    owner = make_user()
    note_id = create_note(owner, "Counts", "one heron, two geese")

    result = call(
        owner, upstream, "replace_note_content", note_id=note_id, operations=BAD_EDITS[code]
    )

    assert result.get("code") == code, result
    assert stored_note(owner, note_id)["data"]["content"]["md"] == "one heron, two geese"


def test_editing_needs_write_access(make_user, upstream):
    owner, reader, writer = make_user(), make_user(), make_user()
    grants = [grant(reader, "read"), grant(writer, "write")]
    note_id = create_note(owner, "Shared", NOTE_BODY, grants)

    refused = call(reader, upstream, "replace_note_content", note_id=note_id, content="vandal")
    accepted = call(writer, upstream, "replace_note_content", note_id=note_id, content="edited")

    assert refused.get("code") == "write_access_denied", f"a reader edited the note: {refused}"
    assert accepted["status"] == "success"
    assert stored_note(owner, note_id)["data"]["content"]["md"] == "edited"


def test_an_unknown_note_and_a_missing_content_are_reported(make_user, upstream):
    owner = make_user()
    note_id = create_note(owner, "Lake log", NOTE_BODY)

    missing = call(owner, upstream, "replace_note_content", note_id="no-such-note", content="x")
    empty = call(owner, upstream, "replace_note_content", note_id=note_id)

    assert (missing["code"], empty["code"]) == ("not_found", "content_required")


def test_the_notes_tools_follow_the_notes_permission(make_user, upstream, admin, preserve):
    preserve("permissions")
    account = make_user()
    with account.client() as client:
        assert NOTE_TOOLS <= offered_tools(client, upstream)
    with admin.client() as client:
        current = client.get(DEFAULT_PERMISSIONS).json()
        barred = {**current, "features": {**current["features"], "notes": False}}
        client.post(DEFAULT_PERMISSIONS, json=barred).raise_for_status()

    with account.client() as client:
        offered = offered_tools(client, upstream)

    assert not NOTE_TOOLS & offered, f"an account barred from notes was offered {offered}"
    assert CHAT_TOOLS <= offered


# --- chats --------------------------------------------------------------------------------------


def earlier_chat(actor, upstream, prompt: str, answer: str) -> str:
    upstream.queue(reply.text(answer))
    with actor.client() as client:
        turn, _ = ask(client, prompt)
    return turn.chat_id


def test_search_chats_finds_earlier_chats_but_not_the_current_one(make_user, upstream):
    account, stranger = make_user(), make_user()
    earlier_id = earlier_chat(account, upstream, "use search_chats about herons", "noted")
    earlier_chat(stranger, upstream, "use search_chats as a stranger", "noted")

    found = call(account, upstream, "search_chats", query="use search_chats")

    assert [chat["id"] for chat in found] == [earlier_id], (
        f"the search listed the current chat or someone else's: {found}"
    )
    assert "about herons" in found[0]["snippet"]


def test_view_chat_returns_the_conversation_in_order(make_user, upstream):
    account, stranger = make_user(), make_user()
    chat_id = earlier_chat(account, upstream, "where do herons nest?", "in the reeds")

    viewed = call(account, upstream, "view_chat", chat_id=chat_id)
    refused = call(stranger, upstream, "view_chat", chat_id=chat_id)

    assert viewed["messages"] == [
        {"role": "user", "content": "where do herons nest?"},
        {"role": "assistant", "content": "in the reeds"},
    ]
    assert refused == {"error": "Chat not found or access denied"}
