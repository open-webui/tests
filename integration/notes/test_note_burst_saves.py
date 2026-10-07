"""Regression: a fast burst of note edits can leave the saved note behind the editor.

Issue #31585 (open, fix PR #31596 open): every edit does awaited work on the server (loading the
note, checking write access, storing and passing on the update) before it replaces the pending save
with its own, so with edits a few milliseconds apart an older edit can finish that work after a
newer one and its text is the one saved. The stored note then misses the last characters the editor
shows. It hits a writer the note was shared with most, because the access check widens the window,
and the owner as well.

Here a tab sends bursts of edits without waiting for the server, each carrying the full text so
far, on several notes at once; once the save delay has passed every note must hold the last text.
On Postgres the instance gets the pool the scaling docs suggest to start from
(`DATABASE_POOL_SIZE=15`, `DATABASE_POOL_MAX_OVERFLOW=20`) and the notes get their bursts one at a
time: with the default pool even one note's burst runs out of connections and the server answers
503, and several notes at once outrun the suggested pool too.

Discriminates: fails on dev 176d31d1d (some notes keep an older edit's text); passes once the
newest edit's save is the one that stays scheduled. On dev f6cbeb1a1 7 or 8 of the 8 notes keep an
older text on Postgres and Redis, and 11 to 22 of the 48 on SQLite; with PR #31596's per-note
ordering applied in a backend copy both tests pass on both, the SQLite writer's only once the wait
was raised from 8 to 30 seconds, as the ordered edits take longer to work through.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import backends
from harness.actors import create_user
from harness.socket_client import connected, note_edit

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# one note at a time on Postgres, where most notes fall behind anyway
NOTES_PER_ROUND = 1 if backends.DATABASE == "postgres" else 6
ROUNDS = 8
EDITS_PER_NOTE = 50
EDIT_SPACING = 0.002
# the server saves half a second after the last edit, once it has worked through the burst
SAVE_WAIT = 30.0
SENTENCE = "packing list with sunscreen and a hat for the trip"
# the pool the scaling docs suggest to start from; unset, Postgres gets SQLAlchemy's 5 + 10, which
# even one note's burst runs out of
POSTGRES_POOL = {"DATABASE_POOL_SIZE": "15", "DATABASE_POOL_MAX_OVERFLOW": "20"}


@pytest.fixture
def make_user(request: pytest.FixtureRequest):
    """Accounts on the shared instance, or on Postgres on one with the pool the docs suggest."""
    if backends.DATABASE == "postgres":
        target = request.getfixturevalue("instance_with")(POSTGRES_POOL)
    else:
        target = request.getfixturevalue("instance")
    return lambda: create_user(target)


def _create_note(owner, grants: list[dict]) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"burst {uuid.uuid4().hex[:8]}",
                "data": {"content": {"md": ""}},
                "access_grants": grants,
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_markdown(owner, note_id: str) -> str:
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}")
    assert note.status_code == 200, note.text
    return str(((note.json().get("data") or {}).get("content") or {}).get("md"))


def _edit_message(note_id: str, text: str) -> dict:
    content = {"md": text}
    return {
        "document_id": f"note:{note_id}",
        "update": note_edit(text),
        "data": {"content": content},
    }


def _type_bursts(writer, owner, note_ids: list[str]) -> str:
    """Send each round of notes a burst of growing texts, then return the last text saved."""
    steps = range(EDITS_PER_NOTE)
    texts = [SENTENCE[: 1 + step * len(SENTENCE) // EDITS_PER_NOTE] for step in steps]
    texts[-1] = SENTENCE
    with connected(writer) as tab:
        for note_id in note_ids:
            tab.join_note(note_id)
        for start in range(0, len(note_ids), NOTES_PER_ROUND):
            for text in texts:
                for note_id in note_ids[start : start + NOTES_PER_ROUND]:
                    tab.client.emit("ydoc:document:update", _edit_message(note_id, text))
                time.sleep(EDIT_SPACING)
            # a call is answered after the emits before it, so the round has been read
            tab.join_note(note_ids[start])
        return _wait_for_saves(owner, note_ids, SENTENCE)


def _wait_for_saves(reader, note_ids: list[str], last_text: str) -> str:
    deadline = time.monotonic() + SAVE_WAIT
    behind: dict[str, str] = {}
    while time.monotonic() < deadline:
        behind = {
            note_id: stored
            for note_id in note_ids
            if (stored := _stored_markdown(reader, note_id)) != last_text
        }
        if not behind:
            break
        time.sleep(0.5)
    assert not behind, (
        f"#31585: after a burst of edits the saved note is behind the editor in {len(behind)} "
        f"of {len(note_ids)} notes; it should read {last_text!r} "
        f"but holds {sorted(set(behind.values()))}"
    )
    return last_text


def test_a_writers_burst_of_edits_saves_the_last_text(make_user):
    owner, writer = make_user(), make_user()
    grants = [
        {"principal_type": "user", "principal_id": writer.id, "permission": "read"},
        {"principal_type": "user", "principal_id": writer.id, "permission": "write"},
    ]
    note_ids = [_create_note(owner, grants) for _ in range(NOTES_PER_ROUND * ROUNDS)]

    saved = _type_bursts(writer, owner, note_ids)

    assert saved == SENTENCE


def test_an_owners_burst_of_edits_saves_the_last_text(make_user):
    owner = make_user()
    note_ids = [_create_note(owner, []) for _ in range(NOTES_PER_ROUND * ROUNDS)]

    saved = _type_bursts(owner, owner, note_ids)

    assert saved == SENTENCE
