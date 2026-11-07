"""Journey: one account keeps at most 20 live note documents open, across all of its tabs.

Since de90e1305 (#33864) the limit on open live documents counts every connection of the account
together, where before each connection had 20 of its own, so opening more tabs multiplied what
one account could hold open on the server. Once the account has 20 open, a further note opens in
none of its tabs, while a note already open in a tab still reopens there, closing one frees a
place, and another account is not held to the first one's count.

Discriminates: passes on dev f659c0d12; with de90e1305 reverted in a backend copy (the count is
per connection again) the second tab opens the 21st note. The allowed cases pass on both.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

LIMIT = 20
ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0


def _create_notes(owner, count: int) -> list[str]:
    note_ids = []
    with owner.client() as client:
        for _ in range(count):
            created = client.post(
                "/api/v1/notes/create",
                json={"title": f"limit {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
            )
            assert created.status_code == 200, created.text
            note_ids.append(created.json()["id"])
    return note_ids


def _share_for_reading(owner, note_id: str, reader) -> None:
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text


def _opens(tab: SocketSession, note_id: str, expected: bool) -> bool:
    """Join the note's live document; whether the server served it anew within the wait."""

    def served() -> int:
        return sum(state["document_id"] == f"note:{note_id}" for state in tab.document_states)

    before = served()
    tab.join_note(note_id)
    deadline = time.monotonic() + (ARRIVAL_TIMEOUT if expected else QUIET_PERIOD)
    while time.monotonic() < deadline:
        if served() > before:
            return True
        time.sleep(0.05)
    return False


def _fill(tab: SocketSession, note_ids: list[str]) -> None:
    for note_id in note_ids:
        assert _opens(tab, note_id, True), f"note {note_id} did not open below the limit"


def test_the_limit_counts_every_tab_of_the_account(make_user):
    owner = make_user()
    *open_ones, one_more = _create_notes(owner, LIMIT + 1)

    with connected(owner) as first_tab, connected(owner) as second_tab:
        _fill(first_tab, open_ones)
        opened_in_second_tab = _opens(second_tab, one_more, False)
        reopened_in_first_tab = _opens(first_tab, open_ones[0], True)

    assert not opened_in_second_tab, (
        f"a second tab opened a note beyond the {LIMIT} the account already had open"
    )
    assert reopened_in_first_tab, "a note already open in a tab no longer reopened at the limit"


def test_closing_a_note_frees_a_place(make_user):
    owner = make_user()
    *open_ones, one_more = _create_notes(owner, LIMIT + 1)

    with connected(owner) as first_tab, connected(owner) as second_tab:
        _fill(first_tab, open_ones)
        first_tab.call("ydoc:document:leave", {"document_id": f"note:{open_ones[0]}"})
        opened = _opens(second_tab, one_more, True)

    assert opened, "after closing a note the account still could not open another"


def test_another_account_is_not_held_to_the_first_ones_count(make_user):
    owner, reader = make_user(), make_user()
    *open_ones, shared = _create_notes(owner, LIMIT + 1)
    _share_for_reading(owner, shared, reader)

    with connected(owner) as owners_tab, connected(reader) as readers_tab:
        _fill(owners_tab, open_ones)
        opened = _opens(readers_tab, shared, True)

    assert opened, "another account was refused because the first one was at its limit"
