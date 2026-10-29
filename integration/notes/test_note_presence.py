"""Journey: a note shows who else has it open and where they are in it.

Every tab that opens a note (`note:join`) gets a `note:presence` list of the people with the note
open, each with only their id, name and where they are: the page whose live document they opened
and their cursor on it. The list follows a person to another page and to a new cursor position.
Leaving the note, closing the tab, losing the grant or leaving the group that held it takes the
person off the list. An account that may not read the note is refused at join, never shows up in
the list and never receives it, also after its grant is taken away.

Discriminates: passes on dev c5981aaea. In a backend copy of it, `publish_note_presence` sending
nothing fails every test; sending the whole socket session in place of id, name and location fails
the both writers test; `note:leave` not publishing it fails the leave, grant and group tests;
`disconnect` not publishing it fails the leave and group tests; `ydoc:document:join` not recording
the page fails the both writers, location and stranger tests; the awareness update not recording
the cursor fails the location test; `note:join` skipping its access check fails the stranger and
group tests; `leave_room_for_users` leaving the note room without `leave_note` fails the grant test.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness.access import grant, make_group
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ARRIVAL_TIMEOUT = 15.0
QUIET_SECONDS = 2.0
LOCATION_FIELDS = {"page_id", "document_id", "cursor", "cursor_position", "updated_at"}


def _create_note(owner, *grants: dict) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"harbour log {uuid.uuid4().hex[:6]}",
                "data": {"content": {"md": ""}},
                "access_grants": list(grants),
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _set_grants(owner, note_id: str, *grants: dict) -> None:
    with owner.client() as client:
        saved = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": list(grants)}
        )
    assert saved.status_code == 200, saved.text


def _add_page(owner, note_id: str, title: str) -> str:
    """A second page, added the way the page tabs add one; returns its id."""
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}")
        assert note.status_code == 200, note.text
        page_id = str(uuid.uuid4())
        added = client.post(
            f"/api/v1/notes/{note_id}/pages/update",
            json={
                "action": "create",
                "new_id": page_id,
                "title": title,
                "revision": note.json()["data"].get("pages_revision", 0),
            },
        )
    assert added.status_code == 200, added.text
    return page_id


def _page_document(note_id: str, page_id: str) -> str:
    return f"note:{note_id}:page:{page_id}:0"


def _watch_presence(session: SocketSession) -> list[dict]:
    """Record every `note:presence` the tab receives."""
    received: list[dict] = []
    session.client.on("note:presence", received.append)
    return received


def _open_note(session: SocketSession, account, note_id: str) -> None:
    """What the note page sends when it opens a note."""
    session.call("note:join", {"note_id": note_id, "auth": {"token": account.token}})


def _open_page(session: SocketSession, note_id: str, page_id: str) -> None:
    """What the editor sends when it shows a page: it joins that page's live document."""
    session.call("ydoc:document:join", {"document_id": _page_document(note_id, page_id)})


def _move_cursor(session: SocketSession, note_id: str, page_id: str, position: int) -> None:
    session.call(
        "ydoc:awareness:update",
        {
            "document_id": _page_document(note_id, page_id),
            "update": [0],
            "cursor": {"anchor": {"type": "x"}, "head": {"type": "x"}},
            "cursor_position": position,
        },
    )


def _lists_for(received: list[dict], note_id: str) -> list[list[dict]]:
    return [event["users"] for event in list(received) if event.get("note_id") == note_id]


def _ids(people: list[dict]) -> list[str]:
    return [person["id"] for person in people]


def _expected_ids(*accounts) -> list[str]:
    return sorted(account.id for account in accounts)


def _wait_for_list(
    received: list[dict], note_id: str, accounts, after: int = 0, match=None
) -> list[dict]:
    """The first list after the first `after` naming exactly `accounts` (and passing `match`)."""
    expected = _expected_ids(*accounts)
    deadline = time.monotonic() + ARRIVAL_TIMEOUT
    while time.monotonic() < deadline:
        for people in _lists_for(received, note_id)[after:]:
            if _ids(people) == expected and (match is None or match(people)):
                return people
        time.sleep(0.05)
    raise AssertionError(
        f"no presence list naming {expected} for the note within {ARRIVAL_TIMEOUT}s; "
        f"got {_lists_for(received, note_id)}"
    )


def _person(people: list[dict], account) -> dict:
    return next(person for person in people if person["id"] == account.id)


def test_both_writers_see_each_other_with_only_their_name_and_place(make_user):
    owner, writer = make_user(), make_user()
    note_id = _create_note(
        owner, grant("user", writer.id, "read"), grant("user", writer.id, "write")
    )

    with connected(owner) as owner_tab, connected(writer) as writer_tab:
        owner_lists, writer_lists = _watch_presence(owner_tab), _watch_presence(writer_tab)
        _open_note(owner_tab, owner, note_id)
        _open_page(owner_tab, note_id, "main")
        _open_note(writer_tab, writer, note_id)
        _open_page(writer_tab, note_id, "main")

        def both_placed(people):
            return all(person.get("page_id") == "main" for person in people)

        seen_by_owner = _wait_for_list(owner_lists, note_id, (owner, writer), match=both_placed)
        seen_by_writer = _wait_for_list(writer_lists, note_id, (owner, writer), match=both_placed)

    for people in (seen_by_owner, seen_by_writer):
        assert _person(people, owner)["name"] == owner.name
        assert _person(people, writer)["name"] == writer.name
    for people in _lists_for(owner_lists, note_id) + _lists_for(writer_lists, note_id):
        for person in people:
            extra = set(person) - {"id", "name"} - LOCATION_FIELDS
            assert not extra, f"the presence list shows more about {person['name']}: {person}"


def test_a_person_who_leaves_or_closes_the_tab_drops_off_the_list(make_user):
    owner, writer = make_user(), make_user()
    note_id = _create_note(
        owner, grant("user", writer.id, "write"), grant("user", writer.id, "read")
    )

    with connected(owner) as owner_tab:
        owner_lists = _watch_presence(owner_tab)
        _open_note(owner_tab, owner, note_id)

        with connected(writer) as writer_tab:
            _open_note(writer_tab, writer, note_id)
            _wait_for_list(owner_lists, note_id, (owner, writer))
            seen = len(_lists_for(owner_lists, note_id))
            writer_tab.call("note:leave", {"note_id": note_id})
            _wait_for_list(owner_lists, note_id, (owner,), after=seen)

        with connected(writer) as closing_tab:
            seen = len(_lists_for(owner_lists, note_id))
            _open_note(closing_tab, writer, note_id)
            _wait_for_list(owner_lists, note_id, (owner, writer), after=seen)
            seen = len(_lists_for(owner_lists, note_id))
        _wait_for_list(owner_lists, note_id, (owner,), after=seen)


def test_the_list_follows_each_person_to_their_page_and_cursor(make_user):
    owner, writer = make_user(), make_user()
    note_id = _create_note(
        owner, grant("user", writer.id, "read"), grant("user", writer.id, "write")
    )
    second_page = _add_page(owner, note_id, "Tides")

    with connected(owner) as owner_tab, connected(writer) as writer_tab:
        owner_lists = _watch_presence(owner_tab)
        _open_note(owner_tab, owner, note_id)
        _open_note(writer_tab, writer, note_id)
        _open_page(writer_tab, note_id, "main")

        def writer_on(page_id):
            return lambda people: _person(people, writer).get("page_id") == page_id

        on_main = _wait_for_list(owner_lists, note_id, (owner, writer), match=writer_on("main"))
        assert _person(on_main, writer)["document_id"] == _page_document(note_id, "main")

        _open_page(writer_tab, note_id, second_page)
        on_second = _wait_for_list(
            owner_lists, note_id, (owner, writer), match=writer_on(second_page)
        )
        assert _person(on_second, writer)["document_id"] == _page_document(note_id, second_page)

        _move_cursor(writer_tab, note_id, second_page, 7)
        at_cursor = _wait_for_list(
            owner_lists,
            note_id,
            (owner, writer),
            match=lambda people: _person(people, writer).get("cursor_position") == 7,
        )

    assert _person(at_cursor, writer)["page_id"] == second_page


def test_an_account_without_access_is_never_listed_and_gets_no_list(make_user):
    owner, stranger = make_user(), make_user()
    note_id = _create_note(owner)

    with connected(owner) as owner_tab, connected(stranger) as stranger_tab:
        owner_lists, stranger_lists = _watch_presence(owner_tab), _watch_presence(stranger_tab)
        _open_note(stranger_tab, stranger, note_id)
        _open_page(stranger_tab, note_id, "main")
        _open_note(owner_tab, owner, note_id)
        _open_page(owner_tab, note_id, "main")
        _wait_for_list(owner_lists, note_id, (owner,), match=lambda p: "page_id" in p[0])
        time.sleep(QUIET_SECONDS)  # nothing more should arrive

    for people in _lists_for(owner_lists, note_id):
        assert stranger.id not in _ids(people), f"a stranger shows up in the note: {people}"
    assert _lists_for(stranger_lists, note_id) == [], (
        f"an account without access received who has the note open: {stranger_lists}"
    )


def test_a_reader_whose_grant_is_taken_away_drops_off_and_stops_getting_the_list(make_user):
    owner, removed, kept = make_user(), make_user(), make_user()
    note_id = _create_note(owner, grant("user", removed.id, "read"), grant("user", kept.id, "read"))

    with (
        connected(owner) as owner_tab,
        connected(removed) as removed_tab,
        connected(kept) as kept_tab,
    ):
        owner_lists, removed_lists = _watch_presence(owner_tab), _watch_presence(removed_tab)
        for tab, account in ((owner_tab, owner), (removed_tab, removed), (kept_tab, kept)):
            _open_note(tab, account, note_id)
        _wait_for_list(removed_lists, note_id, (owner, removed, kept))
        seen = len(_lists_for(owner_lists, note_id))

        _set_grants(owner, note_id, grant("user", kept.id, "read"))
        _wait_for_list(owner_lists, note_id, (owner, kept), after=seen)
        cut_off = len(_lists_for(removed_lists, note_id))
        kept_tab.call("note:leave", {"note_id": note_id})
        _wait_for_list(owner_lists, note_id, (owner,), after=seen)
        time.sleep(QUIET_SECONDS)  # nothing more should arrive

    assert _lists_for(removed_lists, note_id)[cut_off:] == [], (
        "a reader whose grant was taken away still received who has the note open"
    )


def test_a_member_removed_from_the_group_drops_off_and_stops_getting_the_list(admin, make_user):
    owner, leaver, stayer = make_user(), make_user(), make_user()
    group_id = make_group(admin, [leaver, stayer])
    note_id = _create_note(owner, grant("group", group_id, "read"))

    with (
        connected(owner) as owner_tab,
        connected(leaver) as leaver_tab,
        connected(stayer) as stayer_tab,
    ):
        owner_lists, leaver_lists = _watch_presence(owner_tab), _watch_presence(leaver_tab)
        for tab, account in ((owner_tab, owner), (leaver_tab, leaver), (stayer_tab, stayer)):
            _open_note(tab, account, note_id)
        _wait_for_list(leaver_lists, note_id, (owner, leaver, stayer))
        seen = len(_lists_for(owner_lists, note_id))

        with admin.client() as client:
            removed = client.post(
                f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [leaver.id]}
            )
        assert removed.status_code == 200, removed.text
        _wait_for_list(owner_lists, note_id, (owner, stayer), after=seen)
        cut_off = len(_lists_for(leaver_lists, note_id))
        stayer_tab.call("note:leave", {"note_id": note_id})
        _wait_for_list(owner_lists, note_id, (owner,), after=seen)
        time.sleep(QUIET_SECONDS)  # nothing more should arrive

        with connected(leaver) as reopened_tab:
            reopened_lists = _watch_presence(reopened_tab)
            _open_note(reopened_tab, leaver, note_id)
            time.sleep(QUIET_SECONDS)  # nothing should arrive

    assert _lists_for(leaver_lists, note_id)[cut_off:] == [], (
        "a member removed from the group still received who has the note open"
    )
    assert _lists_for(reopened_lists, note_id) == [], (
        "a member removed from the group opened the note again and received who has it open"
    )
