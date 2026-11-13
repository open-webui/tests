"""Journey: a folder page's statuses, tags, people and filters over the API.

Since 06491f013 a folder's page lists its chats as a list or a board. The folder defines its own
statuses (`meta.statuses`, saved with the folder update) and a chat's owner files the chat under
one of them, or under none, with `PATCH /api/v1/folders/{id}/chats/{chat_id}/status`; a status
the folder does not define is refused with 422, and a chat in a status the folder later deletes
reads as having none. Time in a status adds up across stretches, so going back to a status keeps
its earlier time. With `workspace=true` the folder's chat list pages in fifties, filtered before
paging by a search over titles and message text, by tag (`none` for untagged), by status (`none`
for no status), by person and by activity (unread or read), sorted by title or by last update in
either direction, and each chat says whether the caller may change its status and tags.
`GET /{id}/tags` counts the tags in use and `GET /{id}/users` counts each person's chats, the
caller first. A member of a shared folder files their own chat there; who may reach which route
is pinned by integration/security/test_folder_access_matrix.py. The owner of a tag gives it a
colour with `PATCH /api/v1/chats/{id}/tags`, only through their own chat that carries it.

Discriminates: journey, no fix. All pass on dev 06491f013 on SQLite and on PostgreSQL with Redis.
In one backend copy of it holding these eight edits, each turns its test red and the other four
stay green: dropping the status check in the status route (the undefined status test, 200),
keeping a deleted status on the chat (the deleted status test), dropping a status's earlier time
when a chat comes back to it (the earlier time test), ignoring the tag filter (the tag test), the
person filter (the people test), matching titles only in the folder search (the search test), a
page of ten in the workspace list (the paging test) and answering every chat as editable (the
member test, the owner's chat reads editable).
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from harness.access import grant
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TODO = {"id": "s_todo", "name": "To do", "glyph": "todo", "color": "gray"}
DOING = {"id": "s_doing", "name": "Doing", "glyph": "in_progress", "color": "amber"}
DONE = {"id": "s_done", "name": "Done", "glyph": "done", "color": "green"}
READY_MADE = ("backlog", "todo", "in_progress", "waiting", "done")
WORKSPACE_PAGE = 50
OWNERS = "Owner's"


def _folder(client: httpx.Client, statuses: list[dict] | None = None) -> str:
    created = client.post("/api/v1/folders/", json={"name": f"Board {uuid.uuid4().hex[:8]}"})
    assert created.status_code == 200, created.text
    folder_id = created.json()["id"]
    if statuses is not None:
        _set_statuses(client, folder_id, statuses)
    return folder_id


def _set_statuses(client: httpx.Client, folder_id: str, statuses: list[dict]) -> None:
    updated = client.post(
        f"/api/v1/folders/{folder_id}/update", json={"meta": {"statuses": statuses}}
    )
    assert updated.status_code == 200, updated.text


def _chat(client: httpx.Client, folder_id: str, title: str, text: str = "") -> str:
    messages = {}
    if text:
        messages = {
            "q1": {
                "id": "q1",
                "parentId": None,
                "childrenIds": [],
                "role": "user",
                "content": text,
                "timestamp": int(time.time()),
            }
        }
    chat = {"title": title, "history": {"currentId": "q1" if text else None, "messages": messages}}
    created = client.post("/api/v1/chats/new", json={"chat": chat, "folder_id": folder_id})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _file(client: httpx.Client, folder_id: str, chat_id: str, status: str | None) -> httpx.Response:
    return client.patch(
        f"/api/v1/folders/{folder_id}/chats/{chat_id}/status", json={"status": status}
    )


def _listed(client: httpx.Client, folder_id: str, **params) -> dict:
    response = client.get(
        f"/api/v1/folders/{folder_id}/shared/chats",
        params={"workspace": "true", "page": 1, **params},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _by_title(listing: dict) -> dict[str, dict]:
    return {chat["title"]: chat for chat in listing["chats"]}


def _titles(listing: dict) -> list[str]:
    return sorted(chat["title"] for chat in listing["chats"])


def test_a_chat_filed_under_a_status_lists_with_it_and_filters_by_it(make_user):
    with make_user().client() as client:
        folder_id = _folder(client, [TODO, DONE])
        filed = _chat(client, folder_id, "Filed")
        _chat(client, folder_id, "Unfiled")

        answered = _file(client, folder_id, filed, DONE["id"])
        listing = _by_title(_listed(client, folder_id))
        done_only = _listed(client, folder_id, status=DONE["id"])
        no_status = _listed(client, folder_id, status="none")
        either = _listed(client, folder_id, status=[DONE["id"], "none"])

    assert answered.status_code == 200, answered.text
    assert listing["Filed"]["status"] == DONE["id"], listing["Filed"]
    assert isinstance(listing["Filed"]["status_since"], int), listing["Filed"]
    assert listing["Unfiled"]["status"] is None, listing["Unfiled"]
    assert listing["Filed"]["can_edit_status"] is True
    assert _titles(done_only) == ["Filed"], done_only
    assert _titles(no_status) == ["Unfiled"], no_status
    assert _titles(either) == ["Filed", "Unfiled"], either
    assert (done_only["total"], no_status["total"]) == (1, 1)


def test_a_status_the_folder_does_not_define_is_refused(make_user):
    with make_user().client() as client:
        folder_id = _folder(client, [TODO])
        chat_id = _chat(client, folder_id, "Kept")

        answered = _file(client, folder_id, chat_id, "s_missing")
        filtered = client.get(
            f"/api/v1/folders/{folder_id}/shared/chats",
            params={"workspace": "true", "page": 1, "status": "s_missing"},
        )
        listing = _by_title(_listed(client, folder_id))

    assert answered.status_code == 422, f"an undefined status was taken: {answered.text}"
    assert filtered.status_code == 422, filtered.text
    assert listing["Kept"]["status"] is None, listing["Kept"]


def test_clearing_a_status_files_the_chat_under_no_status(make_user):
    with make_user().client() as client:
        folder_id = _folder(client, [TODO])
        chat_id = _chat(client, folder_id, "Cleared")
        assert _file(client, folder_id, chat_id, TODO["id"]).status_code == 200

        answered = _file(client, folder_id, chat_id, None)
        no_status = _listed(client, folder_id, status="none")

    assert answered.status_code == 200, answered.text
    assert [(chat["title"], chat["status"]) for chat in no_status["chats"]] == [("Cleared", None)]


def test_a_chat_in_a_deleted_status_reads_as_having_none(make_user):
    with make_user().client() as client:
        folder_id = _folder(client, [TODO, DONE])
        chat_id = _chat(client, folder_id, "Orphaned")
        assert _file(client, folder_id, chat_id, DONE["id"]).status_code == 200

        _set_statuses(client, folder_id, [TODO])
        listing = _by_title(_listed(client, folder_id))
        no_status = _listed(client, folder_id, status="none")

    assert listing["Orphaned"]["status"] is None, (
        f"a chat still shows a status its folder deleted: {listing['Orphaned']}"
    )
    assert _titles(no_status) == ["Orphaned"], no_status


def test_going_back_to_a_status_keeps_its_earlier_time(make_user):
    with make_user().client() as client:
        folder_id = _folder(client, [TODO, DOING])
        chat_id = _chat(client, folder_id, "Back and forth")
        assert _file(client, folder_id, chat_id, TODO["id"]).status_code == 200
        # the clock counts whole seconds
        time.sleep(1.2)
        assert _file(client, folder_id, chat_id, DOING["id"]).status_code == 200
        assert _file(client, folder_id, chat_id, TODO["id"]).status_code == 200

        chat = _by_title(_listed(client, folder_id))["Back and forth"]

    assert chat["status"] == TODO["id"], chat
    assert chat["status_durations"].get(TODO["id"], 0) >= 1, chat
    assert chat["status_since"] <= chat["status_entered_at"] - 1, (
        f"going back to a status started its time from zero: {chat}"
    )


def test_the_folder_search_matches_titles_and_message_text(make_user):
    tag = uuid.uuid4().hex[:6]
    with make_user().client() as client:
        folder_id = _folder(client)
        _chat(client, folder_id, f"Harbour plan {tag}")
        _chat(client, folder_id, "Misc notes", text=f"ask the lighthouse{tag} keeper")
        _chat(client, folder_id, "Unrelated", text="nothing to see")

        by_title = _listed(client, folder_id, search=f"harbour plan {tag}")
        by_text = _listed(client, folder_id, search=f"lighthouse{tag}")

    assert _titles(by_title) == [f"Harbour plan {tag}"], by_title
    assert _titles(by_text) == ["Misc notes"], f"the search missed a message's text: {by_text}"
    assert f"lighthouse{tag}" in (by_text["chats"][0]["search_snippet"] or "").lower(), by_text


def test_tags_filter_the_list_and_are_counted(make_user):
    name = f"alpha{uuid.uuid4().hex[:6]}"
    with make_user().client() as client:
        folder_id = _folder(client)
        tagged = _chat(client, folder_id, "Tagged")
        _chat(client, folder_id, "Untagged")
        added = client.post(f"/api/v1/chats/{tagged}/tags", json={"name": name})
        assert added.status_code == 200, added.text

        with_tag = _listed(client, folder_id, tag_ids=name)
        untagged = _listed(client, folder_id, tag_ids="none")
        counted = client.get(f"/api/v1/folders/{folder_id}/tags")

    assert _titles(with_tag) == ["Tagged"], with_tag
    assert [tag["id"] for tag in with_tag["chats"][0]["tags"]] == [name]
    assert _titles(untagged) == ["Untagged"], untagged
    assert counted.status_code == 200, counted.text
    assert [(tag["id"], tag["count"]) for tag in counted.json()] == [(name, 1)], counted.json()


def test_a_tag_colour_is_kept_and_only_the_tag_owner_sets_it(make_user):
    name = f"beta{uuid.uuid4().hex[:6]}"
    owner, other = make_user(), make_user()
    with owner.client() as client:
        folder_id = _folder(client)
        tagged = _chat(client, folder_id, "Coloured")
        untagged = _chat(client, folder_id, "Plain")
        assert client.post(f"/api/v1/chats/{tagged}/tags", json={"name": name}).status_code == 200

        coloured = client.patch(
            f"/api/v1/chats/{tagged}/tags", json={"name": name, "color": "blue"}
        )
        not_on_chat = client.patch(
            f"/api/v1/chats/{untagged}/tags", json={"name": name, "color": "red"}
        )
        malformed = client.patch(
            f"/api/v1/chats/{tagged}/tags", json={"name": name, "color": "Red; x"}
        )
    with other.client() as client:
        stranger = client.patch(f"/api/v1/chats/{tagged}/tags", json={"name": name, "color": "red"})
    with owner.client() as client:
        listed_tag = _by_title(_listed(client, folder_id))["Coloured"]["tags"][0]

    assert coloured.status_code == 200, coloured.text
    assert (not_on_chat.status_code, malformed.status_code, stranger.status_code) == (403, 422, 403)
    assert listed_tag["color"] == "blue", f"the tag lost or changed its colour: {listed_tag}"


def test_the_list_pages_in_fifties_sorted_by_title_both_ways(make_user):
    titles = [f"Card {index:02d}" for index in range(WORKSPACE_PAGE + 3)]
    with make_user().client() as client:
        folder_id = _folder(client)
        for title in titles:
            _chat(client, folder_id, title)

        first = _listed(client, folder_id, sort_by="title", sort_dir="asc")
        second = _listed(client, folder_id, sort_by="title", sort_dir="asc", page=2)
        descending = _listed(client, folder_id, sort_by="title", sort_dir="desc")
        unsortable = client.get(
            f"/api/v1/folders/{folder_id}/shared/chats",
            params={"workspace": "true", "page": 1, "sort_by": "user_id"},
        )

    assert [chat["title"] for chat in first["chats"]] == titles[:WORKSPACE_PAGE]
    assert (first["total"], first["has_more"]) == (len(titles), True)
    assert [chat["title"] for chat in second["chats"]] == titles[WORKSPACE_PAGE:]
    assert second["has_more"] is False
    assert [chat["title"] for chat in descending["chats"]][:3] == titles[::-1][:3]
    assert unsortable.status_code == 422, unsortable.text


def test_the_activity_filter_separates_unread_from_read_chats(make_user):
    with make_user().client() as client:
        folder_id = _folder(client)
        _chat(client, folder_id, "Seen")
        fresh = _chat(client, folder_id, "Fresh")
        marked = client.post(f"/api/v1/chats/{fresh}/unread")
        assert marked.status_code == 200, marked.text

        unread = _listed(client, folder_id, activity="unread")
        idle = _listed(client, folder_id, activity="idle")

    assert _titles(unread) == ["Fresh"], unread
    assert _titles(idle) == ["Seen"], idle


def test_a_new_folder_offers_the_ready_made_statuses_and_an_empty_list_offers_none(make_user):
    with make_user().client() as client:
        ready_made = _folder(client)
        opted_out = _folder(client, [])
        chats = {
            folder_id: _chat(client, folder_id, "Intake") for folder_id in (ready_made, opted_out)
        }

        filed = {
            status: _file(client, ready_made, chats[ready_made], status).status_code
            for status in READY_MADE
        }
        done = _listed(client, ready_made, status="done")
        refused = _file(client, opted_out, chats[opted_out], "todo")

    assert filed == dict.fromkeys(READY_MADE, 200), (
        f"a folder without statuses of its own refused a ready-made one: {filed}"
    )
    assert [(chat["title"], chat["status"]) for chat in done["chats"]] == [("Intake", "done")], done
    assert refused.status_code == 422, (
        f"a folder that set no statuses took a ready-made one: {refused.text}"
    )


def _shared_folder(owner: Actor, member: Actor, *permissions: str) -> str:
    with owner.client() as client:
        folder_id = _folder(client, [TODO, DONE])
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={
                "access_grants": [
                    grant("user", member.id, permission)
                    for permission in permissions or ("read", "write")
                ]
            },
        )
    assert shared.status_code == 200, shared.text
    return folder_id


def test_a_writing_member_files_any_chat_but_tags_only_their_own(make_user):
    owner, member = make_user(), make_user()
    folder_id = _shared_folder(owner, member)
    with owner.client() as client:
        owners = _chat(client, folder_id, OWNERS)
    with member.client() as client:
        own = _chat(client, folder_id, "Member's")
        filed = _file(client, folder_id, own, DONE["id"])
        filed_owners = _file(client, folder_id, owners, TODO["id"])
        listing = _by_title(_listed(client, folder_id))

    assert (filed.status_code, filed_owners.status_code) == (200, 200), (
        f"a writing member could not file a chat: {filed.text} / {filed_owners.text}"
    )
    assert (listing["Member's"]["status"], listing[OWNERS]["status"]) == (DONE["id"], TODO["id"])
    assert (listing["Member's"]["can_edit_status"], listing["Member's"]["can_edit_tags"]) == (
        True,
        True,
    )
    assert (listing[OWNERS]["can_edit_status"], listing[OWNERS]["can_edit_tags"]) == (
        True,
        False,
    ), f"a writing member was offered the wrong controls on the owner's chat: {listing[OWNERS]}"


def test_a_reading_member_files_no_chat_and_reads_the_owners_as_fixed(make_user):
    owner, reader = make_user(), make_user()
    folder_id = _shared_folder(owner, reader, "read")
    with owner.client() as client:
        owners = _chat(client, folder_id, OWNERS)
    with reader.client() as client:
        refused = _file(client, folder_id, owners, DONE["id"])
        listing = _by_title(_listed(client, folder_id))

    assert refused.status_code == 403, f"a reading member filed the owner's chat: {refused.text}"
    assert listing[OWNERS]["status"] is None, listing[OWNERS]
    assert (listing[OWNERS]["can_edit_status"], listing[OWNERS]["can_edit_tags"]) == (
        False,
        False,
    ), f"a reading member was offered to change the owner's chat: {listing[OWNERS]}"


def test_the_people_list_counts_chats_and_filters_by_person(make_user):
    owner, member = make_user(), make_user()
    folder_id = _shared_folder(owner, member)
    with owner.client() as client:
        _chat(client, folder_id, "Owner one")
        _chat(client, folder_id, "Owner two")
    with member.client() as client:
        _chat(client, folder_id, "Member one", text="a question")
    with owner.client() as client:
        people = client.get(f"/api/v1/folders/{folder_id}/users")
        members_chats = _listed(client, folder_id, user_ids=member.id)

    assert people.status_code == 200, people.text
    assert [(person["id"], person["count"]) for person in people.json()] == [
        (owner.id, 2),
        (member.id, 1),
    ], people.json()
    assert _titles(members_chats) == ["Member one"], members_chats
