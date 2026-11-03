"""Journey: people discuss the chats, notes and workspace items they can open in comment threads.

v0.13.0 (9d1b4ef36) added comments with replies, reactions and mentions on chats, notes, models,
knowledge bases, prompts, skills and tools. They are off until an admin switches on Comments or
the instance starts with `ENABLE_COMMENTS`, and while off every comment route refuses, existing
comments included. With them on, anyone who can open an item reads its comments and adds their
own, while a stranger can neither read, post, reply nor react. Only the author edits a comment;
the author, the item's owner or an admin deletes one, and a deleted comment reads back without
its text. Replies and reactions read back with their counts, the mention list offers only people
who can open the item, and deleting the item takes its comments with it, so a new preset that
takes a deleted one's id starts with an empty thread.

Discriminates: fails on v0.12.5 (the admin settings have no Comments switch); in a backend copy of
v0.13.0, dropping the `comments.enable` check from `comment_resource` fails the switched-off
test, skipping its `if not allowed` refusal fails every stranger row and the mention test,
dropping the author check from the edit handler or the author-owner-admin check from the delete
handler fails the edit and delete test, skipping the access check in the mentions candidate loop
fails the mention test (the stranger is offered), storing a reply without its `parent_id` or
answering every comment with no reactions fails the replies and reactions test, and
`delete_resource_comments` keeping the comments fails the deletion test.
"""

from __future__ import annotations

import uuid

import pytest

from harness.actors import admin_of, create_user
from harness.comments import (
    COMMENTABLE,
    comment,
    delete,
    edit,
    mention,
    mentions,
    react,
    read,
    replies,
    reply,
    shared_item,
    switch_comments,
    thread,
    unreact,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

KINDS = list(COMMENTABLE)
NOT_FOUND = 404


@pytest.fixture
def comments_on(admin, preserve):
    preserve("admin_config")
    switch_comments(admin, True)


def _posted(response, who: str) -> dict:
    assert response.status_code == 200, f"{who} could not comment: {response.text}"
    return response.json()


def _contents(response) -> list[str]:
    assert response.status_code == 200, response.text
    return [entry["content"] for entry in response.json()["items"]]


def test_comments_start_off_and_while_off_every_comment_route_refuses(admin, make_user, preserve):
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config")
    assert config.status_code == 200, config.text
    assert config.json().get("ENABLE_COMMENTS") is False, (
        f"Comments are not a switch that starts off: {config.json().get('ENABLE_COMMENTS')!r}"
    )

    preserve("admin_config")
    owner, reader = make_user(), make_user()
    note_id = shared_item("note", owner, admin, [reader])
    switch_comments(admin, True)
    posted = _posted(comment(reader, "note", note_id, "written while comments were on"), "reader")
    switch_comments(admin, False)

    answered = {
        "list": thread(owner, "note", note_id).status_code,
        "post": comment(owner, "note", note_id, "written while off").status_code,
        "read": read(owner, posted["id"]).status_code,
        "replies": replies(owner, posted["id"]).status_code,
        "reply": reply(owner, posted["id"], "replied while off").status_code,
        "edit": edit(reader, posted, "edited while off").status_code,
        "delete": delete(reader, posted).status_code,
        "react": react(owner, posted["id"], "👍").status_code,
        "mentions": mentions(owner, "note", note_id, "").status_code,
    }

    assert answered == dict.fromkeys(answered, 403), (
        f"with Comments switched off the comment routes still answered: {answered}"
    )
    switch_comments(admin, True)
    assert _contents(thread(owner, "note", note_id)) == ["written while comments were on"], (
        "a request sent while Comments were off changed the thread"
    )
    kept = read(owner, posted["id"]).json()
    assert (kept["reply_count"], kept["reactions"]) == (0, []), (
        f"a reply or reaction sent while Comments were off was stored: {kept}"
    )


@pytest.mark.slow
def test_the_environment_switch_turns_comments_on(instance_with):
    launched = instance_with({"ENABLE_COMMENTS": "true"})
    owner = create_user(launched)
    with admin_of(launched).client() as client:
        config = client.get("/api/v1/auths/admin/config")
    assert config.json().get("ENABLE_COMMENTS") is True, "ENABLE_COMMENTS=true left the switch off"

    chat_id = shared_item("chat", owner, admin_of(launched), [])
    _posted(comment(owner, "chat", chat_id, "on from the start"), "the chat's owner")

    assert _contents(thread(owner, "chat", chat_id)) == ["on from the start"]


@pytest.mark.parametrize("kind", KINDS)
def test_someone_who_can_open_the_item_comments_and_a_stranger_cannot(
    kind, admin, make_user, comments_on
):
    owner, reader, stranger = make_user(), make_user(), make_user()
    item_id = shared_item(kind, owner, admin, [reader])

    from_reader = _posted(comment(reader, kind, item_id, f"the reader on the {kind}"), "reader")
    from_owner = _posted(comment(owner, kind, item_id, f"the owner on the {kind}"), "owner")

    expected = [f"the reader on the {kind}", f"the owner on the {kind}"]
    assert _contents(thread(owner, kind, item_id)) == expected, f"the owner's {kind} thread"
    assert _contents(thread(reader, kind, item_id)) == expected, f"the reader's {kind} thread"
    assert from_reader["user"]["id"] == reader.id, (
        f"the reader's comment lost its author: {from_reader}"
    )

    refused = {
        "list": thread(stranger, kind, item_id).status_code,
        "post": comment(stranger, kind, item_id, "the stranger").status_code,
        "read": read(stranger, from_owner["id"]).status_code,
        "replies": replies(stranger, from_owner["id"]).status_code,
        "reply": reply(stranger, from_owner["id"], "the stranger replies").status_code,
        "react": react(stranger, from_owner["id"], "👀").status_code,
    }
    assert refused == dict.fromkeys(refused, NOT_FOUND), (
        f"a stranger got into the comments of a {kind}: {refused}"
    )
    assert _contents(thread(owner, kind, item_id)) == expected, (
        f"a stranger's request changed the {kind}'s thread"
    )
    owners_view = read(owner, from_owner["id"]).json()
    assert owners_view["reply_count"] == 0 and owners_view["reactions"] == [], (
        f"a stranger's reply or reaction landed on the {kind}'s comment: {owners_view}"
    )


def test_only_the_author_edits_and_the_author_owner_or_admin_deletes(admin, make_user, comments_on):
    owner, author, other = make_user(), make_user(), make_user()
    item_id = shared_item("knowledge", owner, admin, [author, other])
    posted = _posted(comment(author, "knowledge", item_id, "the author's first words"), "author")

    assert edit(other, posted, "rewritten by another reader").status_code == 403
    assert edit(owner, posted, "rewritten by the owner").status_code == 403
    assert edit(admin, posted, "rewritten by the admin").status_code == 403
    assert delete(other, posted).status_code == 403
    assert _contents(thread(owner, "knowledge", item_id)) == ["the author's first words"], (
        "a refused edit or delete changed the comment"
    )

    edited = edit(author, posted, "the author's second thoughts")
    assert edited.status_code == 200, edited.text
    assert _contents(thread(other, "knowledge", item_id)) == ["the author's second thoughts"]
    stale = edit(author, posted, "an edit from an outdated copy")
    assert stale.status_code == 409, f"an edit over a newer version was taken: {stale.text}"

    removed = delete(author, edited.json())
    assert removed.status_code == 200, removed.text
    gone = read(other, posted["id"]).json()
    assert gone["deleted_at"] and gone["content"] == "", f"a deleted comment kept its text: {gone}"
    assert thread(owner, "knowledge", item_id).json()["total"] == 0

    for remover in (owner, admin):
        by_other = _posted(comment(other, "knowledge", item_id, "to be removed"), "other reader")
        removed = delete(remover, by_other)
        assert removed.status_code == 200, f"{remover.role} could not delete: {removed.text}"
        assert read(other, by_other["id"]).json()["content"] == ""


def test_replies_and_reactions_read_back(admin, make_user, comments_on):
    owner, reader = make_user(), make_user()
    item_id = shared_item("prompt", owner, admin, [reader])
    started = _posted(comment(owner, "prompt", item_id, "Is the tone right?"), "owner")

    first = reply(reader, started["id"], "A bit formal.")
    assert first.status_code == 200, first.text
    second = reply(owner, started["id"], "I will loosen it.", reply_to=first.json()["id"])
    assert second.status_code == 200, second.text

    assert _contents(replies(reader, started["id"])) == ["A bit formal.", "I will loosen it."]
    assert second.json()["reply_to_id"] == first.json()["id"]
    starting = thread(reader, "prompt", item_id).json()
    assert [entry["content"] for entry in starting["items"]] == ["Is the tone right?"], (
        "a reply showed up as a starting comment"
    )
    assert starting["items"][0]["reply_count"] == 2

    for actor, name in ((reader, "👍"), (owner, "👍"), (owner, "🎉"), (owner, "🎉")):
        reacted = react(actor, started["id"], name)
        assert reacted.status_code == 200, reacted.text
    assert sorted(
        (entry["name"], entry["count"], entry["reacted"])
        for entry in read(reader, started["id"]).json()["reactions"]
    ) == [("🎉", 1, False), ("👍", 2, True)]

    assert unreact(reader, started["id"], "👍").status_code == 200
    assert sorted(
        (entry["name"], entry["count"], entry["reacted"])
        for entry in read(owner, started["id"]).json()["reactions"]
    ) == [("🎉", 1, True), ("👍", 1, True)]


def test_the_mention_list_offers_only_people_who_can_open_the_item(admin, make_user, comments_on):
    tag = uuid.uuid4().hex[:8]
    owner = make_user(name=f"Owner {tag}")
    reader = make_user(name=f"Reader {tag}")
    stranger = make_user(name=f"Stranger {tag}")
    item_id = shared_item("skill", owner, admin, [reader])

    offered = mentions(owner, "skill", item_id, tag)
    assert offered.status_code == 200, offered.text
    assert sorted(entry["id"] for entry in offered.json()) == sorted([owner.id, reader.id]), (
        f"the mention list offered {offered.json()}"
    )
    assert stranger.id not in {entry["id"] for entry in offered.json()}

    text = f"{mention(reader)} could you check this?"
    _posted(comment(owner, "skill", item_id, text), "owner")
    assert _contents(thread(reader, "skill", item_id)) == [text]


def test_deleting_the_item_deletes_its_comments(admin, make_user, comments_on):
    owner, reader = make_user(), make_user()
    model_id = shared_item("model", owner, admin, [reader])
    _posted(comment(reader, "model", model_id, "about to go"), "reader")

    with owner.client() as client:
        deleted = client.post("/api/v1/models/model/delete", json={"id": model_id})
        assert deleted.status_code == 200, deleted.text
        # a preset's id is chosen by its creator, so a new one can take the old id
        recreated = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": "New preset",
                "meta": {},
                "params": {},
            },
        )
    assert recreated.status_code == 200, recreated.text

    assert _contents(thread(owner, "model", model_id)) == [], (
        "a deleted preset's comments came back on a new preset with its id"
    )
