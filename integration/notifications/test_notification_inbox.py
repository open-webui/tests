"""Journey: the inbox tells people about comments on their items, mentions and replies.

v0.13.0 (3814b7b2f, #19539) added a notification inbox: a comment on your item, a reply to your
comment, a mention in a comment or a channel and a reply to your channel message each put an
entry in your inbox, never in the inbox of the person who acted or of anyone who cannot open
the item or channel. The inbox lists the entries with an unread count; marking one read or
opening the inbox brings the count down, and nobody marks another person's entry. It is on by
default, and with it switched off (Admin Settings > General or `ENABLE_NOTIFICATIONS`) the inbox
routes refuse and nothing that happens meanwhile lands in anyone's inbox.

Discriminates: fails on v0.12.5 (the admin settings have no Comments or inbox switch); in a
backend copy of v0.13.0, adding the actor to the recipients in `comment_notifications` turns the
comment, reply and mention tests red (the actor is notified), skipping the access check in
`eligible_recipients` together with the resource filter in `visible_clause` turns the comment and
channel mention tests red (the stranger is notified; either check alone keeps them green),
notifying nobody in `channel_notifications` turns the channel test red, dropping `seen_at` from
the update in `read_inbox_item` turns the unread count test red, and dropping the
`notifications.enable` check from `consume_event` and `insert_notifications` turns both
switched-off tests red.

Since 7efb494a2, opening a channel (the page tells the server it read up to now) also marks that
channel's entries in the inbox read. The channel opening test passes on dev e8c0ad85e and fails
on a backend copy with 7efb494a2 reverted (the mention stays unread).
"""

from __future__ import annotations

import pytest

from harness.actors import Actor, admin_of, create_user
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.comments import COMMENTABLE, comment, mention, reply, shared_item, switch_comments
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

INBOX = "/api/v1/notifications/inbox"
ADMIN_CONFIG = "/api/v1/auths/admin/config"


@pytest.fixture
def inbox_on(admin, preserve):
    preserve("admin_config")
    switch_comments(admin, True)
    switch_notifications(admin, True)


def switch_notifications(admin: Actor, on: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        assert current.status_code == 200, current.text
        assert "ENABLE_NOTIFICATIONS" in current.json(), "the admin settings have no inbox switch"
        saved = client.post(ADMIN_CONFIG, json={**current.json(), "ENABLE_NOTIFICATIONS": on})
    assert saved.status_code == 200, saved.text
    assert saved.json()["ENABLE_NOTIFICATIONS"] is on, (
        f"the inbox switch did not take: {saved.text}"
    )


def inbox(actor: Actor) -> list[dict]:
    with actor.client() as client:
        listed = client.get(INBOX, params={"limit": 50})
    assert listed.status_code == 200, f"{actor.name} could not open the inbox: {listed.text}"
    return listed.json()["items"]


def unread(actor: Actor) -> int:
    with actor.client() as client:
        counted = client.get(f"{INBOX}/count")
    assert counted.status_code == 200, counted.text
    return counted.json()["count"]


def entries_about(actor: Actor, object_id: str) -> list[tuple[str, str]]:
    """Reason and actor of each of the account's inbox entries about this comment or message."""
    return [
        (entry["event_type"], (entry["actor"] or {}).get("id"))
        for entry in inbox(actor)
        if entry["object_id"] == object_id
    ]


def _posted(response, who: str) -> dict:
    assert response.status_code == 200, f"{who} could not comment: {response.text}"
    return response.json()


@pytest.mark.parametrize("kind", list(COMMENTABLE))
def test_a_comment_on_my_item_reaches_my_inbox_only(kind, admin, make_user, inbox_on):
    owner, commenter, other_reader, stranger = make_user(), make_user(), make_user(), make_user()
    item_id = shared_item(kind, owner, admin, [commenter, other_reader])

    posted = _posted(comment(commenter, kind, item_id, f"a thought on your {kind}"), "commenter")

    assert entries_about(owner, posted["id"]) == [("comment.created", commenter.id)], (
        f"the owner's inbox does not show the comment on their {kind}"
    )
    entry = next(e for e in inbox(owner) if e["object_id"] == posted["id"])
    assert (entry["resource_type"], entry["resource_id"]) == (kind, item_id)
    assert entry["preview"] == f"a thought on your {kind}"
    for bystander in (commenter, other_reader, stranger):
        assert entries_about(bystander, posted["id"]) == [], (
            f"a comment on someone else's {kind} reached the inbox of {bystander.name}"
        )


def test_a_reply_to_my_comment_reaches_me_and_the_owner_only(admin, make_user, inbox_on):
    owner, author, replier, stranger = make_user(), make_user(), make_user(), make_user()
    item_id = shared_item("note", owner, admin, [author, replier])
    started = _posted(comment(author, "note", item_id, "Should we split this?"), "author")

    answered = _posted(reply(replier, started["id"], "Yes, in two."), "replier")

    assert entries_about(author, answered["id"]) == [("comment.replied", replier.id)], (
        "the comment's author was not told about the reply"
    )
    assert entries_about(owner, answered["id"]) == [("comment.replied", replier.id)], (
        "the note's owner was not told about the reply in its thread"
    )
    assert entries_about(replier, answered["id"]) == [], (
        "the replier was told about their own reply"
    )
    assert entries_about(stranger, answered["id"]) == []


def test_a_mention_in_a_comment_reaches_only_a_mentioned_reader(admin, make_user, inbox_on):
    owner, mentioned, unmentioned, stranger = make_user(), make_user(), make_user(), make_user()
    item_id = shared_item("prompt", owner, admin, [mentioned, unmentioned])

    text = f"{mention(mentioned)} and {mention(stranger)}, could you look at this?"
    posted = _posted(comment(owner, "prompt", item_id, text), "owner")

    assert entries_about(mentioned, posted["id"]) == [("comment.mentioned", owner.id)], (
        "the mentioned reader was not told"
    )
    assert entries_about(owner, posted["id"]) == [], "the author was told about their own mention"
    assert entries_about(unmentioned, posted["id"]) == []
    assert entries_about(stranger, posted["id"]) == [], (
        "a mention told someone who cannot open the prompt about it"
    )


def test_a_channel_reply_or_mention_reaches_only_the_person_meant(admin, make_user, inbox_on):
    enable_channels(admin)
    owner, author, replier, stranger = make_user(), make_user(), make_user(), make_user()
    channel_id = group_channel(owner, author, replier)
    asked = post_message(author, channel_id, "Who has the release notes?")

    answered = post_message(replier, channel_id, "I do.", parent_id=asked)

    assert entries_about(author, answered) == [("channel.replied", replier.id)], (
        "the message's author was not told about the reply in its thread"
    )
    for bystander in (replier, owner, stranger):
        assert entries_about(bystander, answered) == [], (
            f"a reply to someone else's message reached the inbox of {bystander.name}"
        )

    called = post_message(
        replier, channel_id, f"{mention(owner)} and {mention(stranger)}, see above"
    )

    assert entries_about(owner, called) == [("channel.mentioned", replier.id)], (
        "the mentioned member was not told"
    )
    for bystander in (author, replier, stranger):
        assert entries_about(bystander, called) == [], (
            f"a channel mention reached the inbox of {bystander.name}, who was not meant"
        )


def test_opening_the_channel_marks_its_entries_read(admin, make_user, inbox_on):
    enable_channels(admin)
    owner, author = make_user(), make_user()
    channel_id = group_channel(owner, author)
    called = post_message(author, channel_id, f"{mention(owner)}, the ferry is late")
    assert unread(owner) == 1
    with connected(owner) as tab:
        tab.join_channels(owner)
        tab.call(
            "events:channel",
            {"channel_id": channel_id, "message_id": None, "data": {"type": "last_read_at"}},
        )

    [entry] = [entry for entry in inbox(owner) if entry["object_id"] == called]
    assert entry["read_at"], f"opening the channel left its mention unread: {entry}"
    assert unread(owner) == 0, "opening the channel left its mention in the unread count"


def test_marking_read_and_opening_the_inbox_bring_the_unread_count_down(admin, make_user, inbox_on):
    owner, commenter = make_user(), make_user()
    item_id = shared_item("knowledge", owner, admin, [commenter])
    assert unread(owner) == 0
    posted = [
        _posted(comment(commenter, "knowledge", item_id, f"note {n}"), "commenter")["id"]
        for n in range(3)
    ]
    assert unread(owner) == 3

    first = next(e for e in inbox(owner) if e["object_id"] == posted[0])
    with commenter.client() as client:
        someone_elses = client.patch(f"{INBOX}/{first['id']}")
    assert someone_elses.status_code == 404, (
        f"another account marked the entry: {someone_elses.text}"
    )
    assert unread(owner) == 3

    with owner.client() as client:
        marked = client.patch(f"{INBOX}/{first['id']}")
    assert marked.status_code == 200, marked.text
    assert marked.json()["read_at"], f"the entry did not read back as read: {marked.json()}"
    assert unread(owner) == 2

    with owner.client() as client:
        opened = client.post(f"{INBOX}/open")
    assert opened.status_code == 200, opened.text
    assert opened.json()["count"] == 0
    assert unread(owner) == 0
    assert {e["object_id"] for e in inbox(owner)} >= set(posted), "opening the inbox lost entries"


def test_inbox_starts_on_and_while_off_nothing_reaches_it(admin, make_user, preserve):
    with admin.client() as client:
        config = client.get(ADMIN_CONFIG)
    assert config.status_code == 200, config.text
    assert config.json().get("ENABLE_NOTIFICATIONS") is True, (
        f"the inbox does not start on: {config.json().get('ENABLE_NOTIFICATIONS')!r}"
    )

    preserve("admin_config")
    switch_comments(admin, True)
    owner, reader = make_user(), make_user()
    item_id = shared_item("note", owner, admin, [reader])
    switch_notifications(admin, False)

    with owner.client() as client:
        refused = {
            "list": client.get(INBOX).status_code,
            "count": client.get(f"{INBOX}/count").status_code,
            "open": client.post(f"{INBOX}/open").status_code,
        }
    assert refused == dict.fromkeys(refused, 404), f"the switched-off inbox answered: {refused}"
    quiet = _posted(comment(reader, "note", item_id, f"{mention(owner)} while off"), "reader")
    quiet_reply = _posted(reply(owner, quiet["id"], "also while off"), "owner")

    switch_notifications(admin, True)
    assert entries_about(owner, quiet["id"]) == [], "a comment made while off reached the inbox"
    assert entries_about(reader, quiet_reply["id"]) == [], (
        "a reply made while off reached the inbox"
    )
    assert unread(owner) == 0 and unread(reader) == 0
    loud = _posted(comment(reader, "note", item_id, "after switching back on"), "reader")
    assert entries_about(owner, loud["id"]) == [("comment.created", reader.id)]


@pytest.mark.slow
def test_the_environment_switch_turns_the_inbox_off(instance_with):
    launched = instance_with({"ENABLE_NOTIFICATIONS": "false", "ENABLE_COMMENTS": "true"})
    admin = admin_of(launched)
    owner, reader = create_user(launched), create_user(launched)
    with admin.client() as client:
        config = client.get(ADMIN_CONFIG)
    assert config.json().get("ENABLE_NOTIFICATIONS") is False, (
        "ENABLE_NOTIFICATIONS=false left the inbox on"
    )
    with owner.client() as client:
        assert client.get(INBOX).status_code == 404

    item_id = shared_item("chat", owner, admin, [reader])
    quiet = _posted(comment(reader, "chat", item_id, "while the inbox is off"), "reader")
    switch_notifications(admin, True)
    assert entries_about(owner, quiet["id"]) == [], "a comment made while off reached the inbox"
