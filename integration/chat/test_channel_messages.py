"""Journey: talking in a channel over the API, and who may read and write in it.

Two members of a group channel post, edit and delete their messages, react to each other's
and reply in a thread; each change shows in what the other member reads back. Someone outside
the channel is refused every read and write on it and changes nothing. In a standard channel a
read grant opens the messages but not posting or reacting, and no grant opens nothing. That
another member cannot edit or delete a message is pinned by
integration/security/test_channel_message_authorship.py.

Discriminates: in a backend copy, dropping the membership check from the group branch of
`get_channel_messages` turns the listing row of the outsider test red (HTTP 200),
`remove_reaction_by_id_and_user_id_and_name` removing the reaction by name only turns the
reaction test red (one member's removal takes the other's too), `get_messages_by_parent_id`
leaving out the parent turns the thread test red, and asking for `read` instead of `write` in
`new_message_handler` turns the read-grant test red (the reader posts).
"""

from __future__ import annotations

import pytest

from harness.channel_quotes import enable_channels, group_channel, post_message

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

REFUSED = 403


@pytest.fixture
def room(admin, preserve, make_user):
    """A group channel of an author and a member, and someone outside it."""
    preserve("admin_config")
    enable_channels(admin)
    author, member, outsider = make_user(), make_user(), make_user()
    return group_channel(author, member), author, member, outsider


def _get(actor, path: str):
    with actor.client() as client:
        return client.get(path)


def _listing(actor, channel_id: str) -> list[dict]:
    listed = _get(actor, f"/api/v1/channels/{channel_id}/messages")
    assert listed.status_code == 200, listed.text
    return listed.json()


def _react(actor, channel_id: str, message_id: str, action: str, name: str = "heart"):
    with actor.client() as client:
        return client.post(
            f"/api/v1/channels/{channel_id}/messages/{message_id}/reactions/{action}",
            json={"name": name},
        )


def _reactions(actor, channel_id: str) -> list[tuple[str, int, set[str]]]:
    [message] = _listing(actor, channel_id)
    return [
        (reaction["name"], reaction["count"], {entry["id"] for entry in reaction["users"]})
        for reaction in message["reactions"]
    ]


def test_an_author_posts_edits_and_deletes_their_message(room):
    channel_id, author, member, _ = room
    message_id = post_message(author, channel_id, "the kettle is on")

    listed = _listing(member, channel_id)
    with author.client() as client:
        edited = client.post(
            f"/api/v1/channels/{channel_id}/messages/{message_id}/update",
            json={"content": "the kettle has boiled"},
        )
    read_back = _get(member, f"/api/v1/channels/{channel_id}/messages/{message_id}")
    with author.client() as client:
        deleted = client.delete(f"/api/v1/channels/{channel_id}/messages/{message_id}/delete")

    assert [(entry["content"], entry["user"]["name"]) for entry in listed] == [
        ("the kettle is on", author.name)
    ]
    assert edited.status_code == 200, edited.text
    assert read_back.json()["content"] == "the kettle has boiled"
    assert deleted.status_code == 200 and deleted.json() is True, deleted.text
    assert _listing(member, channel_id) == []
    assert _get(member, f"/api/v1/channels/{channel_id}/messages/{message_id}").status_code == 404


def test_each_member_adds_and_removes_their_own_reaction(room):
    channel_id, author, member, _ = room
    message_id = post_message(author, channel_id, "cake in the kitchen")

    assert _react(member, channel_id, message_id, "add").json() is True
    assert _react(author, channel_id, message_id, "add").json() is True
    both = _reactions(member, channel_id)
    assert _react(member, channel_id, message_id, "remove").json() is True
    one = _reactions(author, channel_id)
    assert _react(author, channel_id, message_id, "remove").json() is True

    assert both == [("heart", 2, {author.id, member.id})]
    assert one == [("heart", 1, {author.id})], "removing one member's reaction took another's"
    assert _reactions(member, channel_id) == []


def test_thread_replies_list_under_their_parent(room):
    channel_id, author, member, _ = room
    parent_id = post_message(author, channel_id, "who is in on friday?")
    post_message(member, channel_id, "me", parent_id=parent_id)
    post_message(author, channel_id, "me too", parent_id=parent_id)

    listed = _listing(member, channel_id)
    thread = _get(member, f"/api/v1/channels/{channel_id}/messages/{parent_id}/thread")

    assert [(entry["id"], entry["reply_count"]) for entry in listed] == [(parent_id, 2)]
    assert thread.status_code == 200, thread.text
    assert [entry["content"] for entry in thread.json()] == [
        "me too",
        "me",
        "who is in on friday?",
    ]


# method, path with {channel} and {message}, body
OUTSIDER_ROUTES = [
    ("GET", "/api/v1/channels/{channel}", None),
    ("GET", "/api/v1/channels/{channel}/messages", None),
    ("GET", "/api/v1/channels/{channel}/messages/pinned", None),
    ("GET", "/api/v1/channels/{channel}/messages/{message}", None),
    ("GET", "/api/v1/channels/{channel}/messages/{message}/data", None),
    ("GET", "/api/v1/channels/{channel}/messages/{message}/thread", None),
    ("POST", "/api/v1/channels/{channel}/messages/post", {"content": "let me in"}),
    ("POST", "/api/v1/channels/{channel}/messages/{message}/reactions/add", {"name": "heart"}),
]


@pytest.mark.parametrize(
    "method, path, body", OUTSIDER_ROUTES, ids=[f"{row[0]} {row[1]}" for row in OUTSIDER_ROUTES]
)
def test_someone_outside_the_channel_is_refused(method, path, body, room):
    channel_id, author, member, outsider = room
    message_id = post_message(author, channel_id, "members only")
    post_message(member, channel_id, "in the thread", parent_id=message_id)
    before = _listing(member, channel_id)

    with outsider.client() as client:
        answered = client.request(
            method, path.format(channel=channel_id, message=message_id), json=body
        )

    assert answered.status_code == REFUSED, (
        f"someone outside the channel was not refused {method} {path}: "
        f"HTTP {answered.status_code} {answered.text[:200]}"
    )
    assert "members only" not in answered.text
    assert _listing(member, channel_id) == before


@pytest.fixture
def standard_channel(admin, preserve, make_user):
    """A standard channel the admin opened, a reader with a read grant and a user with none."""
    preserve("admin_config")
    enable_channels(admin)
    reader, ungranted = make_user(), make_user()
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with admin.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={"name": "announcements", "type": None, "access_grants": [grant]},
        )
    assert created.status_code == 200, created.text
    channel_id = created.json()["id"]
    message_id = post_message(admin, channel_id, "the office is closed on monday")
    return channel_id, message_id, reader, ungranted


def test_a_read_grant_opens_a_standard_channel_for_reading_only(standard_channel):
    channel_id, message_id, reader, _ = standard_channel

    listed = _listing(reader, channel_id)
    with reader.client() as client:
        posted = client.post(
            f"/api/v1/channels/{channel_id}/messages/post", json={"content": "noted"}
        )
    reacted = _react(reader, channel_id, message_id, "add")

    assert [entry["content"] for entry in listed] == ["the office is closed on monday"]
    assert posted.status_code == REFUSED, f"a reader posted: HTTP {posted.status_code}"
    assert reacted.status_code == REFUSED, f"a reader reacted: HTTP {reacted.status_code}"
    assert [entry["content"] for entry in _listing(reader, channel_id)] == [
        "the office is closed on monday"
    ]


def test_no_grant_opens_nothing_in_a_standard_channel(standard_channel):
    channel_id, message_id, _, ungranted = standard_channel

    listed = _get(ungranted, f"/api/v1/channels/{channel_id}/messages")
    read = _get(ungranted, f"/api/v1/channels/{channel_id}/messages/{message_id}")

    assert listed.status_code == REFUSED, listed.text
    assert read.status_code == REFUSED, read.text
