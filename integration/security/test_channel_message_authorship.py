"""Regression: channel write access must not double as a moderation capability.

open-webui 0.11.0 fix `c609ec411` (#27197): the message update and delete routes gated a
standard channel on one OR condition, so write access on the channel was enough by itself. Any
member who could post could rewrite or delete someone else's message, and an edited message
kept the original author's name. The same OR let an author whose write access was revoked keep
editing. The fix checks write access first and authorship second.

Twin of unit/security/test_channel_message_authorship.py. No browser twin: the channel page
only offers edit and delete on a person's own messages on both refs, so the fix is only
reachable over the API.

Discriminates: passes on dev bbfa876af; with c609ec411 reverted a member with write access edits
and deletes another member's message, and an author without write access edits their own.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ORIGINAL = "the author's original words"


def _grants(*entries: tuple[str, str]) -> list[dict]:
    return [
        {"principal_type": "user", "principal_id": user_id, "permission": permission}
        for user_id, permission in entries
    ]


def _read_write(*users) -> list[tuple[str, str]]:
    return [(user.id, permission) for user in users for permission in ("read", "write")]


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        enabled = {**config, "ENABLE_CHANNELS": True}
        client.post("/api/v1/auths/admin/config", json=enabled).raise_for_status()


@pytest.fixture
def author(make_user):
    return make_user()


@pytest.fixture
def member(make_user):
    return make_user()


def _create_channel(client, kind: str, others) -> str:
    form = {"name": f"{kind}-room", "type": None if kind == "standard" else kind}
    if kind == "standard":
        form["access_grants"] = _grants(*_read_write(*others))
    else:
        form["user_ids"] = [other.id for other in others]
    created = client.post("/api/v1/channels/create", json=form)
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.fixture(params=["standard", "group", "dm"])
def channel(request, channels_on, admin, author, member):
    """A channel of each kind in which both the author and the member may post."""
    if request.param == "standard":
        with admin.client() as client:
            return _create_channel(client, "standard", [author, member])
    with author.client() as client:
        return _create_channel(client, request.param, [member])


@pytest.fixture
def standard_channel(channels_on, admin, author, member):
    with admin.client() as client:
        return _create_channel(client, "standard", [author, member])


def _post(actor, channel_id: str) -> str:
    with actor.client() as client:
        posted = client.post(
            f"/api/v1/channels/{channel_id}/messages/post", json={"content": ORIGINAL}
        )
    assert posted.status_code == 200, posted.text
    return posted.json()["id"]


def _edit(actor, channel_id: str, message_id: str):
    with actor.client() as client:
        return client.post(
            f"/api/v1/channels/{channel_id}/messages/{message_id}/update",
            json={"content": "rewritten by someone else"},
        )


def _delete(actor, channel_id: str, message_id: str):
    with actor.client() as client:
        return client.delete(f"/api/v1/channels/{channel_id}/messages/{message_id}/delete")


def _stored(actor, channel_id: str, message_id: str):
    with actor.client() as client:
        return client.get(f"/api/v1/channels/{channel_id}/messages/{message_id}")


def test_write_access_does_not_let_a_member_edit_anothers_message(standard_channel, author, member):
    message_id = _post(author, standard_channel)

    refused = _edit(member, standard_channel, message_id)

    assert refused.status_code == 403, (
        f"a member with write access rewrote another member's message, which stays attributed "
        f"to its author (#27197): HTTP {refused.status_code}"
    )
    assert _stored(author, standard_channel, message_id).json()["content"] == ORIGINAL


def test_write_access_does_not_let_a_member_delete_anothers_message(
    standard_channel, author, member
):
    message_id = _post(author, standard_channel)

    refused = _delete(member, standard_channel, message_id)

    assert refused.status_code == 403, (
        f"a member with write access deleted another member's message (#27197): "
        f"HTTP {refused.status_code}"
    )
    assert _stored(author, standard_channel, message_id).status_code == 200


@pytest.mark.parametrize("action", [_edit, _delete], ids=["edit", "delete"])
def test_authorship_does_not_stand_in_for_revoked_write_access(
    action, standard_channel, admin, author, member
):
    message_id = _post(author, standard_channel)
    with admin.client() as client:
        revoked = _grants((author.id, "read"), *_read_write(member))
        client.post(
            f"/api/v1/channels/{standard_channel}/update",
            json={"name": "standard-room", "access_grants": revoked},
        ).raise_for_status()

    refused = action(author, standard_channel, message_id)

    assert refused.status_code == 403, (
        f"an author whose write access was revoked could still {action.__name__[1:]} their "
        f"message (#27197): HTTP {refused.status_code}"
    )


@pytest.mark.parametrize("action", [_edit, _delete], ids=["edit", "delete"])
def test_no_channel_kind_lets_a_member_change_anothers_message(action, channel, author, member):
    message_id = _post(author, channel)

    assert action(member, channel, message_id).status_code == 403
    assert _stored(author, channel, message_id).json()["content"] == ORIGINAL


@pytest.mark.parametrize("action", [_edit, _delete], ids=["edit", "delete"])
def test_the_author_can_change_their_own_message(action, channel, author):
    message_id = _post(author, channel)

    assert action(author, channel, message_id).status_code == 200


@pytest.mark.parametrize("action", [_edit, _delete], ids=["edit", "delete"])
def test_an_admin_can_change_any_message_in_a_standard_channel(
    action, standard_channel, admin, author
):
    message_id = _post(author, standard_channel)

    assert action(admin, standard_channel, message_id).status_code == 200


def test_a_member_with_write_access_can_still_pin_anothers_message(
    standard_channel, author, member
):
    message_id = _post(author, standard_channel)

    with member.client() as client:
        pinned = client.post(
            f"/api/v1/channels/{standard_channel}/messages/{message_id}/pin",
            json={"is_pinned": True},
        )

    assert pinned.status_code == 200, pinned.text


def test_a_missing_message_is_not_found(standard_channel, author):
    assert _edit(author, standard_channel, "no-such-message").status_code == 404
    assert _delete(author, standard_channel, "no-such-message").status_code == 404
