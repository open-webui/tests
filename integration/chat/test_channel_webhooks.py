"""Journey: a channel's incoming webhooks, from the manager's settings to a message in the channel.

The channel's manager, or an admin, creates a webhook and gets its token, lists and deletes it;
a plain member of the same channel can do none of that and never sees a token, not in the
webhook routes and not in anything the channel shows them. An outside service then posts with
the webhook's id and token and no account: the message lands in the channel under the webhook's
name. A wrong token, a deleted webhook and channels switched off are all refused, and nothing
lands.

Discriminates: in a backend copy, dropping the manager check from `create_channel_webhook` turns
the member test red (the member creates one), `get_webhook_by_id_and_token` ignoring the token
turns the wrong-token test red and `post_webhook_message` skipping `check_channels_access` turns
the switched-off test red.
"""

from __future__ import annotations

import json

import httpx
import pytest

from harness.channel_quotes import ADMIN_CONFIG, enable_channels, group_channel

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def channel(admin, preserve, make_user):
    """A group channel a fresh account manages: its id, the manager and a plain member."""
    preserve("admin_config")
    enable_channels(admin)
    manager, member = make_user(), make_user()
    return group_channel(manager, member), manager, member


def _webhook(actor, channel_id: str) -> dict:
    with actor.client() as client:
        created = client.post(
            f"/api/v1/channels/{channel_id}/webhooks/create", json={"name": "Build bot"}
        )
    assert created.status_code == 200, created.text
    return created.json()


def _post_as_webhook(actor, webhook: dict, content: str, token: str | None = None):
    """Post the way an outside service does: the id and token in the path, no account."""
    url = f"{actor.base_url}/api/v1/channels/webhooks/{webhook['id']}/{token or webhook['token']}"
    return httpx.post(url, json={"content": content}, timeout=60.0)


def _messages(actor, channel_id: str) -> list[dict]:
    with actor.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    assert listed.status_code == 200, listed.text
    return listed.json()


@pytest.mark.parametrize("caller", ["manager", "admin"])
def test_a_manager_or_an_admin_creates_lists_and_deletes_a_webhook(channel, admin, caller):
    channel_id, manager, _ = channel
    actor = manager if caller == "manager" else admin

    webhook = _webhook(actor, channel_id)
    with actor.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/webhooks")
        deleted = client.delete(f"/api/v1/channels/{channel_id}/webhooks/{webhook['id']}/delete")
        after = client.get(f"/api/v1/channels/{channel_id}/webhooks")

    assert webhook["name"] == "Build bot" and webhook["token"]
    assert [(entry["id"], entry["token"]) for entry in listed.json()] == [
        (webhook["id"], webhook["token"])
    ]
    assert deleted.status_code == 200 and deleted.json() is True, deleted.text
    assert after.json() == []


def test_a_plain_member_cannot_manage_webhooks_or_see_a_token(channel):
    channel_id, manager, member = channel
    webhook = _webhook(manager, channel_id)
    assert _post_as_webhook(manager, webhook, "deployed").status_code == 200

    with member.client() as client:
        created = client.post(
            f"/api/v1/channels/{channel_id}/webhooks/create", json={"name": "Impostor"}
        )
        listed = client.get(f"/api/v1/channels/{channel_id}/webhooks")
        deleted = client.delete(f"/api/v1/channels/{channel_id}/webhooks/{webhook['id']}/delete")
        seen = [client.get(f"/api/v1/channels/{channel_id}"), client.get("/api/v1/channels/")]

    assert (created.status_code, listed.status_code, deleted.status_code) == (403, 403, 403)
    shown = json.dumps([response.json() for response in seen] + _messages(member, channel_id))
    assert webhook["token"] not in shown, "a plain member was shown the webhook's token"
    with manager.client() as client:
        remaining = client.get(f"/api/v1/channels/{channel_id}/webhooks").json()
    assert [entry["id"] for entry in remaining] == [webhook["id"]]


def test_a_post_with_the_token_lands_in_the_channel_as_the_webhook(channel):
    channel_id, manager, member = channel
    webhook = _webhook(manager, channel_id)

    posted = _post_as_webhook(manager, webhook, "build 42 passed")

    assert posted.status_code == 200, posted.text
    [message] = _messages(member, channel_id)
    assert message["id"] == posted.json()["message_id"]
    assert message["content"] == "build 42 passed"
    assert message["user"] == {"id": webhook["id"], "name": "Build bot", "role": "webhook"}


def test_a_wrong_token_is_refused(channel):
    channel_id, manager, member = channel
    webhook = _webhook(manager, channel_id)

    refused = _post_as_webhook(manager, webhook, "forged", token="not-the-token")

    assert refused.status_code == 401, refused.text
    assert _messages(member, channel_id) == []


def test_a_deleted_webhook_stops_accepting_posts(channel):
    channel_id, manager, member = channel
    webhook = _webhook(manager, channel_id)
    with manager.client() as client:
        client.delete(f"/api/v1/channels/{channel_id}/webhooks/{webhook['id']}/delete")

    refused = _post_as_webhook(manager, webhook, "too late")

    assert refused.status_code == 401, refused.text
    assert _messages(member, channel_id) == []


def test_switched_off_channels_refuse_a_webhook_post(channel, admin):
    channel_id, manager, member = channel
    webhook = _webhook(manager, channel_id)
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        client.post(ADMIN_CONFIG, json={**current, "ENABLE_CHANNELS": False}).raise_for_status()

    refused = _post_as_webhook(manager, webhook, "while off")

    assert refused.status_code == 403, refused.text
    enable_channels(admin)
    assert _messages(member, channel_id) == []
