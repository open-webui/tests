"""Channels for the deleted-quote tests: switch the feature on, share a channel, post into it.

`enable_channels` changes a global setting, so call it after `preserve("admin_config")`.
"""

from __future__ import annotations

import uuid

from harness.actors import Actor

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def enable_channels(admin: Actor) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), "ENABLE_CHANNELS": True})
    saved.raise_for_status()


def group_channel(owner: Actor, *members: Actor) -> str:
    """A group channel the owner creates with these members; returns its id."""
    with owner.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={
                "name": f"channel-{uuid.uuid4().hex[:8]}",
                "type": "group",
                "user_ids": [member.id for member in members],
            },
        )
    created.raise_for_status()
    return created.json()["id"]


def post_message(author: Actor, channel_id: str, content: str, **links: str) -> str:
    """Post as `author`; `links` takes `parent_id` (a thread) and `reply_to_id` (a quote)."""
    with author.client() as client:
        posted = client.post(
            f"/api/v1/channels/{channel_id}/messages/post", json={"content": content, **links}
        )
    posted.raise_for_status()
    return posted.json()["id"]


def delete_message(author: Actor, channel_id: str, message_id: str) -> None:
    with author.client() as client:
        deleted = client.delete(f"/api/v1/channels/{channel_id}/messages/{message_id}/delete")
    deleted.raise_for_status()
