"""Regression: an admin deleting someone else's chat tidies the owner's tags, not the admin's.

open-webui 0.11.4 fix `2690d04ca` (#30171): `DELETE /api/v1/chats/{id}` ran the orphan-tag
cleanup with the deleting admin's id instead of the chat owner's. Deleting a user's last chat
carrying a tag left that tag in the owner's tag list with nothing pointing at it, and removed
the admin's own tag of the same name although an admin chat still carried it.

Twin of unit/security/test_chat_tag_cleanup_owner.py.

Discriminates: passes on dev bbfa876af; with `chat.user_id` turned back into `user.id` in the
cleanup call the owner keeps the orphaned tag and the admin loses theirs.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _tagged_chat(client, tag: str) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": {"title": f"about {tag}"}})
    created.raise_for_status()
    chat_id = created.json()["id"]
    client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": tag}).raise_for_status()
    return chat_id


def _tag_ids(client) -> set[str]:
    listed = client.get("/api/v1/chats/all/tags")
    listed.raise_for_status()
    return {tag["id"] for tag in listed.json()}


@pytest.fixture
def tag() -> str:
    return f"project-{uuid.uuid4().hex[:8]}"


def test_admin_deleting_a_users_last_tagged_chat_removes_the_owners_tag(admin, make_user, tag):
    owner = make_user()
    with owner.client() as owner_client, admin.client() as admin_client:
        chat_id = _tagged_chat(owner_client, tag)
        assert tag in _tag_ids(owner_client)

        deleted = admin_client.delete(f"/api/v1/chats/{chat_id}")
        assert deleted.status_code == 200 and deleted.json() is True, deleted.text

        assert tag not in _tag_ids(owner_client), (
            "the admin deleted the owner's last chat carrying this tag and the owner kept the "
            "tag with nothing pointing at it; the cleanup ran for the admin's account (#30171)"
        )


def test_admin_deleting_a_users_chat_keeps_the_admins_own_tag(admin, make_user, tag):
    owner = make_user()
    with owner.client() as owner_client, admin.client() as admin_client:
        victim_chat = _tagged_chat(owner_client, tag)
        admin_chat = _tagged_chat(admin_client, tag)
        try:
            admin_client.delete(f"/api/v1/chats/{victim_chat}").raise_for_status()
            assert tag in _tag_ids(admin_client), (
                "deleting another user's chat removed the admin's own tag of the same name "
                "while an admin chat still carries it (#30171)"
            )
        finally:
            admin_client.delete(f"/api/v1/chats/{admin_chat}")


def test_a_tag_still_in_use_survives_an_admin_delete(admin, make_user, tag):
    owner = make_user()
    with owner.client() as owner_client, admin.client() as admin_client:
        deleted_chat = _tagged_chat(owner_client, tag)
        _tagged_chat(owner_client, tag)

        admin_client.delete(f"/api/v1/chats/{deleted_chat}").raise_for_status()

        assert tag in _tag_ids(owner_client)


def test_the_owners_own_delete_removes_their_tag(make_user, tag):
    with make_user().client() as owner_client:
        chat_id = _tagged_chat(owner_client, tag)

        owner_client.delete(f"/api/v1/chats/{chat_id}").raise_for_status()

        assert tag not in _tag_ids(owner_client)
