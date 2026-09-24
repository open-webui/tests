"""Regression: the folder page deleted a shared subfolder for a collaborator, and refused a
keep-chats delete to an account without the chat delete permission.

open-webui 0.11.0, fix `915ef7d07` (#27003): the folder page offers Delete to anyone with write
access, and the server let write access on a shared folder delete any folder below it, taking the
owner's chats along. The fix allows only the owner or an admin, so the page shows a refusal.

open-webui 0.11.4, fix `b64cb0604` (#30163): unticking "Delete all contents inside this folder"
still ran the chat delete permission check, so an account without `chat.delete` was refused a
delete that removes no chats.

Twin of unit/security/test_folder_delete_ownership.py.

Discriminates: passes on bbfa876af; fails with 915ef7d07 reverted (the collaborator is told the
folder was deleted) and with b64cb0604 reverted (the keep-chats delete is refused).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

DELETED = "Folder deleted successfully"
REFUSED = "You do not have permission"


def _create_folder(actor, parent_id: str | None = None) -> str:
    with actor.client() as client:
        created = client.post(
            "/api/v1/folders/",
            json={"name": f"folder-{uuid.uuid4().hex[:8]}", "parent_id": parent_id},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _create_chat(actor, folder_id: str) -> str:
    with actor.client() as client:
        created = client.post(
            "/api/v1/chats/new", json={"chat": {"title": "kept safe"}, "folder_id": folder_id}
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_chat(owner, chat_id: str) -> dict | None:
    with owner.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    return found.json() if found.status_code == 200 else None


def _delete_on_the_folder_page(page: Page, folder_id: str, delete_contents: bool) -> str:
    """Delete through the folder page's options menu; returns the toast it shows."""
    page.goto(f"/folders/{folder_id}")
    page.get_by_label("Folder options").click()
    page.get_by_role("button", name="Delete", exact=True).click()
    dialog = page.get_by_role("dialog", name="Delete folder?")
    dialog.get_by_role("checkbox").set_checked(delete_contents)
    dialog.get_by_role("button", name="Confirm").click()
    outcome = page.get_by_text(re.compile(f"{DELETED}|{REFUSED}"))
    expect(outcome).to_be_visible()
    return outcome.inner_text()


def test_a_write_collaborator_is_refused_deleting_a_shared_subfolder(page_for, admin, make_user):
    root = _create_folder(admin)
    child = _create_folder(admin, parent_id=root)
    chat_id = _create_chat(admin, child)
    collaborator = make_user()
    grants = [
        {"principal_type": "user", "principal_id": collaborator.id, "permission": permission}
        for permission in ("read", "write")
    ]
    with admin.client() as client:
        shared = client.post(
            f"/api/v1/folders/{root}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text

    try:
        outcome = _delete_on_the_folder_page(page_for(collaborator), child, delete_contents=True)
        assert REFUSED in outcome, "a write collaborator deleted a folder they do not own (#27003)"
        assert _stored_chat(admin, chat_id) is not None, "the owner's chat was deleted (#27003)"
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/folders/{root}")


def test_a_folder_can_be_deleted_keeping_its_chats_without_chat_delete(
    page_for, admin, preserve, make_user
):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["delete"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    owner = make_user()
    folder_id = _create_folder(owner)
    chat_id = _create_chat(owner, folder_id)

    outcome = _delete_on_the_folder_page(page_for(owner), folder_id, delete_contents=False)

    assert DELETED in outcome, (
        f"a keep-chats folder delete was refused for lack of chat.delete (#30163): {outcome}"
    )
    assert _stored_chat(owner, chat_id)["folder_id"] is None
