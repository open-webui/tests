"""Regression: dragging a folder onto one of its own subfolders in the sidebar must not lose it.

open-webui 0.11.1, fix `23b3a69bc` (#28748): the server accepted a move of a folder under its own
descendant, so the dragged folder and everything below it vanished from the sidebar for good.
The sidebar only ignores a drop on the dragged folder's direct children, so a drop on a
grandchild reaches the server, which now refuses it with an error the page shows as a toast.

Twin of unit/security/test_folder_move_cycle.py.

Discriminates: passes on bbfa876af; with the move route's subtree check removed the drop is
accepted and no error appears.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER_NAMES = ("projects", "2026", "q1")


@pytest.fixture
def owner_with_tree(make_user):
    """projects > 2026 > q1, all expanded so every row is on screen."""
    owner = make_user()
    parent_id = None
    with owner.client() as client:
        for name in FOLDER_NAMES:
            created = client.post("/api/v1/folders/", json={"name": name, "parent_id": parent_id})
            assert created.status_code == 200, created.text
            parent_id = created.json()["id"]
            expanded = client.post(
                f"/api/v1/folders/{parent_id}/update/expanded", json={"is_expanded": True}
            )
            assert expanded.status_code == 200, expanded.text
    return owner


def _open_folders_section(page) -> None:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    toggle = page.get_by_role("button", name="Folders", exact=True)
    expect(toggle).to_be_visible()
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()


def _folder_row(page, name: str):
    return page.get_by_role("button", name=name)


def _drag(page, source, target) -> None:
    # A synthetic drag: a mouse drag across the nested rows sometimes never fires the drop.
    transfer = page.evaluate_handle("() => new DataTransfer()")
    source.dispatch_event("dragstart", {"dataTransfer": transfer})
    target.dispatch_event("dragover", {"dataTransfer": transfer})
    target.dispatch_event("drop", {"dataTransfer": transfer})
    source.dispatch_event("dragend", {"dataTransfer": transfer})


def test_dropping_a_folder_on_its_grandchild_is_refused_and_keeps_the_tree(
    page_for, owner_with_tree
):
    page = page_for(owner_with_tree)
    _open_folders_section(page)
    for name in FOLDER_NAMES:
        expect(_folder_row(page, name)).to_be_visible()
    # every expanded folder has loaded its chats, so the tree has stopped moving
    expect(page.get_by_text("No chats")).to_have_count(len(FOLDER_NAMES))

    with page.expect_response(lambda response: response.url.endswith("/update/parent")):
        _drag(page, _folder_row(page, "projects"), _folder_row(page, "q1"))

    expect(
        page.get_by_text("Cannot move a folder into itself or one of its subfolders")
    ).to_be_visible()
    page.reload()  # the sidebar and its folders section stay open
    for name in FOLDER_NAMES:
        expect(_folder_row(page, name)).to_be_visible()
