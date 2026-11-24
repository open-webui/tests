"""Regression: dragging a folder onto one of its own subfolders in the sidebar must not lose it.

open-webui 0.11.1, fix `23b3a69bc` (#28748): the server accepted a move of a folder under its own
descendant, so the dragged folder and everything below it vanished from the sidebar for good.
Since 20c413371 the sidebar ignores a drop on any folder below the dragged one, so no move is
sent and the tree stays. The server's refusal is pinned in
integration/security/test_folder_move_cycle.py.
A valid drop right after it does move, so the ignored drop is not the drag failing.

Discriminates: passes on dev 9d19883f1. Before 20c413371 the drop on the grandchild reached the
server, which this test now reads as a sent move.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from utils.sidebar_folders import drag, folder_row

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


def _parents(owner) -> dict:
    """Each folder's name mapped to its parent's name."""
    with owner.client() as client:
        folders = client.get("/api/v1/folders/").json()
    names = {folder["id"]: folder["name"] for folder in folders}
    return {folder["name"]: names.get(folder["parent_id"]) for folder in folders}


def _open_folders_section(page) -> None:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    toggle = page.get_by_role("button", name="Folders", exact=True)
    expect(toggle).to_be_visible()
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()


def test_dropping_a_folder_on_its_grandchild_is_refused_and_keeps_the_tree(
    page_for, owner_with_tree
):
    page = page_for(owner_with_tree)
    _open_folders_section(page)
    for name in FOLDER_NAMES:
        expect(folder_row(page, name)).to_be_visible()
    # every expanded folder has loaded its chats, so the tree has stopped moving
    expect(page.get_by_text("No chats")).to_have_count(len(FOLDER_NAMES))

    moves = []
    page.on(
        "request",
        lambda request: moves.append(request) if request.url.endswith("/update/parent") else None,
    )
    drag(page, folder_row(page, "projects"), folder_row(page, "q1"))
    with page.expect_response(lambda response: response.url.endswith("/update/parent")):
        drag(page, folder_row(page, "q1"), folder_row(page, "projects"))
    assert len(moves) == 1, "dropping a folder on its grandchild sent a move"
    parents = _parents(owner_with_tree)
    assert parents["projects"] is None and parents["2026"] == parents["q1"], parents
    page.reload()  # the sidebar and its folders section stay open
    for name in FOLDER_NAMES:
        expect(folder_row(page, name)).to_be_visible()
