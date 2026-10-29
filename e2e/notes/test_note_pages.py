"""Journey: a note's pages, added, named, ordered, nested and deleted in its Pages panel.

Since v0.12.5 (b34885efc) a note holds pages and subpages, listed in the Pages panel that the
Toggle pages button in the editor's header opens. Add page and a row's Add subpage button create
a page whose title field opens at once; the row's Page menu moves it up or down among its
siblings, moves it under another page and deletes it after a confirmation. Each change is
stored, so the panel shows the same pages at the same levels after a reload. A link with
`?page=` opens the note on that page.

Discriminates: journey, no fix. In a frontend build whose page link ignores `?page=` the link
test fails, and in a backend copy of dev 05e115366 whose rename, move and delete actions change
nothing the other three fail (the panel keeps the old titles, levels or pages).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from utils.note_pages import (
    PAGE_CHANGE,
    add_page,
    create_page,
    expect_tree,
    open_pages,
    page_menu,
    page_tab,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST_PAGE_TEXT = "Pack the rain jackets."


def create_note(owner: Actor, title: str) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": title, "data": {"content": {"md": FIRST_PAGE_TEXT}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def stored_tree(owner: Actor, note_id: str) -> list[tuple[str, str | None]]:
    with owner.client() as client:
        pages = client.get(f"/api/v1/notes/{note_id}").json()["data"]["pages"]
    titles = {page["id"]: page["title"] for page in pages}
    return [(page["title"], titles.get(page.get("parent_id"))) for page in pages]


def open_note(page: Page, note_id: str, query: str = "") -> None:
    page.goto(f"/notes/{note_id}{query}")
    expect(page.get_by_role("main").get_by_label("Write something...")).to_be_visible()


def change_page(page: Page, menu_item: str, confirm: str | None = None) -> None:
    """Pick `menu_item` from the open Page menu, confirm when asked, and wait for the save."""
    menu = page.get_by_role("menu")
    with page.expect_response(lambda response: bool(PAGE_CHANGE.search(response.url))) as saved:
        menu.get_by_role("menuitem", name=menu_item).click()
        if confirm:
            page.get_by_role("dialog").get_by_role("button", name=confirm).click()
    assert saved.value.ok, f"{menu_item}: HTTP {saved.value.status}"


def test_pages_added_named_ordered_and_nested_in_the_panel_survive_a_reload(page_for, make_user):
    owner = make_user()
    note_id = create_note(owner, "Trip")
    page = page_for(owner)
    open_note(page, note_id)

    add_page(page, "Route")
    add_page(page, "Budget")
    page_menu(page, "Budget")
    change_page(page, "Move up")
    add_page(page, "Ferry", under="Route")

    expected = [("Trip", 1), ("Budget", 1), ("Route", 1), ("Ferry", 2)]
    expect_tree(page, expected)
    page.reload()
    open_pages(page)
    expect_tree(page, expected)
    assert stored_tree(owner, note_id) == [
        ("Trip", None),
        ("Budget", None),
        ("Route", None),
        ("Ferry", "Route"),
    ]


def test_a_page_moved_under_another_takes_its_place_there_after_a_reload(page_for, make_user):
    owner = make_user()
    note_id = create_note(owner, "Trip")
    create_page(owner, note_id, "Route", "Take the coast road.")
    create_page(owner, note_id, "Budget", "Two hundred euros.")
    page = page_for(owner)
    open_note(page, note_id)
    open_pages(page)

    page_menu(page, "Budget").get_by_role("menuitem", name="Move page…").click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_label("Parent page").select_option(label="Route")
    with page.expect_response(lambda response: bool(PAGE_CHANGE.search(response.url))) as saved:
        dialog.get_by_role("button", name="Move", exact=True).click()
    assert saved.value.ok, f"moving the page: HTTP {saved.value.status}"

    expected = [("Trip", 1), ("Route", 1), ("Budget", 2)]
    expect_tree(page, expected)
    page.reload()
    open_pages(page)
    expect_tree(page, expected)


def test_a_page_link_opens_the_note_on_that_page(page_for, make_user):
    owner = make_user()
    note_id = create_note(owner, "Trip")
    budget_id = create_page(owner, note_id, "Budget", "Two hundred euros for the ferry.")
    page = page_for(owner, path=f"/notes/{note_id}?page={budget_id}")

    editor = page.get_by_role("main").get_by_label("Write something...")
    expect(editor).to_contain_text("Two hundred euros for the ferry.")
    expect(editor).not_to_contain_text(FIRST_PAGE_TEXT)
    open_pages(page)
    expect(page_tab(page, "Budget")).to_have_attribute("aria-selected", "true")


def test_a_deleted_page_is_gone_after_a_reload(page_for, make_user):
    owner = make_user()
    note_id = create_note(owner, "Trip")
    create_page(owner, note_id, "Route", "Take the coast road.")
    create_page(owner, note_id, "Budget", "Two hundred euros.")
    page = page_for(owner)
    open_note(page, note_id)
    open_pages(page)

    page_menu(page, "Route")
    change_page(page, "Delete page", confirm="Confirm")

    expect_tree(page, [("Trip", 1), ("Budget", 1)])
    page.reload()
    open_pages(page)
    expect_tree(page, [("Trip", 1), ("Budget", 1)])
    assert stored_tree(owner, note_id) == [("Trip", None), ("Budget", None)]
