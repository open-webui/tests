"""The page panel of a note, located the way a person sees it.

The editor's header holds a Toggle pages button (named for the open page) that opens the Pages
panel. The panel is a tree whose items are the page titles, each nested at its level; every row
has a Page menu button and an Add subpage button, and the panel ends with Add page. A new page
opens with its title field focused, so a person types the name and presses Enter.
`open_pages(page)` opens the panel and returns the tree, `page_tab` is one page's item,
`page_menu` opens one page's menu, `add_page` creates and names a page and `expect_tree` waits
for the panel to list the given pages at their levels. `create_page` sets a page up over the API
for a test that starts with it in place.
"""

from __future__ import annotations

import re
import uuid

from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor

PAGE_CHANGE = re.compile(r"/api/v1/notes/[^/]+/pages/update$")


def open_pages(page: Page) -> Locator:
    toggle = page.get_by_role("main").get_by_role("button", name="Toggle pages")
    tree = page.get_by_role("tree", name="Pages")
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    expect(tree).to_be_visible()
    return tree


def page_tab(page: Page, title: str) -> Locator:
    return page.get_by_role("tree", name="Pages").get_by_role("treeitem", name=title, exact=True)


def page_menu(page: Page, title: str) -> Locator:
    tab = page_tab(page, title)
    tab.hover()
    tab.locator("xpath=..").get_by_role("button", name="Page menu").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Copy page link")).to_be_visible()
    return menu


def name_new_page(page: Page, title: str) -> None:
    """Type the title of the page just created and wait until the server stored it."""
    field = page.get_by_role("main").get_by_label("Page title")
    expect(field).to_be_focused()
    field.fill(title)
    with page.expect_response(lambda response: bool(PAGE_CHANGE.search(response.url))) as renamed:
        field.press("Enter")
    assert renamed.value.ok, f"renaming the page to {title!r}: HTTP {renamed.value.status}"
    expect(page_tab(page, title)).to_be_visible()


def add_page(page: Page, title: str, under: str | None = None) -> None:
    """Add a top level page, or a subpage of the page titled `under`, and name it `title`."""
    tree = open_pages(page)
    if under is None:
        add = page.get_by_role("button", name="Add page", exact=True)
    else:
        page_tab(page, under).hover()
        add = tree.get_by_role("button", name=f"Add subpage to {under}", exact=True)
    with page.expect_response(lambda response: bool(PAGE_CHANGE.search(response.url))) as created:
        add.click()
    assert created.value.ok, f"adding a page: HTTP {created.value.status}"
    name_new_page(page, title)


def expect_tree(page: Page, rows: list[tuple[str, int]]) -> None:
    """The panel lists exactly these pages, top to bottom, each as (title, nesting level)."""
    tabs = page.get_by_role("tree", name="Pages").get_by_role("treeitem")
    expect(tabs).to_have_text([title for title, _ in rows])
    for index, (_, level) in enumerate(rows):
        expect(tabs.nth(index)).to_have_attribute("aria-level", str(level))


def create_page(owner: Actor, note_id: str, title: str, text: str) -> str:
    """A top level page with `text`, added over the API the way the panel adds one."""
    page_id = str(uuid.uuid4())
    with owner.client() as client:
        revision = client.get(f"/api/v1/notes/{note_id}").json()["data"]["pages_revision"]
        created = client.post(
            f"/api/v1/notes/{note_id}/pages/update",
            json={"action": "create", "new_id": page_id, "title": title, "revision": revision},
        )
        assert created.status_code == 200, created.text
        written = client.post(
            f"/api/v1/notes/{note_id}/update",
            json={"data": {"content": {"md": text}}, "page_id": page_id},
        )
        assert written.status_code == 200, written.text
    return page_id
