"""A folder's page and the chats it lists, located the way a person sees them.

`folder_page(page)` is the page's Folder region, which holds the list and the board.
`folder_chat(scope, title)` is a listed chat's title: a button that opens the chat's hover
preview, still an anchor to the chat (d773f8fad). A Page scope looks inside the Folder region, so
the same chat in the sidebar never matches; a Locator scope (a board column) is searched as given.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page


def folder_page(page: Page) -> Locator:
    return page.get_by_role("region", name="Folder", exact=True)


def folder_chat(scope: Locator | Page, title: str) -> Locator:
    if isinstance(scope, Page):
        scope = folder_page(scope)
    return scope.get_by_role("button", name=title, exact=True)
