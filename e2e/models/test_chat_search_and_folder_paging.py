"""A sidebar folder holding more than sixty chats never showed the rest.

Fix `409fb39` (#26786, open-webui 0.11.0). The sidebar lists a folder through
`GET /api/v1/folders/{id}/shared/chats`, which answered one fixed list of sixty with no count, so
"Show more" never appeared and the chats past the sixtieth were unreachable. The route now pages
in tens and answers `has_more`, which drives "Show more".

Twin of unit/models/test_chat_search_and_folder_paging.py.

Discriminates: passes on upstream dev `bbfa876af`; with the folder route back on one unpaged
call "Show more" never appears and five of the 65 chats stay out of reach.

A search in Shared Chats (Settings > Data Controls) and in an account's chats (Admin Settings >
Users) showed only its first sixty results: the dialog decided there was nothing more to load
before the search had answered (open-webui/open-webui#32675, fix 0f2d1af1c). Both search tests
fail in the pre-fix build of the 0.12.4 changes, which has that commit reverted.

Opening a chat in another tab sent a folder's chat list back to page 1
(open-webui/open-webui#34142): every tab took the chat being marked as read as a reason to reload
the list from its first page. Discriminates: red on upstream dev `262a8c6f7` (the first tab ends
on page 1), green with the fix on branch `fix/folder-page-kept-when-chat-opens-elsewhere`. Since
06491f013 the folder's page loads its next 50 chats as the list's end scrolls into view, so the
test reads a chat past the first 50 from the top of the list and checks the list still holds every
chat it had loaded: green on dev 4acf39ec2, red in a frontend copy whose refresh reloads only the
first page.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.setting_rows import setting_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER_SIZE = 65


def test_show_more_reaches_every_chat_in_a_large_folder(page_for, make_user):
    owner = make_user()
    titles = [f"Paging {index:03d}" for index in range(FOLDER_SIZE)]
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"Big {uuid.uuid4().hex[:6]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        for title in titles:
            created = client.post(
                "/api/v1/chats/new", json={"chat": {"title": title}, "folder_id": folder_id}
            )
            assert created.status_code == 200, created.text
        expanded = client.post(
            f"/api/v1/folders/{folder_id}/update/expanded", json={"is_expanded": True}
        )
        assert expanded.status_code == 200, expanded.text

    page = page_for(owner)
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_role("button", name="Folders", exact=True).click()
    show_more = sidebar.get_by_role("button", name="Show more")
    expect(show_more).to_be_visible()
    while show_more.count():
        show_more.click()
        expect(sidebar.get_by_label("Loading")).to_have_count(0)

    for title in titles:
        expect(sidebar.get_by_text(title, exact=True)).to_be_visible()


SEARCHED_SIZE = 65


def _chats_titled(owner, tag: str, shared: bool = False) -> None:
    """`SEARCHED_SIZE` chats whose titles carry `tag`, each shared by link when `shared`."""
    with owner.client() as client:
        for index in range(SEARCHED_SIZE):
            title = f"Heron {tag} {index:03d}"
            created = client.post("/api/v1/chats/new", json={"chat": {"title": title}})
            assert created.status_code == 200, created.text
            if shared:
                share = client.post(f"/api/v1/chats/{created.json()['id']}/share")
                assert share.status_code == 200, share.text


def _search_to_the_end(dialog: Locator, tag: str) -> Locator:
    """Search the chats dialog for `tag` and scroll to its end; returns the matching rows."""
    page = dialog.page
    matches = dialog.get_by_role("link", name=re.compile(tag))
    expect(matches.first).to_be_visible()
    with page.expect_response(lambda response: f"query={tag}" in response.url) as searched:
        dialog.get_by_placeholder("Search Chats").fill(tag)
    assert searched.value.ok, searched.value.status
    expect(matches.first).to_be_visible()

    # at the end of the list a further page is asked for while its loading row is in view
    matches.last.hover()
    page.mouse.wheel(0, 20000)
    return matches


def _open_shared_chats(page: Page) -> Locator:
    page.goto("/?settings=data_controls")
    settings = page.get_by_role("dialog")
    setting_button(settings, "Shared Chats").click()
    # found by its search box: the Unshare All button leaves while a search is typed
    shared = page.get_by_role("dialog").filter(has=page.get_by_placeholder("Search Chats"))
    expect(shared.get_by_role("button", name="Unshare All Shared Chats")).to_be_visible()
    return shared


def _open_user_chats(page: Page, owner) -> Locator:
    page.goto("/admin/users")
    users = page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(owner.email)
    users.get_by_role("row").filter(has_text=owner.email).get_by_role(
        "button", name="Chats"
    ).click()
    chats = page.get_by_role("dialog").filter(has_text=f"{owner.name}'s Chats")
    expect(chats).to_be_visible()
    return chats


def test_a_search_in_shared_chats_reaches_every_match(page_for, make_user):
    # open-webui/open-webui#32675, fix 0f2d1af1c
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    _chats_titled(owner, tag, shared=True)

    matches = _search_to_the_end(_open_shared_chats(page_for(owner)), tag)

    expect(matches, "a search in Shared Chats stopped at its first page (#32675)").to_have_count(
        SEARCHED_SIZE
    )


def test_a_search_in_a_users_chats_reaches_every_match(page_for, make_user):
    # open-webui/open-webui#32675, fix 0f2d1af1c
    owner = make_user()
    tag = uuid.uuid4().hex[:8]
    _chats_titled(owner, tag)
    page = page_for(make_user(role="admin"))

    matches = _search_to_the_end(_open_user_chats(page, owner), tag)

    expect(
        matches, "a search in an account's chats in Admin Settings stopped at its first page"
    ).to_have_count(SEARCHED_SIZE)


# the folder's page loads its chats in pages of this many
FOLDER_PAGE_SIZE = 50
PAGED_TITLE = re.compile(r"^Paged \d{3}$")


def test_opening_a_chat_in_another_tab_keeps_the_folder_page(page_for, make_user):
    # open-webui/open-webui#34142
    owner = make_user()
    chat_ids: dict[str, str] = {}
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"Pages {uuid.uuid4().hex[:6]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        for index in range(FOLDER_PAGE_SIZE + 5):
            title = f"Paged {index:03d}"
            created = client.post(
                "/api/v1/chats/new", json={"chat": {"title": title}, "folder_id": folder_id}
            )
            assert created.status_code == 200, created.text
            chat_ids[title] = created.json()["id"]
            # unread, so the first tab shows the moment the second one reads it
            unread = client.post(f"/api/v1/chats/{chat_ids[title]}/unread")
            assert unread.status_code == 200, unread.text

    page = page_for(owner, path=f"/folders/{folder_id}")
    rows = page.get_by_role("listitem").filter(has=page.get_by_text(PAGED_TITLE))
    expect(rows).to_have_count(FOLDER_PAGE_SIZE)
    # the list loads its next page as its end scrolls into view
    rows.last.scroll_into_view_if_needed()
    expect(rows).to_have_count(len(chat_ids))
    title = rows.last.get_by_text(PAGED_TITLE).inner_text().strip()
    row = rows.filter(has=page.get_by_text(title, exact=True))
    unread_dot = row.get_by_role("status", name="Unread")
    expect(unread_dot).to_have_count(1)
    # back at the top, so a list cut to its first page does not load the rest again by itself
    rows.first.scroll_into_view_if_needed()

    page_for(owner, path=f"/c/{chat_ids[title]}")

    expect(unread_dot, "the first tab never saw the chat being read").to_have_count(0)
    expect(
        rows, "opening a chat in another tab sent the folder back to its first page (#34142)"
    ).to_have_count(len(chat_ids))
    expect(row).to_be_visible()
