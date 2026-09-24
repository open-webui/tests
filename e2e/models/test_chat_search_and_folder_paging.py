"""A sidebar folder holding more than sixty chats never showed the rest.

Fix `409fb39` (#26786, open-webui 0.11.0). The sidebar lists a folder through
`GET /api/v1/folders/{id}/shared/chats`, which answered one fixed list of sixty with no count, so
"Show more" never appeared and the chats past the sixtieth were unreachable. The route now pages
in tens and answers `has_more`, which drives "Show more".

Twin of unit/models/test_chat_search_and_folder_paging.py.

Discriminates: passes on upstream dev `bbfa876af`; with the folder route back on one unpaged
call "Show more" never appears and five of the 65 chats stay out of reach.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import expect

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
