"""Journey: the sidebar chat list pages in more chats when it is scrolled to its end.

The list asks for sixty chats at a time, newest first. An account with more chats than that sees
the first sixty, and the older ones appear once the list is scrolled to its end; opening one of
them shows its conversation.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose chat list answers the first page
whatever page is asked, and in a frontend build that loads nothing more at the end of the list, it
fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PAGE_SIZE = 60
CHAT_COUNT = PAGE_SIZE + 5
ANSWER = "Kept from long ago."
HISTORY = {
    "currentId": "a1",
    "messages": {
        "q1": {"id": "q1", "parentId": None, "childrenIds": ["a1"], "role": "user", "content": "?"},
        "a1": {
            "id": "a1",
            "parentId": "q1",
            "childrenIds": [],
            "role": "assistant",
            "content": ANSWER,
            "done": True,
        },
    },
}


def test_the_oldest_chat_appears_when_the_list_is_scrolled_to_its_end(page_for, make_user):
    owner = make_user()
    with owner.client() as client:
        for index in range(CHAT_COUNT):
            created = client.post(
                "/api/v1/chats/new",
                json={"chat": {"title": f"Paging {index:03d}", "history": HISTORY}},
            )
            assert created.status_code == 200, created.text
        first_page = client.get("/api/v1/chats/", params={"page": 1}).json()
        second_page = client.get("/api/v1/chats/", params={"page": 2}).json()
    assert len(first_page) == PAGE_SIZE, "the list no longer pages in sixties: retarget this test"
    assert second_page, "the second page of the chat list is empty"
    oldest = second_page[-1]

    page = page_for(owner)
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("button", name=first_page[0]["title"])).to_be_visible()
    expect(sidebar.get_by_role("button", name=oldest["title"])).to_have_count(0)

    loading = sidebar.get_by_text("Loading...")
    expect(loading).to_be_visible()
    loading.scroll_into_view_if_needed()
    row = sidebar.get_by_role("button", name=oldest["title"])
    expect(row).to_be_visible()
    row.click()

    expect(page).to_have_url(f"{owner.base_url}/c/{oldest['id']}")
    # the chat that was open stays mounted for a moment while the picked one loads
    conversation = page.get_by_label("Chat Conversation")
    expect(conversation).to_have_count(1)
    expect(conversation).to_contain_text(ANSWER)
