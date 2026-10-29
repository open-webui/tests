"""Regression: the arrow keys skip menu entries that show up after the menu has opened.

1de86d8d1 (open-webui/open-webui#33359) moves the focus through a menu's entries with the
arrow keys, Home and End, but it marked as menu items only the buttons present when the menu
opened. Entries rendered a moment later stayed plain buttons, which the arrow keys passed over: the
earlier uploads under the chat input's More, Attach Files, and the tag chips in the chat actions
menu. Both stayed reachable with Tab, and a click worked.

Was red on dev 311be1757: entries rendered after the menu opens got no menuitem role, so arrow keys
skipped them (#33359) (open-webui/open-webui#33374, fixed by 05e115366, #33375).

Discriminates: passes on the build of fix PR #33375 (636d8ec2a), fails on dev 311be1757 (ArrowDown
from the first entry never reaches the upload or the chip).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input, open_input_menu
from utils.menus import late_menu_entry

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

HISTORY = {
    "currentId": "a1",
    "messages": {
        "q1": {
            "id": "q1",
            "parentId": None,
            "childrenIds": ["a1"],
            "role": "user",
            "content": "Hi",
        },
        "a1": {
            "id": "a1",
            "parentId": "q1",
            "childrenIds": [],
            "role": "assistant",
            "content": "Hello",
            "done": True,
        },
    },
}

# more presses than either menu has entries, so a skipped entry shows as never reached
ARROW_PRESSES = 12


def arrow_down_reaches(page: Page, target: Locator) -> bool:
    for _ in range(ARROW_PRESSES):
        page.keyboard.press("ArrowDown")
        if target.evaluate("(element) => element === document.activeElement"):
            return True
    return False


def upload(account: Actor, name: str) -> None:
    with account.client() as client:
        uploaded = client.post(
            "/api/v1/files/", files={"file": (name, b"high at noon", "text/plain")}
        )
    assert uploaded.status_code == 200, uploaded.text


def tagged_chat(account: Actor, tag: str) -> str:
    with account.client() as client:
        created = client.post(
            "/api/v1/chats/new", json={"chat": {"title": "Coast", "history": HISTORY}}
        )
        assert created.status_code == 200, created.text
        chat_id = created.json()["id"]
        tagged = client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": tag})
    assert tagged.status_code == 200, tagged.text
    return chat_id


def test_arrow_down_reaches_an_earlier_upload_in_attach_files(page_for, make_user):
    account = make_user()
    name = f"tide-table-{uuid.uuid4().hex[:6]}.txt"
    upload(account, name)
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    menu = open_input_menu(page)
    menu.get_by_role("menuitem", name="Attach Files").click()
    entry = late_menu_entry(menu, name, exact=True)
    expect(entry).to_be_visible()

    first = page.get_by_role("menu").filter(has_text=name).last.get_by_role("menuitem").first
    first.focus()
    expect(first).to_be_focused()

    assert arrow_down_reaches(page, entry), (
        f"ArrowDown from the first entry never reached {name} in Attach Files"
    )


def test_arrow_down_reaches_a_tag_chip_in_the_chat_actions_menu(page_for, make_user):
    account = make_user()
    tag = f"harbour{uuid.uuid4().hex[:8]}"
    chat_id = tagged_chat(account, tag)
    page = page_for(account, path=f"/c/{chat_id}")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Chat actions").click()
    menu = page.get_by_role("menu")
    chip = late_menu_entry(menu, tag, exact=True)
    expect(chip).to_be_visible()

    first = menu.get_by_role("menuitem").first
    first.focus()
    expect(first).to_be_focused()

    assert arrow_down_reaches(page, chip), (
        f"ArrowDown from the first entry never reached the {tag} chip in the chat actions menu"
    )
