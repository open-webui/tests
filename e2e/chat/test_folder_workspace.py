"""Journey: a folder page's statuses, board, tags, filter and search, as a person uses them.

Since 06491f013 a folder's page lists its chats with a status control and tags on every row. A
person adds a status in the Statuses dialog, reached from a chat's status menu, and picks it for
the chat from the same menu; the pick is stored and shown after a reload. Under Display options
the page switches to a board with a column per status, kept after a reload, and a card dragged to
another column takes that column's status. A tag typed into a chat's tag control is stored on the
chat, and the Filter menu's Tags entry then narrows the list to the chats carrying it; the folder
search narrows the list to chats whose title or messages match. A member who may only read the
shared folder sees the owner's chats with neither a status menu nor a tag control.

Discriminates: journey, no fix. All pass on the dev 941bf3674 build. With the dev build on a
backend copy of it whose status route stores nothing the status pick and board drag tests go red,
whose tag filter is ignored the tag test goes red, whose folder search matches nothing the search
test goes red and whose chats all read as editable the reading member test goes red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.menus import late_menu_entry

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TODO = {"id": "s_todo", "name": "To do", "glyph": "todo", "color": "gray"}
DONE = {"id": "s_done", "name": "Done", "glyph": "done", "color": "green"}


def _folder(owner: Actor, statuses: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"Yard {uuid.uuid4().hex[:6]}"})
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        if statuses:
            updated = client.post(
                f"/api/v1/folders/{folder_id}/update", json={"meta": {"statuses": statuses}}
            )
            assert updated.status_code == 200, updated.text
    return folder_id


def _chat(owner: Actor, folder_id: str, title: str, text: str = "") -> str:
    messages = {}
    if text:
        messages = {
            "q1": {"id": "q1", "parentId": None, "childrenIds": [], "role": "user", "content": text}
        }
    chat = {"title": title, "history": {"currentId": "q1" if text else None, "messages": messages}}
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat, "folder_id": folder_id})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _file(owner: Actor, folder_id: str, chat_id: str, status: str) -> None:
    with owner.client() as client:
        filed = client.patch(
            f"/api/v1/folders/{folder_id}/chats/{chat_id}/status", json={"status": status}
        )
    assert filed.status_code == 200, filed.text


def _stored_status(owner: Actor, folder_id: str, chat_id: str) -> str | None:
    with owner.client() as client:
        listed = client.get(
            f"/api/v1/folders/{folder_id}/shared/chats", params={"workspace": "true", "page": 1}
        )
    assert listed.status_code == 200, listed.text
    [chat] = [chat for chat in listed.json()["chats"] if chat["id"] == chat_id]
    return chat["status"]


def _chat_link(scope: Locator | Page, title: str) -> Locator:
    return scope.get_by_role("link", name=re.compile(re.escape(title)))


def _workspace(page: Page) -> Locator:
    return page.get_by_role("region", name="Folder", exact=True)


def _status_button(page: Page, title: str) -> Locator:
    return page.get_by_role("button", name=re.compile(f"^Status for {re.escape(title)}: "))


def _status_saved(page: Page):
    return page.expect_response(
        lambda response: response.request.method == "PATCH" and response.url.endswith("/status")
    )


def test_a_status_added_in_the_dialog_is_picked_for_a_chat_and_kept(page_for, make_user):
    owner = make_user()
    folder_id = _folder(owner)
    chat_id = _chat(owner, folder_id, "Hull survey")
    page = page_for(owner, path=f"/folders/{folder_id}")

    _status_button(page, "Hull survey").click()
    late_menu_entry(page, "Edit statuses…").click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("button", name="Add status").click()
    dialog.get_by_placeholder("Status name").fill("Underway")
    dialog.get_by_role("button", name="Save").click()
    expect(dialog).to_be_hidden()

    _status_button(page, "Hull survey").click()
    with _status_saved(page) as saved:
        page.get_by_role("menuitemradio", name="Underway").click()
    assert saved.value.status == 200, saved.value.text()

    with owner.client() as client:
        statuses = client.get(f"/api/v1/folders/{folder_id}").json()["meta"]["statuses"]
    assert [status["name"] for status in statuses] == ["Underway"], statuses
    assert _stored_status(owner, folder_id, chat_id) == statuses[0]["id"]
    page.reload()
    expect(_status_button(page, "Hull survey")).to_have_attribute(
        "aria-label", re.compile(r": Underway")
    )


def test_a_card_dragged_to_another_column_on_the_board_takes_its_status(page_for, make_user):
    owner = make_user()
    folder_id = _folder(owner, [TODO, DONE])
    chat_id = _chat(owner, folder_id, "Pier repair")
    _file(owner, folder_id, chat_id, TODO["id"])
    page = page_for(owner, path=f"/folders/{folder_id}")
    expect(_chat_link(page, "Pier repair")).to_be_visible()

    page.get_by_role("button", name="Display options").click()
    page.get_by_role("menuitemradio", name="Board").click()
    board = page.get_by_label("Conversation board")
    todo, done = board.get_by_role("region", name="To do"), board.get_by_role("region", name="Done")
    card = todo.get_by_role("article").filter(has=_chat_link(page, "Pier repair"))
    expect(card).to_be_visible()
    expect(done.get_by_role("article")).to_have_count(0)

    with _status_saved(page) as saved:
        card.drag_to(done)
    assert saved.value.status == 200, saved.value.text()

    assert _stored_status(owner, folder_id, chat_id) == DONE["id"]
    expect(done.get_by_role("article").filter(has=_chat_link(page, "Pier repair"))).to_be_visible()
    expect(todo.get_by_role("article")).to_have_count(0)
    page.reload()
    expect(
        page.get_by_label("Conversation board").get_by_role("region", name="Done")
    ).to_contain_text("Pier repair")


def test_a_tag_added_on_a_row_is_stored_and_filters_the_list(page_for, make_user):
    owner = make_user()
    tag = f"moorings{uuid.uuid4().hex[:4]}"
    folder_id = _folder(owner)
    chat_id = _chat(owner, folder_id, "Anchor log")
    _chat(owner, folder_id, "Buoy log")
    page = page_for(owner, path=f"/folders/{folder_id}")
    expect(_chat_link(page, "Buoy log")).to_be_visible()

    page.get_by_role("button", name="Edit tags for Anchor log").click()
    tag_box = page.get_by_label("Find or create a tag")
    tag_box.fill(tag)
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/tags")
    ) as added:
        tag_box.press("Enter")
    assert added.value.status == 200, added.value.text()
    with owner.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}/tags").json()
    assert [entry["name"] for entry in stored] == [tag], stored
    page.keyboard.press("Escape")

    _workspace(page).get_by_role("button", name="Filter").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Tags", exact=True).hover()
    page.get_by_role("menuitemcheckbox", name=re.compile(tag)).click()

    expect(_chat_link(page, "Buoy log")).to_have_count(0)
    expect(_chat_link(page, "Anchor log")).to_be_visible()


def test_the_folder_search_narrows_the_list_to_matching_chats(page_for, make_user):
    owner = make_user()
    word = f"lighthouse{uuid.uuid4().hex[:4]}"
    folder_id = _folder(owner)
    _chat(owner, folder_id, "Keeper rota", text=f"who minds the {word} tonight?")
    _chat(owner, folder_id, "Supply run")
    page = page_for(owner, path=f"/folders/{folder_id}")
    expect(_chat_link(page, "Supply run")).to_be_visible()

    _workspace(page).get_by_role("button", name="Search", exact=True).click()
    search = page.get_by_label("Search this folder")
    search.fill(word)
    search.press("Enter")

    expect(_chat_link(page, "Supply run")).to_have_count(0)
    expect(_chat_link(page, "Keeper rota")).to_be_visible()


def test_a_reading_member_gets_no_status_menu_or_tag_control_on_the_owners_chat(
    page_for, make_user
):
    owner, member = make_user(), make_user()
    folder_id = _folder(owner, [TODO, DONE])
    chat_id = _chat(owner, folder_id, "Harbour dues")
    _file(owner, folder_id, chat_id, DONE["id"])
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={
                "access_grants": [
                    {"principal_type": "user", "principal_id": member.id, "permission": "read"}
                ]
            },
        )
    assert shared.status_code == 200, shared.text

    owners_page = page_for(owner, path=f"/folders/{folder_id}")
    expect(_status_button(owners_page, "Harbour dues")).to_be_visible()
    members_page = page_for(member, path=f"/folders/{folder_id}")
    expect(_chat_link(members_page, "Harbour dues")).to_be_visible()

    expect(_status_button(members_page, "Harbour dues")).to_have_count(0)
    expect(members_page.get_by_role("button", name="Edit tags for Harbour dues")).to_have_count(0)
