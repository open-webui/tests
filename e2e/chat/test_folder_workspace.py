"""Journey: a folder page's statuses, board, tags, filter and search, as a person uses them.

Since 06491f013 a folder's page lists its chats with a status control and tags on every row. A
person adds a status in the Statuses dialog, reached from a chat's status menu, and picks it for
the chat from the same menu; the pick is stored and shown after a reload. Under Display options
the page switches to a board with a column per status, kept after a reload, and a card dragged to
another column takes that column's status. A tag typed into a chat's tag control is stored on the
chat, and the Filter menu's Tags entry then narrows the list to the chats carrying it; the folder
search narrows the list to chats whose title or messages match. A member who may only read the
shared folder sees the owner's chats with neither a status menu nor a tag control. Since 121bc0ec5
a folder that never set up statuses offers five ready-made ones (Backlog, To do, In progress,
Waiting, Done) in the status menu. Since 6378f94a6 the page's section switch shows the folder's
automations: a member who may write pauses one with its Active switch, which a reading member
finds disabled, and a chat an automation made carries the "Made by an automation" mark.

Discriminates: journey, no fix. All pass on the dev 941bf3674 build. With the dev build on a
backend copy of it whose status route stores nothing the status pick and board drag tests go red,
whose tag filter is ignored the tag test goes red, whose folder search matches nothing the search
test goes red and whose chats all read as editable the reading member test goes red. The tests
added for 121bc0ec5 and 6378f94a6 pass on the dev 485103cde build; with that build on backend
copies of it, a folder without statuses offering none turns the ready-made test red, the manage
check falling back to the creator alone the pause test (the toggle answers 404), and the chat
meta no longer carrying the automation the mark test.
"""

from __future__ import annotations

import contextlib
import re
import time
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from utils.folder_page import folder_chat, folder_page
from utils.menus import late_menu_entry

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TODO = {"id": "s_todo", "name": "To do", "glyph": "todo", "color": "gray"}
DONE = {"id": "s_done", "name": "Done", "glyph": "done", "color": "green"}
READY_MADE = ("Backlog", "To do", "In progress", "Waiting", "Done")
# a far-off start, so the automations here never run on their own schedule
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"
RUN_WAIT = 30.0


def _folder(owner: Actor, statuses: list[dict] | None = None) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"Yard {uuid.uuid4().hex[:6]}"})
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
        if statuses is not None:
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


def _status_button(page: Page, title: str) -> Locator:
    return page.get_by_role("button", name=re.compile(f"^Status for {re.escape(title)}: "))


def _status_saved(page: Page):
    return page.expect_response(
        lambda response: response.request.method == "PATCH" and response.url.endswith("/status")
    )


def test_a_status_added_in_the_dialog_is_picked_for_a_chat_and_kept(page_for, make_user):
    owner = make_user()
    # an empty list opts out of the ready-made statuses a new folder starts with (121bc0ec5)
    folder_id = _folder(owner, [])
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
    expect(folder_chat(page, "Pier repair")).to_be_visible()

    page.get_by_role("button", name="Display options").click()
    page.get_by_role("menuitemradio", name="Board").click()
    board = page.get_by_label("Conversation board")
    todo, done = board.get_by_role("region", name="To do"), board.get_by_role("region", name="Done")
    expect(folder_chat(todo, "Pier repair")).to_be_visible()
    card = todo.get_by_role("article")
    expect(card).to_have_count(1)
    expect(done.get_by_role("article")).to_have_count(0)

    with _status_saved(page) as saved:
        card.drag_to(done)
    assert saved.value.status == 200, saved.value.text()

    assert _stored_status(owner, folder_id, chat_id) == DONE["id"]
    expect(folder_chat(done, "Pier repair")).to_be_visible()
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
    expect(folder_chat(page, "Buoy log")).to_be_visible()

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

    folder_page(page).get_by_role("button", name="Filter").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Tags", exact=True).hover()
    page.get_by_role("menuitemcheckbox", name=re.compile(tag)).click()

    expect(folder_chat(page, "Buoy log")).to_have_count(0)
    expect(folder_chat(page, "Anchor log")).to_be_visible()


def test_the_folder_search_narrows_the_list_to_matching_chats(page_for, make_user):
    owner = make_user()
    word = f"lighthouse{uuid.uuid4().hex[:4]}"
    folder_id = _folder(owner)
    _chat(owner, folder_id, "Keeper rota", text=f"who minds the {word} tonight?")
    _chat(owner, folder_id, "Supply run")
    page = page_for(owner, path=f"/folders/{folder_id}")
    expect(folder_chat(page, "Supply run")).to_be_visible()

    folder_page(page).get_by_role("button", name="Search", exact=True).click()
    search = page.get_by_label("Search this folder")
    search.fill(word)
    search.press("Enter")

    expect(folder_chat(page, "Supply run")).to_have_count(0)
    expect(folder_chat(page, "Keeper rota")).to_be_visible()


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
    expect(folder_chat(members_page, "Harbour dues")).to_be_visible()

    expect(_status_button(members_page, "Harbour dues")).to_have_count(0)
    expect(members_page.get_by_role("button", name="Edit tags for Harbour dues")).to_have_count(0)


def test_a_new_folders_status_menu_offers_the_ready_made_statuses(page_for, make_user):
    owner = make_user()
    folder_id = _folder(owner)
    chat_id = _chat(owner, folder_id, "Net mending")
    page = page_for(owner, path=f"/folders/{folder_id}")

    _status_button(page, "Net mending").click()
    for name in READY_MADE:
        expect(page.get_by_role("menuitemradio", name=name, exact=True)).to_be_visible()
    with _status_saved(page) as saved:
        page.get_by_role("menuitemradio", name="In progress", exact=True).click()

    assert saved.value.status == 200, saved.value.text()
    assert _stored_status(owner, folder_id, chat_id) == "in_progress"
    expect(_status_button(page, "Net mending")).to_have_attribute(
        "aria-label", re.compile(r": In progress")
    )


@pytest.fixture
def automations_allowed(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text


def _share(owner: Actor, folder_id: str, writer: Actor, reader: Actor) -> None:
    grants = [
        {"principal_type": "user", "principal_id": account.id, "permission": permission}
        for account, permission in ((writer, "read"), (writer, "write"), (reader, "read"))
    ]
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


@contextlib.contextmanager
def _automation(owner: Actor, folder_id: str, name: str) -> Iterator[dict]:
    """An active automation in the folder, on a schedule far off; deleted afterwards."""
    with owner.client() as client:
        created = client.post(
            "/api/v1/automations/create",
            json={
                "name": name,
                "is_active": True,
                "folder_id": folder_id,
                "data": {
                    "prompt": f"Count the boats, batch {uuid.uuid4().hex[:6]}.",
                    "model_id": MOCK_MODEL_ID,
                    "rrule": SCHEDULE,
                },
            },
        )
        assert created.status_code == 200, created.text
        try:
            yield created.json()
        finally:
            client.delete(f"/api/v1/automations/{created.json()['id']}/delete")


def _show_automations(page: Page) -> Locator:
    folder_page(page).get_by_role("button", name="Switch section").click()
    page.get_by_role("menuitemradio", name=re.compile("^Automations")).click()
    return folder_page(page).get_by_role("list", name="Automations")


def test_a_writing_member_pauses_an_automation_in_the_folders_automations_section(
    page_for, make_user, automations_allowed
):
    owner, writer, reader = make_user(), make_user(), make_user()
    folder_id = _folder(owner)
    _share(owner, folder_id, writer, reader)
    name = f"Boat count {uuid.uuid4().hex[:4]}"

    with _automation(owner, folder_id, name) as automation:
        page = page_for(writer, path=f"/folders/{folder_id}")
        row = _show_automations(page).get_by_role("listitem").filter(has_text=name)
        switch = row.get_by_role("switch", name="Active")
        expect(switch).to_be_checked()
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith(f"/automations/{automation['id']}/toggle")
            )
        ) as toggled:
            switch.click()
        assert toggled.value.status == 200, toggled.value.text()
        expect(switch).not_to_be_checked()
        expect(row).to_contain_text("Paused")
        with owner.client() as client:
            stored = client.get(f"/api/v1/automations/{automation['id']}").json()
        assert stored["is_active"] is False, f"the member's pause was not stored: {stored}"

        readers_page = page_for(reader, path=f"/folders/{folder_id}")
        readers_row = _show_automations(readers_page).get_by_role("listitem").filter(has_text=name)
        expect(readers_row).to_contain_text("Paused")
        expect(readers_row.get_by_role("switch", name="Active")).to_be_disabled()


def test_a_chat_made_by_an_automation_is_marked_in_the_folders_list(
    page_for, make_user, upstream, automations_allowed
):
    owner = make_user()
    folder_id = _folder(owner)
    _chat(owner, folder_id, "Hand written")
    name = f"Tide report {uuid.uuid4().hex[:4]}"

    with _automation(owner, folder_id, name) as automation:
        upstream.queue(
            reply.text("Low water at six.", match=reply.answering(automation["data"]["prompt"]))
        )
        with owner.client() as client:
            started = client.post(f"/api/v1/automations/{automation['id']}/run")
            assert started.status_code == 200, started.text
        _wait_for_run(owner, automation["id"])
        page = page_for(owner, path=f"/folders/{folder_id}")

        expect(folder_chat(page, "Hand written")).to_be_visible()
        expect(folder_chat(page, name)).to_be_visible()
        marks = folder_page(page).get_by_label("Made by an automation")
        expect(marks).to_have_count(1)
        row = (
            folder_page(page)
            .get_by_role("listitem")
            .filter(has=page.get_by_role("button", name=name, exact=True))
        )
        expect(row.get_by_label("Made by an automation")).to_be_visible()


def _wait_for_run(owner: Actor, automation_id: str) -> None:
    with owner.client() as client:
        deadline = time.monotonic() + RUN_WAIT
        while time.monotonic() < deadline:
            runs = client.get(f"/api/v1/automations/{automation_id}/runs").json()
            if runs:
                assert runs[0]["status"] == "success", runs[0]
                return
            time.sleep(0.2)
    raise AssertionError(f"the automation recorded no run within {RUN_WAIT}s")
