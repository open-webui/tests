"""Journey: pages open as tabs beside the chat, and a new chat keeps them.

Since 0.13.4 a page picked from the sidebar (Notes) or from the user menu (Workspace) opens in a tab
of its own beside the chat that is open, which stays in its tab; picking the page again selects
its tab instead of adding one. Calendar, Automations and the Playground open the same way from the
user menu and, once pinned, from the sidebar, each showing its own content in its tab. A page
dragged from the sidebar into the tab bar opens as a tab there, and onto the edge of the chat it
opens in a pane beside it. A note or the home page linked in a reply opens the same way. Starting
a new chat, from New Chat or by opening a folder in the sidebar, swaps the chat in its tab and
leaves the page tabs beside it. A folder keeps tabs of its own and, reopened from the sidebar
after leaving it, comes back as it was left, with the tab last in front still in front.

Discriminates: passes on dev 485103cde. In a frontend copy of it whose sidebar, user menu and note
links navigate in place of opening a tab, all three fail (the chat's tab is gone, no Notes tab
opens); in one whose new chat clears the open tabs, the third fails while the note test passes.
The narrow window test fails on dev d3e87f8b1 (the tabs panel opens over the chat, #34314) and
passes with its fix 468db999e. The Calendar, Automations and Playground tests and the home page
link pass on dev f80a4f782; in a frontend copy of it whose user menu Calendar loads the page in
place of the app the user menu test fails (the chat's tab is gone), whose Automations tab has no
page the sidebar test fails, whose sidebar pages cannot be dragged the sidebar test fails (no tab
from the drop) and whose reply links to the home page are no page tab the link test fails.
The new chat test passes on dev f725d5c75 and fails on dev 5fc1b1b59, before ee7846286 made a
folder reopened from outside come back as it was left (the folder's own tab was in front).
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from contextlib import contextmanager

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.calendar_api import HOUR_NS, NO_REMINDER, create_event, default_calendar_id, to_ns
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID
from harness.workspace_search import create_note
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, expect_reply, send
from utils.page_tabs import page_tab
from utils.sidebar_folders import folder_link
from utils.user_menu import user_menu_trigger

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT = [
    {"role": "user", "content": "Which ferry crosses to the island?"},
    {"role": "assistant", "content": "The morning ferry leaves from the east pier."},
]


def _tabs(page: Page) -> Locator:
    return page.get_by_role("tablist", name="Tabs").first


def _tab(page: Page, name: str) -> Locator:
    return _tabs(page).get_by_role("tab", name=name, exact=True)


def _saved_chat(account, title: str) -> str:
    with account.client() as client:
        chat_id, _ = seed_chat(client, CHAT)
        renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert renamed.status_code == 200, renamed.text
    return chat_id


def _open_chat(page_for, account) -> tuple[Page, str, str]:
    title = f"Island ferry {uuid.uuid4().hex[:8]}"
    chat_id = _saved_chat(account, title)
    page = page_for(account, path=f"/c/{chat_id}")
    expect_reply(page, CHAT[1]["content"])
    return page, chat_id, title


def _expect_selected(page: Page, name: str, path: str) -> None:
    """The tab `name` is selected and the address shows its page (the workspace opens a section)."""
    expect(_tab(page, name)).to_have_attribute("aria-selected", "true")
    expect(page).to_have_url(re.compile(rf"{re.escape(path)}(?:/[a-z]+)?$"))


@contextmanager
def _calendar_automations_and_playground(account):
    """Each of the three pages with something of the account's own to show: an event today, an
    automation and the playground's message box. Yields (label, path, content) rows.
    """
    event_title = f"Harbour tour {uuid.uuid4().hex[:8]}"
    noon = dt.datetime.combine(dt.date.today(), dt.time(12)).astimezone()
    automation_name = f"Tide report {uuid.uuid4().hex[:8]}"
    with account.client() as client:
        event = create_event(
            client,
            default_calendar_id(client),
            title=event_title,
            start_at=to_ns(noon),
            end_at=to_ns(noon) + HOUR_NS,
            meta=NO_REMINDER,
        )
        assert event.status_code == 200, event.text
        automation = client.post(
            "/api/v1/automations/create",
            json={
                "name": automation_name,
                "is_active": False,
                "data": {
                    "prompt": "Write the tide report.",
                    "model_id": MOCK_MODEL_ID,
                    "rrule": "DTSTART:20990101T090000\nRRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
                    "target": {"type": "chat"},
                },
            },
        )
        assert automation.status_code == 200, automation.text
    try:
        yield [
            ("Calendar", "/calendar", lambda panel: panel.get_by_role("button", name=event_title)),
            ("Automations", "/automations", lambda panel: panel.get_by_text(automation_name)),
            (
                "Playground",
                "/playground",
                lambda panel: panel.get_by_role("textbox", name="Enter a user message here"),
            ),
        ]
    finally:
        with account.client() as client:
            client.delete(f"/api/v1/automations/{automation.json()['id']}/delete")


def _expect_page_content(page: Page, label: str, content) -> None:
    expect(
        content(page_tab(page, label)).first, f"the {label} tab opened without its page in it"
    ).to_be_visible()


def _drag_to(page: Page, source: Locator, x: float, y: float) -> None:
    """Drag `source` by the mouse to the point, in steps as a hand moves."""
    start = source.bounding_box()
    page.mouse.move(start["x"] + start["width"] / 2, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(start["x"] + start["width"] / 2 + 10, start["y"] + start["height"] / 2, steps=5)
    page.mouse.move(x, y, steps=20)
    page.mouse.up()


def _empty_part_of_tab_bar(page: Page) -> tuple[float, float]:
    """A point on the tab bar past its tabs and New tab button, where a dropped page joins it."""
    strip = _tabs(page)
    bar = strip.locator("xpath=ancestor::div[@data-panel-tab-bar]").bounding_box()
    tabs = strip.bounding_box()
    return (tabs["x"] + tabs["width"] + bar["x"] + bar["width"]) / 2, bar["y"] + bar["height"] / 2


def _back_to_the_chat(page: Page, chat_id: str, title: str) -> None:
    chat_tab = _tab(page, title)
    expect(chat_tab, "the chat's tab is gone once a page opened").to_be_visible()
    chat_tab.click()
    _expect_selected(page, title, f"/c/{chat_id}")
    expect(page.get_by_role("log").get_by_text(CHAT[1]["content"])).to_be_visible()


def test_a_page_from_the_sidebar_or_the_user_menu_opens_in_a_tab_beside_the_chat(
    page_for, make_user
):
    # an admin's user menu offers the workspace whatever the default permissions say
    account = make_user(role="admin")
    page, chat_id, title = _open_chat(page_for, account)

    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    _back_to_the_chat(page, chat_id, title)

    user_menu_trigger(page).click()
    page.get_by_role("menu").get_by_role("menuitem", name="Workspace").click()
    _expect_selected(page, "Workspace", "/workspace")
    _back_to_the_chat(page, chat_id, title)

    # a page already in a tab is selected again, not opened twice
    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    # the chat, Notes and the workspace
    expect(_tabs(page).get_by_role("tab")).to_have_count(3)


def test_calendar_automations_and_the_playground_open_from_the_user_menu_in_tabs_beside_the_chat(
    page_for, make_user
):
    # the playground is offered to admins only
    account = make_user(role="admin")
    with _calendar_automations_and_playground(account) as pages:
        page, chat_id, title = _open_chat(page_for, account)
        for label, path, content in pages:
            user_menu_trigger(page).click()
            page.get_by_role("menu").get_by_role("menuitem", name=label, exact=True).click()
            _expect_selected(page, label, path)
            _expect_page_content(page, label, content)
            _back_to_the_chat(page, chat_id, title)
        # the chat and the three pages
        expect(_tabs(page).get_by_role("tab")).to_have_count(4)


def test_a_page_from_the_sidebar_opens_in_a_tab_or_dragged_beside_the_chat(page_for, make_user):
    account = make_user(role="admin")
    with account.client() as client:
        pinned = client.post(
            "/api/v1/users/user/settings/update",
            json={
                "ui": {
                    "showChangelog": False,
                    "pinnedMenuItems": ["calendar", "automations", "playground"],
                }
            },
        )
    assert pinned.status_code == 200, pinned.text
    with _calendar_automations_and_playground(account) as pages:
        (calendar, calendar_path, calendar_content), automations, playground = pages
        page, chat_id, title = _open_chat(page_for, account)

        label, path, content = playground
        page.get_by_role("link", name=label, exact=True).click()
        _expect_selected(page, label, path)
        _expect_page_content(page, label, content)
        _back_to_the_chat(page, chat_id, title)

        _drag_to(
            page,
            page.get_by_role("link", name=calendar, exact=True),
            *_empty_part_of_tab_bar(page),
        )
        expect(
            _tab(page, calendar), "Calendar dropped on the tab bar opened no tab"
        ).to_be_visible()
        expect(page.get_by_role("tablist", name="Tabs")).to_have_count(1)
        _tab(page, calendar).click()
        _expect_selected(page, calendar, calendar_path)
        _expect_page_content(page, calendar, calendar_content)
        _back_to_the_chat(page, chat_id, title)

        label, path, content = automations
        chat_area = page.get_by_role("tabpanel", name=title, exact=True).bounding_box()
        _drag_to(
            page,
            page.get_by_role("link", name=label, exact=True),
            chat_area["x"] + chat_area["width"] - 20,
            chat_area["y"] + chat_area["height"] / 2,
        )
        # dropped on the chat's right edge it opens in a pane of its own beside the chat
        panes = page.get_by_role("tablist", name="Tabs")
        expect(panes, "the page dropped on the chat's edge opened no pane").to_have_count(2)
        expect(panes.last.get_by_role("tab")).to_have_count(1)
        expect(panes.last.get_by_role("tab", name=label, exact=True)).to_have_attribute(
            "aria-selected", "true"
        )
        expect(_tab(page, title)).to_have_attribute("aria-selected", "true")
        _expect_page_content(page, label, content)
        expect(
            page.get_by_role("log").get_by_text(CHAT[1]["content"]),
            "the chat is no longer shown beside the dropped page",
        ).to_be_visible()


def test_a_note_or_the_home_page_linked_in_a_reply_opens_in_a_tab_beside_the_chat(
    page_for, make_user, upstream
):
    account = make_user()
    note_title = f"Packing list {uuid.uuid4().hex[:8]}"
    with account.client() as client:
        note_id = create_note(client, note_title, "Rain jacket, ferry tickets.")
    page, chat_id, title = _open_chat(page_for, account)
    question = f"Where is my packing list? ({uuid.uuid4().hex[:8]})"
    upstream.queue(
        reply.text(
            f"It is in your note [packing list](/notes/{note_id}), or start from the "
            "[home page](/home).",
            match=reply.answering(question),
        )
    )
    send(page, question)
    expect_reply(page, "It is in your note")

    page.get_by_role("log").get_by_role("button", name=re.compile(re.escape(note_title))).click()

    _expect_selected(page, note_title, f"/notes/{note_id}")
    _back_to_the_chat(page, chat_id, title)

    page.get_by_role("log").get_by_role("link", name="home page", exact=True).click()
    _expect_selected(page, "Home", "/home")
    _back_to_the_chat(page, chat_id, title)


def test_a_narrow_window_shows_the_chat_once_the_tab_beside_it_is_closed(page_for, make_user):
    """Regression open-webui/open-webui#34314: once the tab beside a chat was closed, narrowing
    the window to phone width covered the chat with an empty tabs panel."""
    account = make_user()
    page, chat_id, title = _open_chat(page_for, account)
    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    _tabs(page).get_by_role("button", name="Close Notes", exact=True).click()
    expect(_tab(page, "Notes")).to_have_count(0)
    _expect_selected(page, title, f"/c/{chat_id}")

    page.set_viewport_size({"width": 390, "height": 844})
    tabs_button = page.get_by_role("button", name="Tabs", exact=True)
    drawer = page.get_by_role("dialog", name="Tabs")
    expect(tabs_button, "narrowing the window opened the tabs panel").to_have_attribute(
        "aria-expanded", "false"
    )
    expect(drawer, "an empty tabs panel covers the chat").to_have_count(0)
    expect(page.get_by_role("log").get_by_text(CHAT[1]["content"])).to_be_in_viewport()

    # the panel still opens when asked for
    tabs_button.click()
    expect(drawer).to_be_visible()
    drawer.get_by_role("button", name="Hide tabs").click()
    expect(drawer).to_have_count(0)
    expect(tabs_button).to_have_attribute("aria-expanded", "false")


def test_a_new_chat_keeps_the_page_tabs_open_beside_the_chat(page_for, make_user):
    account = make_user()
    folder_name = f"Islands {uuid.uuid4().hex[:8]}"
    with account.client() as client:
        created = client.post("/api/v1/folders/", json={"name": folder_name})
    assert created.status_code == 200, created.text
    folder_id = created.json()["id"]
    page, _, title = _open_chat(page_for, account)
    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    _tab(page, title).click()

    page.get_by_role("link", name="New Chat").first.click()
    expect(page).to_have_url(re.compile(r"/$"))
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(_tab(page, title)).to_have_count(0)
    expect(_tab(page, "Notes"), "a new chat closed the Notes tab").to_be_visible()

    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    folder_link(sidebar, folder_name).click()
    expect(page).to_have_url(re.compile(rf"/folders/{folder_id}$"))
    expect(_tab(page, folder_name)).to_have_attribute("aria-selected", "true")
    # since 1fbb42be9 a folder keeps tabs of its own: the chat's are not carried in, its own stay
    expect(_tab(page, "Notes"), "the chat's Notes tab was carried into the folder").to_have_count(0)
    page.get_by_role("link", name="Notes", exact=True).click()
    _expect_selected(page, "Notes", "/notes")
    page.get_by_role("link", name="New Chat").first.click()
    expect(page).to_have_url(re.compile(r"/$"))
    # since ee7846286 a folder reopened from outside comes back as it was left, Notes in front
    folder_link(sidebar, folder_name).click()
    expect(_tab(page, "Notes"), "the folder lost the Notes tab opened in it").to_be_visible()
    _expect_selected(page, "Notes", "/notes")
    expect(_tabs(page).get_by_role("tab")).to_have_count(2)
    _tab(page, folder_name).click()
    _expect_selected(page, folder_name, f"/folders/{folder_id}")
