"""One chat on the page, in the main view or in a side tab, driven the same way.

A page shows its chat in the main view and can hold further chats in tabs beside it: a side chat
(`/side`, `/btw`, the New Chat tab shortcut), a new chat opened from a pinned model, or a saved
chat dragged in from the sidebar. Each of them is a `ChatSurface`: the tab panel holding the
chat, with its message box, its messages and the buttons on them. The helpers in
`utils/chat_ui.py` take a surface wherever they take a page, so one test body runs in either
place and every difference between the two stays in this module.

What is not part of a chat's panel belongs to the page and is reached through `surface.page`:
dialogs, menus, tooltips and notifications, the sidebar, the address and the keyboard.

Beside a saved chat the address stays on that chat while a side tab is selected (c63565927,
open-webui/open-webui#34304), so a reload brings the side tab back beside it. Beside a new chat,
which has no address yet, the address follows the selected side tab, and a reload opens that
tab's chat as the page's own chat. `surface.reload()` therefore hands back the surface the chat is
on after the reload, and a side tab's chat is found by `surface.chat_id` or, by what was said in
it, `stored_chat_holding`.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.plain_http import loopback_url
from harness.upstream import MOCK_MODEL_ID

REPLY_TIMEOUT_MS = 30_000
MAIN_CHAT = "main chat"
SIDE_CHAT = "side chat"
MAIN_PANEL_ID = "chat-panel-main-chat-view"
NEW_TAB = re.compile(r"^New Chat \d+$")
_CHAT_ADDRESS = re.compile(r"/c/([^/?#]+)(?:[?#].*)?$")


@dataclass
class ChatSurface:
    page: Page
    panel: Locator
    place: str
    # the saved chat a side tab was opened beside, whose address the page keeps
    beside_chat_id: str | None = None

    @property
    def is_side(self) -> bool:
        return self.place != MAIN_CHAT

    @property
    def input(self) -> Locator:
        """The chat's message box."""
        # a message being edited is a text box of the panel as well
        return self.panel.locator("[role='textbox'][id*='chat-input']")

    @property
    def conversation(self) -> Locator:
        return self.panel.get_by_label("Chat Conversation")

    @property
    def replies(self) -> Locator:
        return self.conversation.locator(".chat-assistant")

    @property
    def last_reply(self) -> Locator:
        return self.replies.last

    @property
    def messages(self) -> Locator:
        """The chat's messages, a question or a reply each, with the buttons under them."""
        return self.panel.get_by_role("log").get_by_role("listitem")

    @property
    def stop_button(self) -> Locator:
        return self.panel.get_by_role("button", name="Stop")

    def button(self, name: str | re.Pattern) -> Locator:
        """A button of the chat's panel: under a message or beside the message box."""
        return self.panel.get_by_role("button", name=name, exact=True)

    @property
    def tab(self) -> Locator:
        """The chat's tab; its name is the chat's title."""
        label = self.panel.get_attribute("aria-labelledby")
        assert label, "the chat's panel names no tab"
        return self.page.locator(f"[id='{label}']")

    @property
    def title(self) -> str:
        """The title of the chat in a side tab, which the tab is named after."""
        expect(self.tab).not_to_have_attribute("title", NEW_TAB, timeout=REPLY_TIMEOUT_MS)
        title = self.tab.get_attribute("title")
        assert title, "the side tab names no chat"
        return title

    @property
    def chat_id(self) -> str:
        """The id of the saved chat on this surface."""
        if not self.beside_chat_id:
            expect(self.page).to_have_url(_CHAT_ADDRESS, timeout=REPLY_TIMEOUT_MS)
            return _CHAT_ADDRESS.search(self.page.url).group(1)
        title = self.title
        deadline = time.monotonic() + REPLY_TIMEOUT_MS / 1000
        while True:
            found = [
                chat["id"]
                for chat in _account_chats(self.page)
                if chat["title"] == title and chat["id"] != self.beside_chat_id
            ]
            if len(found) == 1:
                return found[0]
            assert time.monotonic() < deadline, (
                f"the account holds {len(found)} chats titled {title!r} beside the main chat"
            )
            time.sleep(0.2)

    @property
    def tab_bar(self) -> Locator:
        """The bar holding the chat's tab, with the "Chat actions" menu of the chat it shows."""
        return self.page.locator("[data-panel-tab-bar]").filter(has=self.tab)

    def show(self) -> None:
        """Select the chat's tab, which makes it the chat the address and a reload follow."""
        # a panel in view can still be the unfocused one of two side by side
        self.tab.click()
        expect(self.panel).to_be_visible()

    def send(self, text: str) -> None:
        expect(self.input).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        self.input.click()
        self.page.keyboard.type(text)
        self.page.keyboard.press("Enter")

    def expect_reply(self, text: str) -> None:
        expect(self.last_reply).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)

    def reload(self) -> ChatSurface:
        """Reload the page with this chat selected; returns the surface it is on afterwards.

        A side tab beside a saved chat comes back beside it, and one beside a new chat opens as
        the page's chat."""
        if not self.is_side:
            self.page.reload()
            return main_chat(self.page)
        if self.panel.is_visible():
            # pressing on the panel focuses its chat as a click into it does, also under a dialog
            self.panel.dispatch_event("pointerdown")
        else:
            self.show()
        if not self.beside_chat_id:
            self.page.reload()
            return main_chat(self.page)
        title = self.title
        self.page.reload()
        expect(self.page, "the reload left the chat the side tab is beside").to_have_url(
            re.compile(rf"/c/{re.escape(self.beside_chat_id)}(?:[?#].*)?$")
        )
        tab = self.page.get_by_role("tab", name=side_tab_name(title))
        expect(tab).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        # a tab sharing the main chat's bar comes back behind it, unlike a tab split beside it
        if tab.get_attribute("aria-selected") != "true":
            tab.click()
        return side_chat_titled(self.page, title, self.place, self.beside_chat_id)


def stored_chat_holding(actor, text: str) -> dict:
    """The one stored chat of `actor` with a message reading `text`.

    A tab beside a saved chat leaves the address on that chat (c63565927), so a test finds the
    chat a tab shows by what was said in it."""
    holding = []
    with actor.client() as client:
        listed = client.get("/api/v1/chats/", params={"page": 1})
        assert listed.status_code == 200, listed.text
        for entry in listed.json():
            stored = client.get(f"/api/v1/chats/{entry['id']}")
            assert stored.status_code == 200, stored.text
            messages = stored.json()["chat"].get("history", {}).get("messages", {})
            if any(message.get("content") == text for message in messages.values()):
                holding.append(stored.json())
    assert len(holding) == 1, (
        f"expected one stored chat saying {text!r}, found {[chat['title'] for chat in holding]}"
    )
    return holding[0]


def main_chat(page: Page) -> ChatSurface:
    return ChatSurface(page, page.locator(f"#{MAIN_PANEL_ID}"), MAIN_CHAT)


def surface_of(chat: Page | ChatSurface) -> ChatSurface:
    """The chat a helper is handed: a page stands for the page's main chat."""
    return chat if isinstance(chat, ChatSurface) else main_chat(chat)


def _page_chat_id(page: Page) -> str | None:
    """The saved chat the page's address names, if any."""
    found = _CHAT_ADDRESS.search(page.url)
    return found.group(1) if found else None


def _held(
    page: Page, panel: Locator, place: str = SIDE_CHAT, beside: str | None = None
) -> ChatSurface:
    """The surface on `panel`, held by its id: the panel's name changes with the chat's title."""
    expect(panel).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    panel_id = panel.get_attribute("id")
    assert panel_id, "the tab's panel has no id to hold on to it by"
    assert panel_id != MAIN_PANEL_ID, f"the {place} opened as the page's own chat, not in a tab"
    surface = ChatSurface(page, page.locator(f"[id='{panel_id}']"), place, beside)
    expect(surface.input).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return surface


def side_tab_name(title: str) -> re.Pattern:
    """The name of the side tab of the chat titled `title`; while it answers, a spinner on the tab
    reads out "Generating..." before the title."""
    return re.compile(rf"^(?:Generating\.\.\. )?{re.escape(title)}$")


def side_chat_titled(
    page: Page, title: str, place: str = SIDE_CHAT, beside: str | None = None
) -> ChatSurface:
    """The side tab whose chat is titled `title`: a side chat takes its first question as title.
    `beside` is the saved chat the tab sits beside."""
    return _held(page, page.get_by_role("tabpanel", name=side_tab_name(title)), place, beside)


def _panels(page: Page) -> Locator:
    # by attribute: a tab that is not the selected one of its group keeps its panel, hidden
    return page.locator("[role='tabpanel']")


def _panel_ids(page: Page) -> set[str]:
    return set(_panels(page).evaluate_all("panels => panels.map((panel) => panel.id)"))


def _panel_ids_once_ready(page: Page) -> set[str]:
    # the page's own chat mounts its panel a moment after the load
    expect(main_chat(page).input).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return _panel_ids(page)


def _new_panel(page: Page, before: set[str], place: str, beside: str | None) -> ChatSurface:
    """The tab panel that was not on the page `before`, once it shows a message box; `beside` is
    the saved chat the page showed before the tab opened."""
    expect(_panels(page)).to_have_count(len(before) + 1, timeout=REPLY_TIMEOUT_MS)
    (panel_id,) = _panel_ids(page) - before
    return _held(page, page.locator(f"[id='{panel_id}']"), place, beside)


def open_new_chat_tab(page: Page, place: str = SIDE_CHAT) -> ChatSurface:
    """Open an empty side chat with the New Chat tab shortcut."""
    before = _panel_ids_once_ready(page)
    beside = _page_chat_id(page)
    page.keyboard.press("Control+Alt+O")
    return _new_panel(page, before, place, beside)


def open_side_chat(page: Page, command: str = "side", place: str = SIDE_CHAT) -> ChatSurface:
    """Open an empty side chat by sending `/side` (or `/btw`) alone in the main chat."""
    before = _panel_ids_once_ready(page)
    beside = _page_chat_id(page)
    main_chat(page).send(f"/{command}")
    return _new_panel(page, before, place, beside)


def ask_aside(
    page: Page, question: str, command: str = "side", place: str = SIDE_CHAT
) -> ChatSurface:
    """Send `/side <question>` in the main chat; the side chat it opens takes the question as its
    title once it is saved."""
    before = _panel_ids_once_ready(page)
    beside = _page_chat_id(page)
    main_chat(page).send(f"/{command} {question}")
    return _new_panel(page, before, place, beside)


def show_the_folders_chat(page: Page, folder_name: str, directory: str | None = None) -> None:
    """Show the chat input of a folder page that just opened.

    A folder on a machine opens its page on the Files section, browsing its `directory` (or the
    machine's home), with the chat input below it (24f49bf46; a Files tab in front of the folder's
    own tab before, 1fbb42be9)."""
    folder = page.get_by_role("main").get_by_role("region", name="Folder")
    expect(folder.get_by_role("button", name=folder_name, exact=True)).to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    if directory is not None:
        # the folder page reads the folder and its machine before it shows Files, a cold load
        expect(folder.get_by_role("button", name="Switch section")).to_have_text(
            "Files", timeout=REPLY_TIMEOUT_MS
        )
        path = folder.get_by_role("region", name="File browser").get_by_role("textbox", name="Path")
        shown = directory.rstrip("/").rsplit("/", 1)[-1] or "Home"
        expect(path.get_by_role("button", name=shown, exact=True)).to_be_visible(
            timeout=REPLY_TIMEOUT_MS
        )
    expect(main_chat(page).input).to_be_visible(timeout=REPLY_TIMEOUT_MS)


def start_on_folder_page(page: Page, question: str, place: str = "folder chat") -> ChatSurface:
    """Send `question` from a folder page's box; the chat it starts opens in a tab beside the
    folder (since 06b47d455)."""
    before = _panel_ids_once_ready(page)
    beside = _page_chat_id(page)
    main_chat(page).send(question)
    return _new_panel(page, before, place, beside)


def open_sidebar(page: Page) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    opener = page.get_by_role("button", name="Open Sidebar", exact=True)
    if opener.is_visible():
        opener.click()
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    return sidebar


def close_sidebar(page: Page) -> None:
    page.get_by_role("button", name="Close Sidebar", exact=True).click()
    expect(page.get_by_role("button", name="Open Sidebar", exact=True)).to_be_visible()


def drag_beside_the_chat(page: Page, row: Locator) -> Locator:
    """Drag a sidebar chat over the right side of the main chat; returns the drop hint.

    The mouse is still down when this returns, so the caller sees the hint and then drops."""
    # a saved chat holds its reply a moment before its panel shows, and a box needs it shown
    chat_panel = main_chat(page).panel
    expect(chat_panel).to_be_visible()
    expect(row).to_be_visible()
    target = chat_panel.bounding_box()
    start = row.bounding_box()
    assert target and start, "the chat or the sidebar row is not on screen"
    page.mouse.move(start["x"] + start["width"] / 2, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(
        target["x"] + target["width"] * 0.9, target["y"] + target["height"] / 2, steps=20
    )
    return page.get_by_role("status").filter(has_text="Open in side panel")


def drop_chat_beside(page: Page, title: str, place: str = SIDE_CHAT) -> ChatSurface:
    """Drag the saved chat titled `title` from the sidebar into a tab beside the main chat."""
    row = open_sidebar(page).get_by_role("button", name=re.compile(rf"^{re.escape(title)}(?: |$)"))
    expect(row).to_be_visible()
    before = _panel_ids(page)
    beside = _page_chat_id(page)
    expect(drag_beside_the_chat(page, row)).to_be_visible()
    page.mouse.up()
    surface = _new_panel(page, before, place, beside)
    close_sidebar(page)
    return surface


def open_pinned_model_tab(page: Page, model_name: str, place: str = SIDE_CHAT) -> ChatSurface:
    """Click a model pinned to the sidebar, which opens a new chat on it in a tab, in front (#35160)."""
    sidebar = open_sidebar(page)
    before = _panel_ids(page)
    beside = _page_chat_id(page)
    sidebar.get_by_role("link", name=re.compile(rf"\b{re.escape(model_name)}$")).click()
    surface = _new_panel(page, before, place, beside)
    close_sidebar(page)
    return surface


def open_chat_tab_from(page: Page, link: Locator, place: str = SIDE_CHAT) -> ChatSurface:
    """Click `link`, which opens a new chat in a tab (a `/?model=` link since da850248d)."""
    before = _panel_ids(page)
    beside = _page_chat_id(page)
    link.click()
    return _new_panel(page, before, place, beside)


def reload_beside_the_chat(side: ChatSurface, title: str) -> ChatSurface:
    """Reload with the main tab selected and select the saved side tab titled `title` again."""
    page = side.page
    main_chat(page).show()
    page.reload()
    tab = page.get_by_role("tab", name=side_tab_name(title))
    expect(tab).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    tab.click()
    return side_chat_titled(page, title, side.place, side.beside_chat_id)


# --- the places a test body can run in ----------------------------------------------------------

ASKED_ASIDE = "side chat asked with /side"
OPENED_ASIDE = "side chat opened with /side"
TAB_BESIDE_A_CHAT = "new chat tab beside a chat"
PINNED_MODEL_TAB = "pinned model tab"
DRAGGED_TAB = "saved chat dragged into a tab"
TEMPORARY_SIDE = "temporary side chat"
AFTER_RELOAD = "side chat after a reload"
AFTER_SWITCHING = "side chat after switching chats"
ADMIN_SIDE = "side chat of an admin"
BESIDE_ANSWERING = "side chat while the chat answers"
SECOND_SIDE = "second side chat"

# every way a side chat starts and every state it can be in, each one step from the plain one
SIDE_CONDITIONS = (
    SIDE_CHAT,
    ASKED_ASIDE,
    OPENED_ASIDE,
    TAB_BESIDE_A_CHAT,
    PINNED_MODEL_TAB,
    DRAGGED_TAB,
    TEMPORARY_SIDE,
    AFTER_RELOAD,
    AFTER_SWITCHING,
    ADMIN_SIDE,
    BESIDE_ANSWERING,
    SECOND_SIDE,
)
EVERYWHERE = (MAIN_CHAT, *SIDE_CONDITIONS)
# the conditions whose side chat is still empty, for what happens on a chat's first message
EMPTY_SIDE_CONDITIONS = (
    SIDE_CHAT,
    OPENED_ASIDE,
    TAB_BESIDE_A_CHAT,
    PINNED_MODEL_TAB,
    TEMPORARY_SIDE,
    ADMIN_SIDE,
    BESIDE_ANSWERING,
    SECOND_SIDE,
)

PARENT_CHAT = [
    {"role": "user", "content": "Which pass links Lienz and Matrei?"},
    {"role": "assistant", "content": "The Felbertauern road links them through a tunnel."},
]
DRAGGED_CHAT = [
    {"role": "user", "content": "What should I pack for the Grossglockner road?"},
    {"role": "assistant", "content": "A warm jacket, the road tops out at 2,504 metres."},
]
OPENING_ANSWER = "The Felbertauern tunnel is 5.3 kilometres long."
STILL_ANSWERING = ["The main ", "chat is ", "still ", "answering ", "this."]


@dataclass
class ChatOpener:
    """Opens the chat a test body runs in, at the place named `place`.

    `open(actor)` gives a new chat there and `open(actor, chat_id=...)` the saved chat `chat_id`:
    in the main view it is the page's chat, in a side tab it is dragged in from the sidebar.
    Further keywords go to the browser context, as with `page_for`."""

    place: str
    page_for: Callable[..., Page]
    upstream: object
    admin: object
    held: list[threading.Event]
    parent_title: str = ""

    def __call__(self, actor, chat_id: str | None = None, **context_options) -> ChatSurface:
        if self.place == MAIN_CHAT:
            path = f"/c/{chat_id}" if chat_id else "/"
            return main_chat(self.page_for(actor, path=path, **context_options))
        if self.place == SIDE_CHAT:
            page = self.page_for(actor, **context_options)
            expect(main_chat(page).input).to_be_visible(timeout=REPLY_TIMEOUT_MS)
            if chat_id:
                return drop_chat_beside(page, _title_of(actor, chat_id), self.place)
            return open_new_chat_tab(page, self.place)
        if chat_id:
            assert self.place == DRAGGED_TAB, f"a saved chat cannot be opened as a {self.place}"
        return _CONDITIONS[self.place](self, actor, chat_id, context_options)

    def beside_a_chat(self, actor, context_options: dict) -> Page:
        """A page on a saved chat of `actor` with one answered question, titled `parent_title`."""
        self.parent_title = f"Lienz to Matrei {uuid.uuid4().hex[:8]}"
        with actor.client() as client:
            chat_id, _ = seed_chat(client, PARENT_CHAT)
            renamed = client.post(
                f"/api/v1/chats/{chat_id}", json={"chat": {"title": self.parent_title}}
            )
        assert renamed.status_code == 200, renamed.text
        page = self.page_for(actor, path=f"/c/{chat_id}", **context_options)
        expect(main_chat(page).last_reply).to_contain_text(
            PARENT_CHAT[1]["content"], timeout=REPLY_TIMEOUT_MS
        )
        return page

    def asked_aside(self, actor, context_options: dict) -> tuple[ChatSurface, str]:
        """A saved side chat with one answered question; the surface and the chat's title."""
        page = self.beside_a_chat(actor, context_options)
        question = f"By the way, how long is that tunnel? ({uuid.uuid4().hex[:8]})"
        self.upstream.queue(reply.text(OPENING_ANSWER, match=reply.answering(question)))
        side = ask_aside(page, question, place=self.place)
        side.expect_reply(OPENING_ANSWER)
        expect(side.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)
        return side, question


def _account_chats(page: Page) -> list[dict]:
    """The chats of the page's account, as its sidebar lists them."""
    token = page.evaluate("localStorage.token")
    listed = page.request.get(
        loopback_url(page.url, "/api/v1/chats/?include_pinned=true&include_folders=true"),
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listed.ok, f"the chat list answered {listed.status}: {listed.text()}"
    return listed.json()


def _title_of(actor, chat_id: str) -> str:
    with actor.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    assert found.status_code == 200, found.text
    return found.json()["title"]


def _asked_aside(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    return opener.asked_aside(actor, context_options)[0]


def _opened_aside(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    return open_side_chat(opener.beside_a_chat(actor, context_options), place=opener.place)


def _tab_beside_a_chat(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    return open_new_chat_tab(opener.beside_a_chat(actor, context_options), opener.place)


def _pinned_model_tab(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    with actor.client() as client:
        pinned = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"pinnedModels": [MOCK_MODEL_ID]}}
        )
    assert pinned.status_code == 200, pinned.text
    page = opener.beside_a_chat(actor, context_options)
    return open_pinned_model_tab(page, MOCK_MODEL_ID, opener.place)


def _dragged_tab(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    if chat_id:
        title = _title_of(actor, chat_id)
    else:
        title = f"Packing for the Grossglockner {uuid.uuid4().hex[:8]}"
        with actor.client() as client:
            chat_id, _ = seed_chat(client, DRAGGED_CHAT)
            renamed = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
        assert renamed.status_code == 200, renamed.text
    return drop_chat_beside(opener.beside_a_chat(actor, context_options), title, opener.place)


def _temporary_side(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    side = open_side_chat(opener.beside_a_chat(actor, context_options), place=opener.place)
    side.input.click()
    side.page.keyboard.type("/temporary")
    side.page.get_by_role("button", name=re.compile(r"^Temporary")).click()
    expect(side.input).to_have_text("")
    return side


def _after_reload(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    side, title = opener.asked_aside(actor, context_options)
    return reload_beside_the_chat(side, title)


def _after_switching(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    side, title = opener.asked_aside(actor, context_options)
    page = side.page
    main = main_chat(page)
    main.show()
    sidebar = open_sidebar(page)
    sidebar.get_by_role("link", name="New Chat").last.click()
    expect(main.replies).to_have_count(0)
    sidebar.get_by_role(
        "button", name=re.compile(rf"^{re.escape(opener.parent_title)}(?: |$)")
    ).click()
    expect(main.last_reply).to_contain_text(PARENT_CHAT[1]["content"], timeout=REPLY_TIMEOUT_MS)
    close_sidebar(page)
    tab = page.get_by_role("tab", name=side_tab_name(title))
    expect(tab).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    tab.click()
    return side_chat_titled(page, title, opener.place, side.beside_chat_id)


def _admin_side(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    with opener.admin.client() as client:
        promoted = client.post(
            f"/api/v1/users/{actor.id}/update",
            json={"role": "admin", "name": actor.name, "email": actor.email},
        )
    assert promoted.status_code == 200, promoted.text
    return open_side_chat(opener.beside_a_chat(actor, context_options), place=opener.place)


def _beside_answering(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    page = opener.beside_a_chat(actor, context_options)
    side = open_side_chat(page, place=opener.place)
    question = f"Tell me about the old pass road ({uuid.uuid4().hex[:8]})"
    release = threading.Event()
    opener.held.append(release)
    opener.upstream.queue(
        reply.text(STILL_ANSWERING, hold=release, hold_after=3, match=reply.answering(question))
    )
    main = main_chat(page)
    main.send(question)
    expect(main.last_reply).to_contain_text(STILL_ANSWERING[0], timeout=REPLY_TIMEOUT_MS)
    return side


def _second_side(opener: ChatOpener, actor, chat_id, context_options) -> ChatSurface:
    page = opener.beside_a_chat(actor, context_options)
    open_side_chat(page, place=opener.place)
    return open_side_chat(page, place=opener.place)


_CONDITIONS = {
    ASKED_ASIDE: _asked_aside,
    OPENED_ASIDE: _opened_aside,
    TAB_BESIDE_A_CHAT: _tab_beside_a_chat,
    PINNED_MODEL_TAB: _pinned_model_tab,
    DRAGGED_TAB: _dragged_tab,
    TEMPORARY_SIDE: _temporary_side,
    AFTER_RELOAD: _after_reload,
    AFTER_SWITCHING: _after_switching,
    ADMIN_SIDE: _admin_side,
    BESIDE_ANSWERING: _beside_answering,
    SECOND_SIDE: _second_side,
}
