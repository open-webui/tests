"""Driving the chat page the way a person does: type, press Enter, read what appears.

Every helper that acts on the chat takes the page, meaning the page's main chat, or a
`ChatSurface` from `utils/chat_surface.py`, meaning that chat: the main one or one in a side tab.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from harness.plain_http import loopback_url
from utils.chat_surface import REPLY_TIMEOUT_MS as REPLY_TIMEOUT_MS
from utils.chat_surface import ChatSurface

Chat = Page | ChatSurface


def _page(chat: Chat) -> Page:
    return chat.page if isinstance(chat, ChatSurface) else chat


def chat_scope(chat: Chat) -> Page | Locator:
    """What a chat's own elements are looked for in: its panel, or the page for a main chat."""
    return chat.panel if isinstance(chat, ChatSurface) else chat


def chat_input(chat: Chat) -> Locator:
    if isinstance(chat, ChatSurface):
        return chat.input
    return chat.locator("#chat-input")


def send(chat: Chat, text: str) -> None:
    page = _page(chat)
    expect(chat_input(chat)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    chat_input(chat).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def link_dialog(page: Page) -> Locator:
    """The dialog a chat link asks in before it loads its `load-url` or `youtube` page or calls."""
    return page.get_by_role("dialog", name="Open link")


def conversation(chat: Chat) -> Locator:
    """The open chat's messages, never the preview a hovered sidebar chat shows."""
    if isinstance(chat, ChatSurface):
        return chat.conversation
    return chat.get_by_role("main").get_by_label("Chat Conversation")


def replies(chat: Chat) -> Locator:
    return conversation(chat).locator(".chat-assistant")


def last_reply(chat: Chat) -> Locator:
    return replies(chat).last


def listed_source(chat: Chat, name: str) -> Locator:
    """An entry of an opened source list, which reads out its number before the name."""
    label = re.compile(rf"^View source: \d+ {re.escape(name)}$")
    return conversation(chat).get_by_role("button", name=label)


def expect_reply(chat: Chat, text: str) -> None:
    expect(last_reply(chat)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)


def side_chat_tab(page: Page, question: str) -> Locator:
    """The tab of the side chat opened with `question`, which it takes as its title once saved."""
    return page.get_by_role("tab", name=question, exact=True)


def side_chat_panel(page: Page, question: str) -> Locator:
    return page.get_by_role("tabpanel", name=question, exact=True)


def chat_tab_bar(page: Page) -> Locator:
    """The bar over the open chat: its tabs, the "Chat actions" menu and who else has it open."""
    # the bar has no role or name of its own, only the tab list inside it
    return (
        page.get_by_role("main")
        .locator("[data-panel-tab-bar]")
        .filter(has=page.get_by_role("tablist", name="Tabs"))
    )


def split_screen(page: Page) -> None:
    """Show the conversation beside the other tabs. Since 617618c9e the tabs fill the window, so
    a tab opened by the chat (artifacts, a call) covers the conversation; with no other tab yet,
    Split screen opens the first one the chat offers."""
    button = page.get_by_role("button", name="Split screen", exact=True)
    button.click()
    expect(button).to_have_attribute("aria-pressed", "true")
    expect(chat_input(page)).to_be_visible()


def expect_split_screen(chat: Chat) -> None:
    """Wait until the conversation shows beside the other tabs, as it does once the chat opens
    the artifacts pane (since 0b6d43405, open-webui/open-webui#33878)."""
    button = _page(chat).get_by_role("button", name="Split screen", exact=True)
    expect(button).to_have_attribute("aria-pressed", "true")
    expect(chat_input(chat)).to_be_visible()


def stop_button(chat: Chat) -> Locator:
    if isinstance(chat, ChatSurface):
        return chat.stop_button
    return chat.get_by_role("button", name="Stop")


def regenerate_buttons(chat: Chat) -> Locator:
    """The Regenerate button under each finished reply."""
    return conversation(chat).get_by_label("Regenerate", exact=True)


def typing_cursor(reply: Locator) -> Locator:
    """The blinking cursor a reply shows while it is still being written."""
    return reply.locator(".animate-cursor-pulse")


def close_menu(chat: Chat) -> None:
    """Close the open menu and wait until it is gone, so the next click or key reaches the page."""
    page = _page(chat)
    menu = page.get_by_role("menu")
    # a tooltip the pointer left open over the menu takes the first Escape for itself
    for _ in range(3):
        page.keyboard.press("Escape")
        try:
            menu.first.wait_for(state="detached", timeout=2_000)
            break
        except PlaywrightTimeoutError:
            continue
    expect(menu).to_have_count(0)


def menu_entries(menu: Locator) -> Locator:
    """Every entry of a chat input menu: plain items and the ones that switch on and off."""
    return (
        menu.get_by_role("menuitem")
        .or_(menu.get_by_role("menuitemcheckbox"))
        .or_(menu.get_by_role("menuitemradio"))
    )


def _open_composer_menu(chat: Chat, button: str) -> Locator:
    page = _page(chat)
    expect(chat_input(chat)).to_be_visible()
    if isinstance(chat, ChatSurface):
        chat.button(button).last.click()
    else:
        page.get_by_role("button", name=button, exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu_entries(menu).first).to_be_visible()
    return menu


def open_input_menu(chat: Chat) -> Locator:
    """The chat input's More menu (uploads, attachments, tool permissions), once it shows."""
    return _open_composer_menu(chat, "More")


def open_integrations_menu(chat: Chat) -> Locator:
    """The chat input's integrations (tools, skills, filters, web search), once they show: since
    dev 834c47ee0 they sit in the More menu, under its own entries."""
    return _open_composer_menu(chat, "More")


def skill_group_of(entry: Locator) -> Locator:
    """The heading a skill of the integrations menu's Skills list is listed under: since dev
    cdbe9788c the list is grouped by source, a terminal's skills under the terminal's name."""
    return entry.locator("xpath=preceding-sibling::div[1]")


def expect_picked(chat: Chat, entry: str, count: int) -> None:
    """Check that the integrations' `entry` row (Tools, Skills) counts `count` picks, then close."""
    row = open_integrations_menu(chat).get_by_role("menuitem", name=re.compile(rf"^{entry}"))
    expect(row).to_contain_text(f"{count} selected")
    close_menu(chat)


def show_input_selections(actor) -> None:
    """Turn on the account's Show Input Selections, which puts the picked tools, skills and
    switches beside the + button (off by default since dev 834c47ee0)."""
    with actor.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"showInputSelections": True}}
        )
    assert saved.status_code == 200, saved.text


def integration_row(menu: Locator, name: str, role: str = "menuitemcheckbox") -> Locator:
    """The Integrations menu row of `name`: the innermost block holding its entry and the buttons
    beside it (Valves, Disconnect OAuth)."""
    entry = menu.page.get_by_role(role, name=name)
    return menu.locator("div").filter(has=entry).last


def set_tool_approval_mode(chat: Chat, mode: str) -> None:
    """Pick how this chat's tool calls are approved, under the More menu's Tool Permissions."""
    menu = open_input_menu(chat)
    permissions = menu.get_by_role("menuitem", name="Tool Permissions")
    permissions.click()
    menu.get_by_role("menuitemradio", name=mode).click()
    expect(permissions).to_contain_text(mode)
    close_menu(chat)


def turn_on_integration(chat: Chat, name: str) -> None:
    """Switch `name` on for the next message from the chat's Integrations menu."""
    switch = open_integrations_menu(chat).get_by_role("menuitemcheckbox", name=name, exact=True)
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    close_menu(chat)


def open_add_context(chat: Chat) -> Locator:
    """The More menu's Add Context list (webpages, files, knowledge, notes, chats), once it shows:
    since dev 834c47ee0 they sit there, a level down from the menu's first page."""
    menu = open_input_menu(chat)
    menu.get_by_role("menuitem", name="Add Context").click()
    expect(menu.get_by_role("menuitem", name="Upload Files")).to_have_count(0)
    return menu


def _get(page: Page, path: str):
    """What `path` answers the page's account."""
    token = page.evaluate("localStorage.token")
    answered = page.request.get(
        loopback_url(page.url, path), headers={"Authorization": f"Bearer {token}"}
    )
    assert answered.ok, f"{path} answered {answered.status}: {answered.text()}"
    return answered.json()


def _names_by_id(page: Page, path: str) -> dict[str, str]:
    """The name of each model or tool the account is offered at `path`, by id."""
    entries = _get(page, path)
    if isinstance(entries, dict):
        entries = entries["data"]
    return {entry["id"]: entry["name"] for entry in entries}


def _pick_model(chat: ChatSurface, name: str) -> None:
    page = chat.page
    chat.button(re.compile("^(Select a model|Selected model)")).last.click()
    page.get_by_role("combobox", name="Search In Models").fill(name)
    page.get_by_role("listbox", name="Available models").get_by_role(
        "option", name=f"Select {name} model", exact=True
    ).click()
    expect(chat.button(f"Selected model: {name}")).to_be_visible()


def _pick_tool(chat: ChatSurface, name: str) -> None:
    menu = open_input_menu(chat)
    menu.get_by_role("menuitem", name=re.compile(r"^Tools")).click()
    tool = menu.get_by_role("menuitemcheckbox", name=name, exact=True)
    if tool.get_attribute("aria-checked") != "true":
        tool.click()
    expect(tool).to_have_attribute("aria-checked", "true")
    close_menu(chat)


def start_from_link(chat: ChatSurface, link: str) -> None:
    """Start the chat the way a link such as `/?models=<id>&tools=<id>` does.

    The main chat opens the link. A side chat has no address of its own, so the model is picked
    in its model selector and each tool in its More menu."""
    if not chat.is_side:
        chat.page.goto(link)
        return
    asked = parse_qs(urlsplit(link).query)
    model_ids = asked.pop("models", []) + asked.pop("model", [])
    tool_ids = [tool_id for value in asked.pop("tools", []) for tool_id in value.split(",")]
    assert not asked, f"a side chat cannot be started with {sorted(asked)} from a link"
    if model_ids:
        (model_id,) = model_ids
        _pick_model(chat, _names_by_id(chat.page, "/api/models")[model_id])
    if tool_ids:
        tool_names = _names_by_id(chat.page, "/api/v1/tools/")
        for tool_id in tool_ids:
            _pick_tool(chat, tool_names[tool_id])


def _compare_with(chat: ChatSurface, current: str, names: list[str]) -> None:
    """Add the models `names` beside `current` through the chat's Compare."""
    if not names:
        return
    chat.button(f"Selected model: {current}").click()
    chat.page.get_by_role("button", name="Compare").click()
    for name in names:
        chat.page.get_by_role("combobox", name="Search In Models").fill(name)
        chat.page.get_by_role("option", name=f"Select {name} model", exact=True).click()
    chat.page.keyboard.press("Escape")


def start_on_default_models(chat: ChatSurface) -> None:
    """Have a new chat ask the account's default models, as a new main chat does.

    A side chat starts on the model of the chat it was opened from, so it picks them in its own
    model selector: the first, then the others through Compare."""
    model_ids = _get(chat.page, "/api/v1/users/user/settings")["ui"]["models"]
    names = _names_by_id(chat.page, "/api/models")
    first, *others = (names[model_id] for model_id in model_ids)
    if chat.is_side:
        _pick_model(chat, first)
        _compare_with(chat, first, others)
    shown = f"Selected model: {first} +{len(others)}" if others else f"Selected model: {first}"
    expect(chat.button(shown)).to_be_visible()


def turn_temporary(chat: Chat) -> None:
    """Make a chat without messages temporary: the main chat with the Temporary Chat toggle in
    its header, a side chat, which has no header of its own, with `/temporary`."""
    page = _page(chat)
    if not isinstance(chat, ChatSurface) or not chat.is_side:
        page.get_by_role("button", name="Temporary Chat").click()
        expect(page.locator("#chat-pane").get_by_text("Temporary Chat", exact=True)).to_be_visible()
        return
    chat.input.click()
    page.keyboard.type("/temporary")
    # the / menu's entry; a new main chat beside has a Temporary Chat toggle as well
    page.get_by_role("tooltip").get_by_role("button", name=re.compile(r"^Temporary")).click()
    expect(chat.input).to_have_text("")
