"""Editing a model in the workspace's model editor and chatting on it, the way an admin does.

`open_editor` opens the editor of a model by id and waits for its name, `section` opens one of
the editor's collapsible parts (Features, Builtin Tools, Prompts, Voice), `set_checkbox` turns a
switch in one of them on or off, `default_features` holds the switches under the Features caption
"Start enabled in new chats" and `set_default_feature` turns one of them on or off, `pick` ticks or
unticks an entry in the search list of Tools, Skills, Filters or Actions and `save` saves and
waits for the models list.
`model_default_terminal` is the Default machine row of the editor or of Model Defaults,
`expect_model_default_terminal` waits until it shows a terminal and
`choose_model_default_terminal` picks one in its list.
`open_chat_on` starts a new chat on the model, `open_menu` opens one of the chat input's menus and
`sent_request` sends a question and returns the request the scripted provider got for it.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_surface import ChatSurface
from utils.chat_ui import Chat, chat_input, expect_reply, send, start_from_link

DEFAULT_FEATURES_CAPTION = re.compile(r"^\s*Start enabled in new chats\s*$")


def open_editor(page: Page, model: dict) -> Locator:
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def save(editor: Locator) -> None:
    editor.get_by_role("button", name="Save & Update").click()
    expect(editor.page).to_have_url(re.compile(r"/workspace/models/?$"))


def model_default_terminal(editor: Locator) -> Locator:
    """The default machine, its label followed by the machine it shows: the Default machine row
    of Model Defaults, or the title of the model editor's Machine section, which holds a Default
    row."""
    defaults_row = editor.get_by_role("button", name=re.compile(r"^Default machine"))
    title = editor.page.locator("summary > span:first-child").get_by_text("Machine", exact=True)
    editor_section = editor.locator("details").filter(has=title).locator("summary")
    return defaults_row.or_(editor_section)


def expect_model_default_terminal(editor: Locator, name: str) -> None:
    expect(model_default_terminal(editor)).to_have_text(
        re.compile(rf"^\s*(Default machine|Machine)\W*{re.escape(name)}\s*$")
    )


def choose_model_default_terminal(editor: Locator, name: str) -> None:
    row = model_default_terminal(editor)
    if row.evaluate("element => element.tagName") == "SUMMARY":
        row = section(editor, "Machine").get_by_role("button", name=re.compile(r"^Default\b"))
    row.click()
    editor.page.get_by_role("menuitem", name=name, exact=True).click()
    expect_model_default_terminal(editor, name)


def section(editor: Locator, title: str) -> Locator:
    """The editor's collapsible part titled `title`, opened."""
    title_text = editor.page.locator("summary > span:first-child").get_by_text(title, exact=True)
    part = editor.locator("details").filter(has=title_text)
    if part.get_attribute("open") is None:
        part.locator("summary").first.click()
    expect(part).to_have_attribute("open", "")
    return part


def switches(editor: Locator, title: str) -> Locator:
    """Where the switches of the section `title` sit; in Features only the capability switches,
    as its default features below them carry the same names."""
    part = section(editor, title)
    if title != "Features":
        return part
    return part.locator("xpath=./div/div[not(preceding-sibling::p)]")


def default_features(editor: Locator) -> Locator:
    """The switches under the Features caption "Start enabled in new chats"."""
    caption = section(editor, "Features").locator("p").filter(has_text=DEFAULT_FEATURES_CAPTION)
    return caption.locator("xpath=following-sibling::div[1]")


def _set_switch(switch: Locator, checked: bool) -> None:
    if (switch.get_attribute("aria-checked") == "true") != checked:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", str(checked).lower())


def set_checkbox(editor: Locator, title: str, label: str, checked: bool) -> None:
    _set_switch(switches(editor, title).get_by_role("switch", name=label, exact=True), checked)


def set_default_feature(editor: Locator, label: str, checked: bool) -> None:
    _set_switch(default_features(editor).get_by_role("switch", name=label, exact=True), checked)


def selector_row(editor: Locator, label: str) -> Locator:
    """The editor's `label` row (Tools, Skills, Filters, Actions) that opens its search list."""
    # the row is named by its label followed by what it holds ("Tools None", "Filters a, b")
    trigger = editor.get_by_role("button", name=re.compile(rf"^{label}\b"))
    return trigger.and_(editor.locator("button[aria-expanded]"))


def pick(editor: Locator, label: str, name: str, ticked: bool = True) -> None:
    """Tick or untick `name` in the search list behind the editor's `label` row (Tools, Skills,
    Filters, Actions)."""
    selector_row(editor, label).click()
    editor.page.get_by_placeholder(f"Search {label.lower()}").fill(name)
    # an entry reads out its description after its name since e6beee356
    item = editor.page.get_by_role("menuitem").filter(has=editor.page.get_by_text(name, exact=True))
    if (item.get_attribute("aria-pressed") == "true") != ticked:
        item.click()
    expect(item).to_have_attribute("aria-pressed", str(ticked).lower())
    editor.page.get_by_role("menuitem", name="Done").click()
    expect(editor.page.get_by_placeholder(f"Search {label.lower()}")).to_have_count(0)


def open_chat_on(chat: Chat, model: dict) -> None:
    """Start a new chat on the model: the page's chat opens the link, a side chat picks it."""
    selected = f"Selected model: {model['name']}"
    if isinstance(chat, ChatSurface):
        start_from_link(chat, f"/?model={model['id']}")
        expect(chat.button(selected)).to_be_visible()
    else:
        chat.goto(f"/?model={model['id']}")
        expect(chat.get_by_role("button", name=selected)).to_be_visible()
    expect(chat_input(chat)).to_be_visible()


def open_menu(page: Page, button_name: str) -> Locator:
    page.get_by_role("button", name=button_name, exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def sent_request(page: Chat, upstream, question: str, answer: str = "noted") -> dict:
    """Send `question` in the open chat; the request the provider got for it."""
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, answer)
    return next(filter(reply.answering(question), upstream.chat_requests()))


def offered_tool_names(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}
