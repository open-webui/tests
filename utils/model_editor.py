"""Editing a model in the workspace's model editor and chatting on it, the way an admin does.

`open_editor` opens the editor of a model by id and waits for its name, `section` opens one of
the editor's collapsible parts (Capabilities, Default Features, Builtin Tools, Prompts, Voice),
`set_checkbox` turns a switch in one of them on or off, `pick` ticks or unticks an entry in the
search list of Tools, Skills, Filters or Actions and `save` saves and waits for the models list.
`open_chat_on` starts a new chat on the model, `open_menu` opens one of the chat input's menus and
`sent_request` sends a question and returns the request the scripted provider got for it.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send


def open_editor(page: Page, model: dict) -> Locator:
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def save(editor: Locator) -> None:
    editor.get_by_role("button", name="Save & Update").click()
    expect(editor.page).to_have_url(re.compile(r"/workspace/models/?$"))


def section(editor: Locator, title: str) -> Locator:
    """The editor's collapsible part titled `title`, opened."""
    title_text = editor.page.locator("summary > span:first-child").get_by_text(title, exact=True)
    part = editor.locator("details").filter(has=title_text)
    if part.get_attribute("open") is None:
        part.locator("summary").first.click()
    expect(part).to_have_attribute("open", "")
    return part


def set_checkbox(editor: Locator, title: str, label: str, checked: bool) -> None:
    switch = section(editor, title).get_by_role("switch", name=label, exact=True)
    if (switch.get_attribute("aria-checked") == "true") != checked:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", str(checked).lower())


def pick(editor: Locator, label: str, name: str, ticked: bool = True) -> None:
    """Tick or untick `name` in the search list behind the editor's `label` row (Tools, Skills,
    Filters, Actions)."""
    trigger = editor.get_by_role("button", name=label, exact=True)
    trigger = trigger.and_(editor.locator("button[aria-expanded]"))
    trigger.click()
    editor.page.get_by_placeholder(f"Search {label.lower()}").fill(name)
    item = editor.page.get_by_role("button", name=name, exact=True)
    if (item.get_attribute("aria-pressed") == "true") != ticked:
        item.click()
    expect(item).to_have_attribute("aria-pressed", str(ticked).lower())
    editor.page.get_by_role("button", name="Done").click()
    expect(editor.page.get_by_placeholder(f"Search {label.lower()}")).to_have_count(0)


def open_chat_on(page: Page, model: dict) -> None:
    page.goto(f"/?model={model['id']}")
    expect(page.get_by_role("button", name=f"Selected model: {model['name']}")).to_be_visible()
    expect(chat_input(page)).to_be_visible()


def open_menu(page: Page, button_name: str) -> Locator:
    page.get_by_role("button", name=button_name, exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def sent_request(page: Page, upstream, question: str, answer: str = "noted") -> dict:
    """Send `question` in the open chat; the request the provider got for it."""
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, answer)
    return next(filter(reply.answering(question), upstream.chat_requests()))


def offered_tool_names(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}
