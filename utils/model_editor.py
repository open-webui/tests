"""Editing a model in the workspace's model editor and chatting on it, the way an admin does.

`open_editor` opens the editor of a model by id and waits for its name, `set_checkbox` ticks or
unticks a box under one of the editor's titled parts (Capabilities, Default Features, Builtin
Tools) and `save` saves and waits for the models list. `open_chat_on` starts a new chat on the
model, `open_menu` opens one of the chat input's menus and `sent_request` sends a question and
returns the request the scripted provider got for it.
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
    """The innermost part of the editor under `title` that holds checkboxes."""
    titled = editor.locator("div").filter(has=editor.page.get_by_text(title, exact=True))
    return titled.filter(has=editor.page.get_by_role("checkbox")).last


def set_checkbox(editor: Locator, title: str, label: str, checked: bool) -> None:
    checkbox = section(editor, title).get_by_role("checkbox", name=label, exact=True)
    if (checkbox.get_attribute("aria-checked") == "true") != checked:
        checkbox.click()
    expect(checkbox).to_have_attribute("aria-checked", str(checked).lower())


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
