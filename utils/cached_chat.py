"""Chatting in the browser on the Prompt Caching page's model, and reading what the provider got.

`ask` sends a prompt and waits for the last of its scripted replies, each tied to that prompt, so
a tool round and the answer after it are one call. `attach` uploads a file through the chat
input's menu and `attach_from_menu` attaches a knowledge base, note or chat from it.
`pick_terminal` picks an Open Terminal for the chat before the first message, and `chat_requests`
is every streamed request the provider received, the ones the cache sees.
"""

from __future__ import annotations

import itertools
import re

from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.upstream import Reply
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button

CALL_NUMBERS = itertools.count(1)


def attach(page: Page, name: str, content: str | bytes, mime_type: str = "text/plain") -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    buffer = content.encode() if isinstance(content, str) else content
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": buffer})


def attach_from_menu(page: Page, submenu: str, name: str) -> None:
    """Attach a knowledge base, note or chat through the chat input's More menu."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button").first).to_be_visible()
    if menu.get_by_role("button", name=submenu).count() == 0:
        menu.get_by_role("button").first.click()  # it reopens on the list last picked from
    menu.get_by_role("button", name=submenu).click()
    menu.get_by_role("button", name=name).first.click()
    expect(page.locator("form").get_by_text(name).first).to_be_visible()


def ask(page: Page, upstream, prompt: str, *replies: Reply) -> None:
    """Send `prompt` and wait for the last scripted reply's text to show."""
    for scripted in replies:
        scripted.match = reply.answering(prompt)
        upstream.queue(scripted)
    send(page, prompt)
    expect_reply(page, replies[-1].content)


def calling(name: str, arguments: dict, **options) -> Reply:
    return reply.tool_call(name, arguments, call_id=f"call_{next(CALL_NUMBERS)}", **options)


def calling_together(*calls: tuple[str, dict]) -> Reply:
    return Reply(tool_calls=[calling(name, arguments).tool_calls[0] for name, arguments in calls])


def pick_terminal(page: Page, name: str) -> None:
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("button", name=name).click()
    expect(page.get_by_role("region", name="File browser")).to_be_visible()


def chat_requests(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if body.get("stream")]


def offered_tools(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def called_tools(request: dict) -> set[str]:
    """Every tool the model called in the conversation `request` carries."""
    return {
        call["function"]["name"]
        for message in request["messages"]
        for call in message.get("tool_calls") or []
    }


def tool_results(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "tool"
    )


def turn_on_tool(page: Page, tool_name: str) -> None:
    """Switch a workspace tool on for this chat from the Integrations menu."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    page.get_by_role("button", name=tool_name).click()
    page.keyboard.press("Escape")
