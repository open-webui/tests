"""Journey: a long chat in the browser on the Prompt Caching page's setup only appends.

The Prompt Caching docs page promises that with its cache-optimal setup the request Open WebUI
sends to the provider keeps its start: every request repeats the tool list and every message of
the one before it byte for byte and only adds to the end. Its integration twin checks each
feature family over HTTP; here a person drives chats through the web client. One goes through a
file attached in the chat input and searched with the file tools, a reply with reasoning, the
model's knowledge searched, a native tool call, a page reload, the chat continued from a second
browser and a while passing between turns. Another asks about an image attached in the chat
input across a reload. Another has a real Open Terminal picked for the chat from the first turn,
with a file written, a command run and the file read back across a reload. Every consecutive
pair of the provider's requests is checked.

Two terminal tests stay red, on what the page does not name as a breaker, each rewriting the tool
list at the very start of the prefix: opening a folder in the terminal's file browser moves the
working directory written into the run_command tool's description
(open-webui/open-webui#31589), and a reload with the terminal's shell open closes it, which drops
the two user shell tools (open-webui/open-webui#31590, fix PR #31602 open).

Discriminates: passes on dev 176d31d1d apart from those two, which fail there. In backend
copies, a clock value added to the model's system prompt and the tool list shuffled per request
each turned every test red; with no working directory in the tool description and the shell
tools offered whether or not the shell is open, the two terminal tests passed.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.mcp_server import SNAPSHOT_PNG
from harness.prompt_caching import (
    assert_append_only,
    cache_optimal_model,
    prefix_break,
    turn_off_memory_system_context,
)
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTES = ("berths.txt", "Berth 3 is reserved for the pilot boat.\nBerth 5 is free on Mondays.\n")


@pytest.fixture
def cached_setup(admin, preserve):
    """The page's setup: its model, and the memory system context switched off."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin) as model:
        yield model


def attach(page: Page, name: str, content: str | bytes, mime_type: str = "text/plain") -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    buffer = content.encode() if isinstance(content, str) else content
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": buffer})


def ask(page: Page, upstream, prompt: str, *replies: reply.Reply) -> None:
    """Send `prompt` and wait for the last scripted reply's text to show."""
    for scripted in replies:
        scripted.match = reply.answering(prompt)
        upstream.queue(scripted)
    send(page, prompt)
    expect_reply(page, replies[-1].content)


def calling(name: str, arguments: dict, call_id: str) -> reply.Reply:
    return reply.tool_call(name, arguments, call_id=call_id)


def first_break(requests: list[dict]) -> str | None:
    for earlier, later in zip(requests, requests[1:]):
        broken = prefix_break(earlier, later)
        if broken:
            return broken
    return None


def test_a_long_chat_in_the_browser_only_appends(page_for, cached_setup, make_user, upstream):
    person = make_user()
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    attach(page, *NOTES)
    ask(
        page,
        upstream,
        "which berth is free?",
        calling("query_chat_files", {"query": "free berth"}, "call_files"),
        reply.text("Berth 5 is free (berths.txt)."),
    )
    ask(
        page,
        upstream,
        "and for the pilot boat?",
        reply.text("Berth 3, as the file says.", reasoning="the file names berth 3"),
    )
    ask(
        page,
        upstream,
        "when does the ferry leave?",
        calling("query_knowledge_files", {"query": "ferry"}, "call_knowledge"),
        reply.text("At 06:40 (handbook.txt)."),
    )
    chat_url = page.url

    page.reload()
    ask(page, upstream, "is that every day?", reply.text("Every day, the handbook says."))

    second = page_for(person)
    second.goto(chat_url)
    expect(second.get_by_text("Every day, the handbook says.")).to_be_visible()
    time.sleep(1.5)  # a clock value in the prompt would move on between the turns
    ask(
        second,
        upstream,
        "what time is it now?",
        calling("get_current_timestamp", {}, "call_clock"),
        reply.text("Just after six."),
    )
    ask(second, upstream, "thanks, that is all", reply.text("Safe travels."))

    requests = [body for body in upstream.chat_requests() if body.get("stream")]
    assert len(requests) == 9, f"expected nine requests to the provider, got {len(requests)}"
    tool_results = "\n".join(
        str(entry["content"]) for entry in requests[-1]["messages"] if entry["role"] == "tool"
    )
    assert "Berth 5 is free" in tool_results and "06:40" in tool_results, tool_results
    assert "<attached_files>" in str(requests[0]["messages"][1]["content"])
    assert_append_only(requests)


def test_a_chat_about_an_attached_image_only_appends(page_for, cached_setup, make_user, upstream):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    attach(page, "harbour.png", SNAPSHOT_PNG, "image/png")
    ask(page, upstream, "what is in this picture?", reply.text("A calm harbour."))
    ask(page, upstream, "any boats?", reply.text("Two boats."))
    page.reload()
    ask(page, upstream, "thanks", reply.text("Enjoy the view."))

    requests = [body for body in upstream.chat_requests() if body.get("stream")]
    first_question = requests[0]["messages"][1]["content"]
    assert any(part.get("type") == "image_url" for part in first_question), first_question
    assert_append_only(requests)


def pick_terminal(page: Page, name: str) -> None:
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("button", name=name).click()
    expect(page.get_by_role("region", name="File browser")).to_be_visible()


@pytest.fixture
def terminal_chat(page_for, cached_setup, admin, make_user, preserve, open_terminal):
    """A page on the cached model with the Open Terminal picked for the chat, and its home."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    (open_terminal.home / "tickets").mkdir(exist_ok=True)
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")
    pick_terminal(page, connection["name"])
    return page


def test_a_chat_with_an_open_terminal_only_appends(terminal_chat, open_terminal, upstream):
    page = terminal_chat
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(
        page,
        upstream,
        "note the ferry time",
        calling("write_file", {"path": "ferry.txt", "content": "06:40 from pier 7\n"}, "call_w"),
        reply.text("Written."),
    )
    ask(
        page,
        upstream,
        "list my files",
        calling("run_command", {"command": "ls && cat ferry.txt", "wait": 10}, "call_ls"),
        reply.text("ferry.txt is there."),
    )
    page.reload()
    ask(
        page,
        upstream,
        "read it back",
        calling("read_file", {"path": "ferry.txt"}, "call_read"),
        reply.text("06:40 from pier 7."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = [body for body in upstream.chat_requests() if body.get("stream")]
    assert {"run_command", "write_file", "read_file"} <= {
        tool["function"]["name"] for tool in requests[0]["tools"]
    }
    assert "You have access to a computer" in requests[0]["messages"][0]["content"]
    assert (open_terminal.home / "ferry.txt").read_text() == "06:40 from pier 7\n"
    assert_append_only(requests)


@pytest.mark.regression
def test_opening_a_folder_in_the_terminal_file_browser_keeps_the_prefix(terminal_chat, upstream):
    page = terminal_chat
    ask(page, upstream, "good morning", reply.text("Morning."))
    with page.expect_request(lambda sent: sent.method == "POST" and "/files/cwd" in sent.url):
        page.get_by_role("region", name="File browser").get_by_text("tickets").first.click()
    ask(page, upstream, "anything new?", reply.text("Nothing new."))

    requests = [body for body in upstream.chat_requests() if body.get("stream")]
    broken = first_break(requests)
    assert broken is None, (
        "#31589: opening a folder in the terminal's file browser rewrote the run_command tool "
        f"definition, which carries the folder, at the start of the cached prefix: {broken}"
    )


def test_a_reload_with_the_terminal_shell_open_keeps_the_prefix(terminal_chat, upstream):
    page = terminal_chat
    browser = page.get_by_role("region", name="File browser")
    browser.get_by_role("button", name="Expand terminal").click()
    expect(browser.get_by_role("tab", name="Shell")).to_be_visible()
    ask(page, upstream, "good morning", reply.text("Morning."))
    page.reload()
    ask(page, upstream, "anything new?", reply.text("Nothing new."))

    requests = [body for body in upstream.chat_requests() if body.get("stream")]
    broken = first_break(requests)
    assert broken is None, (
        "the reload closed the terminal's shell, which dropped the two user shell tools from "
        f"the tool list at the start of the cached prefix: {broken}"
    )
