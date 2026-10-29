"""Journey: a person selects text in a terminal file and asks about it from the box that opens.

A real Open Terminal runs with a Python file in its home. In a chat with the terminal picked the
person opens the file in the File browser, selects one of its lines and the "Ask or edit…" box
opens under the selection. What they type there is sent as a chat message: the model gets the
selected line quoted with "> " above the instruction, and the stored message carries the file
with its path in the terminal. Asking while a reply is still running queues the request, which
is sent once that reply ends. Asking about a file with unsaved edits first offers Save and send
or Cancel: Cancel sends nothing and leaves the file on disk alone, Save and send writes the edit
to the file and then sends.

Discriminates: passes on dev 3b8205505; on a build where the file browser's ask box drops the
quote, where asking during a reply sends at once instead of queueing, and where the unsaved-edit
prompt is skipped, the matching tests here fail.
"""

from __future__ import annotations

import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal
from utils.chat_ui import expect_reply, send, stop_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_URL = re.compile(r"/c/([0-9a-f-]{36})$")
PLAN = "harvest = 12\nrack_the_perry()\nbottle_the_cider()\n"
SELECTED = "rack_the_perry()"


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def browser(person, page_for, admin, preserve, open_terminal):
    """The File browser region of a chat with the Open Terminal picked."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, connection["name"])
    return page.get_by_role("region", name="File browser")


@pytest.fixture
def plan(open_terminal, browser):
    """A Python file of the test's own in the terminal's home, opened in the File browser."""
    path = open_terminal.home / f"plan-{uuid.uuid4().hex[:8]}.py"
    path.write_text(PLAN)
    browser.get_by_label("Refresh", exact=True).first.click()
    browser.get_by_text(path.name, exact=True).click()
    expect(browser.get_by_text(SELECTED, exact=True)).to_be_visible()
    return path


def select_line(browser: Locator, text: str) -> None:
    browser.get_by_text(text, exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.press("Shift+Home")


def ask_box(page: Page) -> Locator:
    return page.get_by_role("form", name="Ask about this file")


def ask(page: Page, instruction: str) -> None:
    box = ask_box(page)
    expect(box).to_be_visible()
    box.get_by_label("Ask or edit…").fill(instruction)
    box.get_by_label("Ask or edit…").press("Enter")


def requests_asking(upstream, instruction: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(instruction)(body)]


def last_user_message(body: dict) -> str:
    return [entry for entry in body["messages"] if entry["role"] == "user"][-1]["content"]


def stored_user_files(person, page: Page) -> list[dict]:
    expect(page).to_have_url(CHAT_URL)
    chat_id = CHAT_URL.search(page.url).group(1)
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    messages = stored.json()["chat"]["history"]["messages"].values()
    return [file for message in messages if message["role"] == "user" for file in message["files"]]


def test_asking_about_a_selection_sends_it_quoted_with_the_file(browser, plan, person, upstream):
    page = browser.page
    instruction = "Rename this call to rack_the_cider"
    upstream.queue(reply.text("Renamed it.", match=reply.answering(instruction)))

    select_line(browser, SELECTED)
    expect(ask_box(page).get_by_text(SELECTED)).to_be_visible()
    ask(page, instruction)

    expect_reply(page, "Renamed it.")
    sent = last_user_message(requests_asking(upstream, instruction)[-1])
    assert sent.endswith(f"\n> {SELECTED}\n\n{instruction}"), (
        f"the model should get the selected line quoted above the instruction: {sent!r}"
    )
    files = stored_user_files(person, page)
    assert [(file["type"], file["path"]) for file in files] == [("filesystem", str(plan))], (
        f"the message should carry the terminal file with its path: {files}"
    )


def test_asking_while_a_reply_runs_sends_after_it_ends(browser, plan, upstream):
    page = browser.page
    instruction = "Explain what racking does"
    release = threading.Event()
    upstream.queue(
        reply.text("Tally begun.", match=reply.answering("Start the tally"), hold=release),
        reply.text("Racking moves the perry.", match=reply.answering(instruction)),
    )
    send(page, "Start the tally")
    expect(stop_button(page)).to_be_visible()

    select_line(browser, SELECTED)
    ask(page, instruction)

    queued = (
        page.get_by_role("main")
        .locator("div")
        .filter(has=page.get_by_role("button", name="Send now"))
    )
    expect(queued.filter(has_text=instruction).last).to_be_visible()
    assert requests_asking(upstream, instruction) == [], "the request went out during the reply"
    release.set()
    expect_reply(page, "Racking moves the perry.")
    expect(queued).to_have_count(0)
    asked = requests_asking(upstream, instruction)
    assert len(asked) == 1, f"the queued request should be sent once: {len(asked)}"
    replies_seen = [
        entry["content"] for entry in asked[0]["messages"] if entry["role"] == "assistant"
    ]
    assert "Tally begun." in replies_seen, (
        f"the queued request should follow the finished reply: {replies_seen}"
    )


def test_unsaved_edits_are_kept_unsent_on_cancel_and_saved_on_save_and_send(
    browser, plan, upstream
):
    page = browser.page
    instruction = "Check the harvest count"
    upstream.queue(reply.text("Thirteen it is.", match=reply.answering(instruction)))
    browser.get_by_text("harvest = 12", exact=True).click()
    page.keyboard.press("End")
    page.keyboard.press("Backspace")
    page.keyboard.type("3")
    expect(browser.get_by_text("harvest = 13", exact=True)).to_be_visible()

    select_line(browser, SELECTED)
    ask(page, instruction)
    prompt = page.get_by_role("dialog", name="Unsaved changes")
    expect(prompt.get_by_role("button", name="Save and send")).to_be_visible()
    expect(prompt.get_by_role("button", name="Cancel")).to_be_visible()
    expect(prompt.get_by_role("button", name="Discard changes")).to_have_count(0)
    prompt.get_by_role("button", name="Cancel").click()

    expect(prompt).to_have_count(0)
    expect(page.get_by_text("File request cancelled.")).to_be_visible()
    expect(browser.get_by_text("harvest = 13", exact=True)).to_be_visible()
    assert plan.read_text() == PLAN, "Cancel should leave the file on disk alone"
    assert requests_asking(upstream, instruction) == [], "Cancel should send nothing"

    ask_box(page).get_by_role("button", name="Send").click()
    prompt.get_by_role("button", name="Save and send").click()

    expect_reply(page, "Thirteen it is.")
    assert plan.read_text() == PLAN.replace("12", "13"), "Save and send should save the edit"
    asked = requests_asking(upstream, instruction)
    assert len(asked) == 1, f"Save and send should send the request once: {len(asked)}"
