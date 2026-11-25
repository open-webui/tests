"""Regression: a message sent while a freshly loaded new chat was still setting up lost its reply.

Issue open-webui/open-webui#32819. Loading the page starts a new chat whose setup waits for the
function and skill lists. A first message sent before those arrived was wiped from the screen when
the setup finished, so its reply never showed, the page stayed on the empty new chat and only the
sidebar held the finished chat. The fix stops the setup from resetting a chat that got a message
in the meantime. From b29f7781a to 14892a184 the page also held the input back until the
setup was done.

Since dev 14892a184 the new chat shows its input before the model list arrives and sets itself up
once the list is in. A message sent before that stays in the box, and the setup puts it back after
clearing the input. The test holds the model list from a fresh page load and sends the first
message: it must still be in the box once the setup has picked the model, and sending it then gets
its reply in a saved chat. Retargeted for 14892a184 from the earlier flow, where the message was
sent at once and the setup must not reset it.

Discriminates: passes on dev 9b6241e54; fails on a b393504d9 build with the setup no longer
putting the typed text back (the box is empty after the setup). Before 14892a184 it failed on the
dev ca8f451cc build (the message and its reply vanished) and passed with the fix d1511abce.

A side chat creates its chat when its first message is sent. From dev c0b90785e its first message
showed while that chat was still being created; dev 03d8d3e35 kept it in the box until the chat
existed, and since dev c26c928ce it shows at once again, and a chat that cannot be created takes
the message back into the box to send again. The second test holds the creation, then refuses it:
the message shows while the creation is held, is back in the box after the refusal and gets its
reply when sent again. Retargeted for c26c928ce from the box-only behaviour of 03d8d3e35; passes
on dev 53d741157.

Instant first message (0.13.4): the first message of a new chat shows at once while the chat is
still being created, unless a file is still uploading with it. The third test holds whatever
creates the chat and checks the message is in the conversation and out of the box meanwhile, then
that the reply arrives in a saved chat with its own address, its sidebar entry and both messages
stored. The fourth holds an attached file's upload: the message sent meanwhile stays on screen but
reaches the model only once the upload is done, and the model can read the file. The same check for
a chat in a picked folder is in e2e/tools/test_terminal_working_directory.py.
Discriminates: both pass on the dev f80a4f782 build. On a build of it that creates the chat
before it shows the first message and sends a message with a file still uploading at once, both
fail (the message waits for the held creation; the model is asked before the upload is done).
"""

from __future__ import annotations

import json
import re
import time
import uuid

import pytest
from playwright.sync_api import Page, Route, expect

from harness import upstream as reply
from utils.cached_chat import this_turns_tool_results
from utils.chat_surface import open_new_chat_tab
from utils.chat_ui import chat_input, conversation, expect_reply, open_input_menu, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def hold(page: Page, pattern: str) -> list[Route]:
    held: list[Route] = []
    page.route(pattern, lambda route: held.append(route))
    return held


def wait_until_held(page: Page, held: list[Route], what: str, timeout_ms: int = 15_000) -> None:
    for _ in range(timeout_ms // 100):
        if held:
            return
        page.wait_for_timeout(100)
    raise AssertionError(f"the page never asked for {what}")


MODEL_LIST = re.compile(r"/api/models(\?|$)")


def test_a_first_message_sent_during_new_chat_setup_keeps_its_reply(page_for, make_user, upstream):
    page = page_for(make_user())
    question = f"early question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("early answer", match=reply.answering(question)))

    model_lists = hold(page, MODEL_LIST)
    page.reload()
    wait_until_held(page, model_lists, "the model list")

    # the input is offered before the setup, which waits for the model list; typed here since
    # send() waits for the model the setup picks
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("button", name="Selected model")).to_have_count(0)
    chat_input(page).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect(chat_input(page), "a message sent during setup left the box").to_contain_text(question)
    with page.expect_response(MODEL_LIST):
        for route in model_lists:
            route.continue_()
    page.unroute(MODEL_LIST)
    expect(page.get_by_role("button", name="Selected model")).to_be_visible()
    expect(chat_input(page), "the setup wiped the waiting message").to_contain_text(question)

    chat_input(page).click()
    page.keyboard.press("Enter")
    expect_reply(page, "early answer")
    expect(conversation(page)).to_contain_text(question)
    expect(page).to_have_url(re.compile(r"/c/"))

    # a New Chat started afterwards still opens empty
    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(re.compile(r"/c/"))
    expect(conversation(page).get_by_text(question)).to_have_count(0)


def test_the_first_message_of_a_side_chat_comes_back_to_the_box_when_its_chat_is_refused(
    page_for, make_user, upstream
):
    # an admin may open side chats whatever the default permissions say
    page = page_for(make_user(role="admin"))
    side = open_new_chat_tab(page)
    question = f"early question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("early answer", match=reply.answering(question)))

    creations = hold(page, "**/api/v1/chats/new")
    try:
        side.send(question)
        wait_until_held(page, creations, "the side chat to be created")
        expect(side.conversation.get_by_text(question)).to_be_visible()
        expect(side.input).not_to_contain_text(question)
    finally:
        with page.expect_response("**/api/v1/chats/new"):
            for route in creations:
                route.fulfill(status=500, json={"detail": "the database is busy"})
    page.unroute("**/api/v1/chats/new")

    expect(side.input, "a refused chat creation lost the message").to_contain_text(question)
    expect(side.conversation.get_by_text(question)).to_have_count(0)

    side.input.click()
    page.keyboard.press("Enter")
    side.expect_reply("early answer")
    expect(side.conversation.get_by_text(question)).to_be_visible()


CHAT_CREATION = re.compile(r"/api/v1/chats/new$|/api/chat/completions$")
FILE_UPLOAD = re.compile(r"/api/v1/files/(\?|$)")


def hold_uploads(page: Page) -> list[Route]:
    held: list[Route] = []

    def handle(route: Route) -> None:
        if route.request.method == "POST":
            held.append(route)
        else:
            route.continue_()

    page.route(FILE_UPLOAD, handle)
    return held


def stored_messages(owner, chat_id: str, answer: str) -> list[dict]:
    """The chat's stored messages once the reply holding `answer` is saved as done."""
    stored: dict = {}
    with owner.client() as client:
        for _ in range(100):
            found = client.get(f"/api/v1/chats/{chat_id}")
            assert found.status_code == 200, found.text
            stored = found.json()["chat"]["history"]["messages"]
            if any(m.get("done") and answer in json.dumps(m) for m in stored.values()):
                return list(stored.values())
            time.sleep(0.1)
    raise AssertionError(f"the reply was never saved in the chat; it holds {stored}")


def test_the_first_message_of_a_new_chat_shows_while_the_chat_is_created(
    page_for, make_user, upstream
):
    owner = make_user()
    page = page_for(owner)
    question = f"instant question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("instant answer", match=reply.answering(question)))

    creations = hold(page, CHAT_CREATION)
    try:
        send(page, question)
        wait_until_held(page, creations, "the new chat to be created")
        expect(
            conversation(page).get_by_text(question),
            "the first message waited for its chat to be created",
        ).to_be_visible()
        expect(chat_input(page)).not_to_contain_text(question)
    finally:
        for route in creations:
            route.continue_()
        page.unroute(CHAT_CREATION)

    expect_reply(page, "instant answer")
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat_id = page.url.rsplit("/c/", 1)[1]
    expect(
        page.locator(f'a[href="/c/{chat_id}"]'), "the new chat is missing in the sidebar"
    ).to_contain_text(question)
    saved = stored_messages(owner, chat_id, "instant answer")
    assert [m["content"] for m in saved if m["role"] == "user"] == [question], saved


def test_the_first_message_waits_for_a_file_still_uploading_with_it(page_for, make_user, upstream):
    owner = make_user()
    page = page_for(owner)
    question = f"what does the attachment say? {uuid.uuid4().hex[:6]}"
    answering = reply.answering(question)
    upstream.queue(
        reply.tool_call("grep_chat_files", {"pattern": "mentions"}, match=answering),
        reply.text("Noted.", match=answering),
    )

    uploads = hold_uploads(page)
    try:
        menu = open_input_menu(page)
        with page.expect_file_chooser() as chooser:
            menu.get_by_role("menuitem", name="Upload Files").click()
        chooser.value.set_files(
            {
                "name": "harbour.txt",
                "mimeType": "text/plain",
                "buffer": b"The file mentions herons.",
            }
        )
        wait_until_held(page, uploads, "the file to be uploaded")
        send(page, question)
        expect(
            page.get_by_role("main").get_by_text(question).first, "the message was lost"
        ).to_be_visible()
        page.wait_for_timeout(1500)  # bounded: nothing may reach the model while the file uploads
        asked = [body for body in upstream.chat_requests() if answering(body)]
        assert not asked, "the first message went out while its file was still uploading"
    finally:
        for route in uploads:
            route.continue_()
        page.unroute(FILE_UPLOAD)

    expect_reply(page, "Noted.")
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    expect(conversation(page).get_by_text("harbour.txt")).to_be_visible()
    answered = [body for body in upstream.chat_requests() if answering(body)]
    read_back = json.dumps(this_turns_tool_results(answered[-1]))
    assert "herons" in read_back, (
        f"the model could not read the file; its tools returned {read_back}"
    )
