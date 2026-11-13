"""Regression: a message sent while a freshly loaded new chat was still setting up lost its reply.

Issue open-webui/open-webui#32819. Loading the page starts a new chat whose setup waits for the
function and skill lists. A first message sent before those arrived was wiped from the screen when
the setup finished, so its reply never showed, the page stayed on the empty new chat and only the
sidebar held the finished chat. The fix stops the setup from resetting a chat that got a message
in the meantime. Since b29f7781a the page also holds the input back until the setup is done.

The test holds the function list from a fresh page load and sends the first message as early as
the page allows: while the list is still held when the input is already offered, else right after
the list arrives. The send waits for its answer while the rest of the setup finishes.

Discriminates: fails on the dev ca8f451cc build (the message and its reply vanish and the page
stays on the new chat), passes with the fix d1511abce and on dev 2b1fb53f4; on 2b1fb53f4 with both
the held-back input and the fix's reset guard removed it fails the same way.

Since 0.13.4 the first message of a new chat shows at once while the chat is still being
created. A side chat creates its chat when its first message is sent, so the second test holds
that creation and expects the message on screen meanwhile. It passes on dev 485103cde and fails
in a frontend copy of it that creates the chat before showing the message.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, Route, expect

from harness import upstream as reply
from utils.chat_surface import open_new_chat_tab
from utils.chat_ui import chat_input, conversation, expect_reply, send

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


def release(page: Page, function_lists: list[Route]) -> None:
    with page.expect_response("**/api/v1/skills/**"):
        for route in function_lists:
            route.continue_()


def test_a_first_message_sent_during_new_chat_setup_keeps_its_reply(page_for, make_user, upstream):
    page = page_for(make_user())
    question = f"early question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("early answer", match=reply.answering(question)))

    function_lists = hold(page, "**/api/v1/functions/")
    sends = hold(page, "**/api/chat/completions")
    page.reload()
    wait_until_held(page, function_lists, "the function list")

    # the setup sets the input's state before it asks for the list, so this reads it settled
    if chat_input(page).is_visible():
        send(page, question)
        wait_until_held(page, sends, "the reply")
        release(page, function_lists)
    else:
        release(page, function_lists)
        send(page, question)
        wait_until_held(page, sends, "the reply")
    for route in sends:
        route.continue_()

    expect(conversation(page)).to_contain_text(question)
    expect_reply(page, "early answer")
    expect(page).to_have_url(re.compile(r"/c/"))

    # a New Chat started afterwards still opens empty
    page.get_by_role("link", name="New Chat").first.click()
    expect(page).not_to_have_url(re.compile(r"/c/"))
    expect(conversation(page).get_by_text(question)).to_have_count(0)


def test_the_first_message_of_a_side_chat_shows_while_its_chat_is_still_being_created(
    page_for, make_user, upstream
):
    # an admin may open side chats whatever the default permissions say
    page = page_for(make_user(role="admin"))
    side = open_new_chat_tab(page)
    question = f"instant question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("instant answer", match=reply.answering(question)))

    creations = hold(page, "**/api/v1/chats/new")
    try:
        side.send(question)
        wait_until_held(page, creations, "the side chat to be created")
        expect(
            side.conversation.get_by_text(question),
            "the first message waited for its chat to be created",
        ).to_be_visible()
    finally:
        for route in creations:
            route.continue_()

    side.expect_reply("instant answer")
