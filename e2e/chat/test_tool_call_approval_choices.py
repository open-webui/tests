"""Journey: with tool permissions on, the account allows or denies each tool call, or allows all.

The admin's tool permissions switch puts a model's tool call up for approval when the account's
composer is set to Ask for approval. Deny keeps the tool from running and tells the model the user
rejected it, and the reply still finishes; Allow runs it. Ctrl+Alt+Enter and Ctrl+Alt+Backspace
allow and deny the call on screen. With Full access, the default, the tool runs without asking.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose allow and deny actions are swapped
the deny, allow and shortcut tests fail, in one that asks whatever the composer says the full
access test fails, and in a frontend build that ignores the allow and deny shortcuts the shortcut
test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
REJECTED = "Error: tool call rejected by user."


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        client.post(
            CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True}
        ).raise_for_status()


def _set_approval_mode(page: Page, mode: str) -> None:
    expect(chat_input(page)).to_be_visible()
    # the sidebar has a "More" button of its own
    page.locator("#input-menu-button").click()
    page.get_by_role("button", name="Tool Permissions").click()
    page.get_by_role("button", name=mode).click()
    page.keyboard.press("Escape")


def _ask_for_the_time(page: Page, upstream, answer: str) -> str:
    question = f"what time is it? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(question)),
        reply.text(answer, match=reply.answering(question)),
    )
    send(page, question)
    return question


def _tool_results(upstream, question: str) -> list[str]:
    final = [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]
    return [str(entry["content"]) for entry in final["messages"] if entry["role"] == "tool"]


def _approval_button(page: Page, name: str):
    button = conversation(page).get_by_role("button", name=name, exact=True)
    expect(button).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    return button


def test_a_denied_call_never_runs_and_the_model_is_told(
    tool_approval_on, page_for, make_user, upstream
):
    page = page_for(make_user())
    _set_approval_mode(page, "Ask for approval")
    question = _ask_for_the_time(page, upstream, "I was not allowed to check.")

    _approval_button(page, "Deny").click()

    expect_reply(page, "I was not allowed to check.")
    assert _tool_results(upstream, question) == [REJECTED]


def test_an_allowed_call_runs(tool_approval_on, page_for, make_user, upstream):
    page = page_for(make_user())
    _set_approval_mode(page, "Ask for approval")
    question = _ask_for_the_time(page, upstream, "Checked the clock.")

    _approval_button(page, "Allow").click()

    expect_reply(page, "Checked the clock.")
    [result] = _tool_results(upstream, question)
    assert result != REJECTED
    assert result.strip()


def test_the_shortcuts_allow_and_deny_the_call_on_screen(
    tool_approval_on, page_for, make_user, upstream
):
    page = page_for(make_user())
    _set_approval_mode(page, "Ask for approval")
    allowed = _ask_for_the_time(page, upstream, "Allowed by keyboard.")
    _approval_button(page, "Allow")
    page.keyboard.press("Control+Alt+Enter")
    expect_reply(page, "Allowed by keyboard.")
    assert _tool_results(upstream, allowed) != [REJECTED]

    denied = _ask_for_the_time(page, upstream, "Denied by keyboard.")
    _approval_button(page, "Deny")
    page.keyboard.press("Control+Alt+Backspace")
    expect_reply(page, "Denied by keyboard.")
    assert _tool_results(upstream, denied)[-1:] == [REJECTED]


def test_full_access_runs_the_call_without_asking(tool_approval_on, page_for, make_user, upstream):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    question = _ask_for_the_time(page, upstream, "Checked without asking.")

    expect_reply(page, "Checked without asking.")
    expect(conversation(page).get_by_role("button", name="Allow", exact=True)).to_have_count(0)
    [result] = _tool_results(upstream, question)
    assert result != REJECTED
