"""Journey: with tool permissions on, the account allows or denies each tool call, or allows all.

The admin's tool permissions switch puts a model's tool call up for approval when the account's
composer is set to Ask for approval. Deny keeps the tool from running and tells the model the user
rejected it, and the reply still finishes; Allow runs it. Ctrl+Alt+Enter and Ctrl+Alt+Backspace
allow and deny the call on screen. With Full access, the default, the tool runs without asking.
The mode an account picks is kept for its next chats and for no other account. A call waiting for
approval marks its chat in the sidebar and still waits, with Allow on screen, after the page is
opened again. With the admin's Auto permissions on (0.13.1), Auto is the default: a reviewer
model rates each call, and only a call it rates risky stops for Allow or Deny. A side chat asks
before a tool call the same way its chat does. Expected but missing on f659c0d12 (strict xfail):
an Always allow choice on the approval card that lets the same tool run without asking next time.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose allow and deny actions are swapped
the deny, allow and shortcut tests fail, in one that asks whatever the composer says the full
access test fails, and in a frontend build that ignores the allow and deny shortcuts the shortcut
test fails. On f659c0d12, in a backend copy that never marks a chat as waiting the sidebar test
fails, in one whose Auto never stops a call the Auto test fails, and in one that drops the picked
mode from the saved account settings the kept-mode test fails. The side chat test is red on
f659c0d12, a real bug: the server pauses the side chat's call, but the side chat shows no Allow or
Deny, so it stays on "Preparing" for good (open-webui/open-webui#33873, side chats saved since
3b1a95383).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.backends import wait_until
from harness.listener import ReceivedRequest, json_answer
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    conversation,
    expect_reply,
    open_input_menu,
    send,
    set_tool_approval_mode,
    side_chat_panel,
)

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


def rated_risky_when_asked_so(request: ReceivedRequest):
    """The reviewer rates a call risky when the user's request says so."""
    evidence = json.loads(request.json()["input"])
    probability = 0.9 if "risky" in evidence["user_request"] else 0.1
    answer = {"type": "predicate", "name": "requires_confirmation", "probability": probability}
    return json_answer({"answers": [answer]})


@pytest.fixture
def auto_permissions_on(admin, preserve, listener):
    """Tool permissions with Auto on, reviewed by the local listener; yields the listener."""
    listener.route("POST", "/v1/decisions", rated_risky_when_asked_so)
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_TOOL_PERMISSIONS": True,
                "ENABLE_TOOL_PERMISSIONS_AUTO": True,
                "TOOL_PERMISSIONS_DECISIONS_API_BASE_URL": f"{listener.base_url}/v1",
                "TOOL_PERMISSIONS_DECISIONS_MODEL": "reviewer-model",
                "TOOL_PERMISSIONS_DECISIONS_THRESHOLD": 0.5,
            },
        )
        assert saved.status_code == 200, saved.text
    return listener


def _ask_for_the_time(page: Page, upstream, answer: str, prompt: str = "what time is it?") -> str:
    question = f"{prompt} {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(question)),
        reply.text(answer, match=reply.answering(question)),
    )
    send(page, question)
    return question


def _tool_results(upstream, question: str) -> list[str]:
    final = [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]
    return [str(entry["content"]) for entry in reply.conversation(final) if entry["role"] == "tool"]


def _approval_button(page: Page, name: str):
    """The card's `name` button, once the server holds the call as waiting for approval."""
    button = conversation(page).get_by_role("button", name=name, exact=True)
    expect(button).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    _wait_until_the_server_asks(page)
    return button


def _wait_until_the_server_asks(page: Page) -> None:
    # the card can show before the pause is stored, and an answer sent then is refused
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat_id = page.url.rsplit("/c/", 1)[1]
    token = page.evaluate("localStorage.token")

    def waiting() -> bool:
        stored = page.request.get(
            f"/api/v1/chats/{chat_id}", headers={"Authorization": f"Bearer {token}"}
        ).json()["chat"]["history"]
        output = stored["messages"][stored["currentId"]].get("output") or []
        return any(
            item.get("type") == "function_call" and item.get("status") == "pending"
            for item in output
        )

    assert wait_until(waiting, timeout=REPLY_TIMEOUT_MS / 1000), "the server never paused the call"


def test_a_denied_call_never_runs_and_the_model_is_told(
    tool_approval_on, page_for, make_user, upstream
):
    page = page_for(make_user())
    set_tool_approval_mode(page, "Ask for approval")
    question = _ask_for_the_time(page, upstream, "I was not allowed to check.")

    _approval_button(page, "Deny").click()

    expect_reply(page, "I was not allowed to check.")
    assert _tool_results(upstream, question) == [REJECTED]


def test_an_allowed_call_runs(tool_approval_on, page_for, make_user, upstream):
    page = page_for(make_user())
    set_tool_approval_mode(page, "Ask for approval")
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
    set_tool_approval_mode(page, "Ask for approval")
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


def _picked_mode(page: Page) -> Locator:
    """The Tool Permissions entry of the More menu, which reads out the picked mode."""
    return open_input_menu(page).get_by_role("menuitem", name="Tool Permissions")


def _open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar).to_be_visible()
    return sidebar


def _waiting_marker(sidebar: Locator) -> Locator:
    return sidebar.get_by_role("status", name="Waiting for approval")


def test_the_picked_mode_is_kept_for_new_chats_of_that_account_only(
    tool_approval_on, page_for, make_user, upstream
):
    account = make_user()
    set_tool_approval_mode(page_for(account), "Ask for approval")
    with account.client() as client:
        assert wait_until(
            lambda: (
                client.get("/api/v1/users/user/settings")
                .json()["ui"]["params"]
                .get("tool_approval_mode")
                == "ask"
            ),
            timeout=10,
        ), "the picked mode was never saved for the account"

    later = page_for(account)
    _ask_for_the_time(later, upstream, "Asked again in a new chat.")
    _approval_button(later, "Allow").click()
    expect_reply(later, "Asked again in a new chat.")
    expect(_picked_mode(later)).to_contain_text("Ask for approval")

    other = page_for(make_user())
    expect(_picked_mode(other)).to_contain_text("Full access")


def test_a_call_waiting_for_approval_is_marked_and_still_waits_after_a_reload(
    tool_approval_on, page_for, make_user, upstream
):
    account = make_user()
    page = page_for(account)
    set_tool_approval_mode(page, "Ask for approval")
    question = _ask_for_the_time(page, upstream, "Checked after the reload.")
    _approval_button(page, "Allow")
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat_path = page.url[page.url.index("/c/") :]
    expect(_waiting_marker(_open_sidebar(page))).to_be_visible()

    reopened = page_for(account, path=chat_path)
    sidebar = _open_sidebar(reopened)
    expect(_waiting_marker(sidebar)).to_be_visible()
    _approval_button(reopened, "Allow").click()

    expect_reply(reopened, "Checked after the reload.")
    [result] = _tool_results(upstream, question)
    assert result != REJECTED
    expect(_waiting_marker(sidebar)).to_have_count(0)


def test_auto_runs_a_safe_call_and_stops_a_risky_one_for_approval(
    auto_permissions_on, page_for, make_user, upstream
):
    page = page_for(make_user())

    safe = _ask_for_the_time(page, upstream, "Checked without asking.")
    expect_reply(page, "Checked without asking.")
    expect(conversation(page).get_by_role("button", name="Allow", exact=True)).to_have_count(0)
    assert _tool_results(upstream, safe) != [REJECTED]

    risky = _ask_for_the_time(page, upstream, "Checked once allowed.", prompt="risky: what time?")
    _approval_button(page, "Allow").click()
    expect_reply(page, "Checked once allowed.")
    # the chat's earlier call and its result are sent again ahead of this one
    assert _tool_results(upstream, risky)[-1] != REJECTED
    reviewed = [
        json.loads(entry.json()["input"])["user_request"] for entry in auto_permissions_on.received
    ]
    assert len(reviewed) == 2, f"the reviewer was not asked once per call: {reviewed}"
    assert safe in reviewed[0] and risky in reviewed[1], reviewed
    expect(_picked_mode(page)).to_contain_text("Auto")


@pytest.fixture
def side_chat_allowed(admin, preserve) -> None:
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def test_a_side_chat_asks_before_a_tool_call_like_its_chat(
    tool_approval_on, side_chat_allowed, page_for, make_user, upstream
):
    """Red on f659c0d12 (open-webui/open-webui#33873): the side chat shows no Allow or Deny."""
    page = page_for(make_user())
    set_tool_approval_mode(page, "Ask for approval")
    _ask_for_the_time(page, upstream, "The chat checked the clock.")
    _approval_button(page, "Allow").click()
    expect_reply(page, "The chat checked the clock.")

    aside = f"by the way, what time is it now? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(aside)),
        reply.text("The side chat checked the clock.", match=reply.answering(aside)),
    )
    send(page, f"/side {aside}")
    panel = side_chat_panel(page, aside)
    allow = panel.get_by_role("button", name="Allow", exact=True)
    expect(allow).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    asked = [body for body in upstream.chat_requests() if reply.answering(aside)(body)]
    assert len(asked) == 1, "the side chat ran the tool before it was allowed"
    # the chat's own tool result travels along as the side chat's context
    context_results = _tool_results(upstream, aside)
    allow.click()

    expect(panel).to_contain_text("The side chat checked the clock.", timeout=REPLY_TIMEOUT_MS)
    results = _tool_results(upstream, aside)
    assert results[:-1] == context_results, results
    assert results[-1] != REJECTED


@pytest.mark.xfail(
    strict=True,
    reason="gap: the approval card offers only Allow and Deny, no way to always allow a tool",
)
def test_a_tool_always_allowed_runs_without_asking_next_time(
    tool_approval_on, page_for, make_user, upstream
):
    page = page_for(make_user())
    set_tool_approval_mode(page, "Ask for approval")
    _ask_for_the_time(page, upstream, "Checked and remembered.")
    _approval_button(page, "Allow")
    always = conversation(page).get_by_role("button", name=re.compile(r"^Always allow"))
    # the card is on screen, so its buttons are all there
    assert always.count() == 1, "the approval card has no Always allow choice"
    always.click()
    expect_reply(page, "Checked and remembered.")

    _ask_for_the_time(page, upstream, "Checked without asking again.")
    expect_reply(page, "Checked without asking again.")
    expect(conversation(page).get_by_role("button", name="Allow", exact=True)).to_have_count(0)
