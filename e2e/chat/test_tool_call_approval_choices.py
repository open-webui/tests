"""Journey: with tool permissions on, the account allows or denies each tool call, or allows all.

The admin's tool permissions switch puts a model's tool call up for approval when the account's
composer is set to Ask for approval. Deny keeps the tool from running and tells the model the user
rejected it, and the reply still finishes; Allow runs it. Ctrl+Alt+Enter and Ctrl+Alt+Backspace
allow and deny the call on screen. With Full access, the default, the tool runs without asking.
The mode an account picks is kept for its next chats and for no other account. A call waiting for
approval marks its chat in the sidebar and still waits, with Allow on screen, after the page is
opened again. With the admin's Auto permissions on (0.13.1), Auto is the default and only a call
the review finds risky stops for Allow or Deny: with the default engine (since e8c0ad85e) the chat
model marks each call itself, with the Decisions engine a reviewer model rates it. In Admin
Settings > Interface, Auto permissions shows an Engine choice set to Default, and picking Decisions
API shows its address, model and threshold, saved with the approval policy and shown again after
a reload. A side chat asks before a tool call the same way its chat does.

The deny, allow, shortcut, full access, kept mode and both Auto tests run in the main chat and in a
side chat (`utils/chat_surface.py`) with the same assertions; the mode picked in a side chat is
kept for the account's next chats as the one picked in the main chat is. PR
open-webui/open-webui#34064 says the approval shortcuts still act on the last answer on the page;
with a call waiting in a tab beside the chat as well, the shortcuts in the chat's message box
answer the other tab's call on dev 782f4b705 (open-webui/open-webui#34186).

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose allow and deny actions are swapped
the deny, allow and shortcut tests fail, in one that asks whatever the composer says the full
access test fails, and in a frontend build that ignores the allow and deny shortcuts the shortcut
test fails. On f659c0d12, in a backend copy that never marks a chat as waiting the sidebar test
fails, in one whose Auto never stops a call the Auto test fails, and in one that drops the picked
mode from the saved account settings the kept-mode test fails. The Decisions Auto test was
retargeted for e8c0ad85e and picks that engine; the chat model review test passes on e8c0ad85e
and fails in a backend copy of it that never offers the mark (the safe call asks too). The
engine test fails in backend copies whose default engine is `decisions` or that never save the
engine. The side
chat test is red on f659c0d12, a real bug: the server pauses the side chat's call, but the side
chat shows no Allow or Deny, so it stays on "Preparing" for good (open-webui/open-webui#33873,
side chats saved since 3b1a95383).
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
from harness.plain_http import loopback_url
from utils.chat_surface import MAIN_CHAT, SIDE_CHAT, ChatSurface, main_chat, open_new_chat_tab
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    Chat,
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
                "TOOL_PERMISSIONS_AUTO_ENGINE": "decisions",
                "TOOL_PERMISSIONS_AUTO_DECISIONS_API_BASE_URL": f"{listener.base_url}/v1",
                "TOOL_PERMISSIONS_AUTO_DECISIONS_MODEL": "reviewer-model",
                "TOOL_PERMISSIONS_AUTO_DECISIONS_THRESHOLD": 0.5,
            },
        )
        assert saved.status_code == 200, saved.text
    return listener


def _ask_for_the_time(page: Chat, upstream, answer: str, prompt: str = "what time is it?") -> str:
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


def _approval_button(chat: Chat, name: str):
    """The card's `name` button, once the server holds the call as waiting for approval."""
    button = conversation(chat).get_by_role("button", name=name, exact=True)
    expect(button).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    _wait_until_the_server_asks(chat.page if isinstance(chat, ChatSurface) else chat)
    return button


def _wait_until_the_server_asks(page: Page) -> None:
    # the card can show before the pause is stored, and an answer sent then is refused
    expect(page).to_have_url(re.compile(r"/c/[\w-]+$"))
    chat_id = page.url.rsplit("/c/", 1)[1]
    token = page.evaluate("localStorage.token")

    def waiting() -> bool:
        stored = page.request.get(
            loopback_url(page.url, f"/api/v1/chats/{chat_id}"),
            headers={"Authorization": f"Bearer {token}"},
        ).json()["chat"]["history"]
        output = stored["messages"][stored["currentId"]].get("output") or []
        return any(
            item.get("type") == "function_call" and item.get("status") == "pending"
            for item in output
        )

    assert wait_until(waiting, timeout=REPLY_TIMEOUT_MS / 1000), "the server never paused the call"


def test_a_denied_call_never_runs_and_the_model_is_told(
    tool_approval_on, chat_for, make_user, upstream
):
    chat = chat_for(make_user())
    set_tool_approval_mode(chat, "Ask for approval")
    question = _ask_for_the_time(chat, upstream, "I was not allowed to check.")

    _approval_button(chat, "Deny").click()

    expect_reply(chat, "I was not allowed to check.")
    assert _tool_results(upstream, question) == [REJECTED]


def test_an_allowed_call_runs(tool_approval_on, chat_for, make_user, upstream):
    chat = chat_for(make_user())
    set_tool_approval_mode(chat, "Ask for approval")
    question = _ask_for_the_time(chat, upstream, "Checked the clock.")

    _approval_button(chat, "Allow").click()

    expect_reply(chat, "Checked the clock.")
    [result] = _tool_results(upstream, question)
    assert result != REJECTED
    assert result.strip()


def test_the_shortcuts_allow_and_deny_the_call_on_screen(
    tool_approval_on, chat_for, make_user, upstream
):
    chat = chat_for(make_user())
    set_tool_approval_mode(chat, "Ask for approval")
    allowed = _ask_for_the_time(chat, upstream, "Allowed by keyboard.")
    _approval_button(chat, "Allow")
    chat.page.keyboard.press("Control+Alt+Enter")
    expect_reply(chat, "Allowed by keyboard.")
    assert _tool_results(upstream, allowed) != [REJECTED]

    denied = _ask_for_the_time(chat, upstream, "Denied by keyboard.")
    _approval_button(chat, "Deny")
    chat.page.keyboard.press("Control+Alt+Backspace")
    expect_reply(chat, "Denied by keyboard.")
    assert _tool_results(upstream, denied)[-1:] == [REJECTED]


# a tab asked later is the other chat: next to the main chat, or next to a side chat
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT)
def test_the_shortcuts_in_a_message_box_answer_that_chats_call_not_a_later_one_beside_it(
    tool_approval_on, chat_for, make_user, upstream
):
    """Red on dev 782f4b705 in both places: the shortcuts answer the last call on the page, the
    other tab's (open-webui/open-webui#34186)."""
    # an admin may open a New Chat tab beside the main chat
    chat = chat_for(make_user(role="admin"))
    set_tool_approval_mode(chat, "Ask for approval")
    allowed = _ask_for_the_time(chat, upstream, "Allowed in this chat.")
    _approval_button(chat, "Allow")
    main_chat(chat.page).show()
    other = open_new_chat_tab(chat.page)
    _ask_for_the_time(other, upstream, "The other chat was answered.")
    _approval_button(other, "Allow")

    chat.show()
    chat_input(chat).click()
    chat.page.keyboard.press("Control+Alt+Enter")
    expect_reply(chat, "Allowed in this chat.")
    assert _tool_results(upstream, allowed) != [REJECTED]

    denied = _ask_for_the_time(chat, upstream, "Denied in this chat.")
    _approval_button(chat, "Deny")
    chat_input(chat).click()
    chat.page.keyboard.press("Control+Alt+Backspace")
    expect_reply(chat, "Denied in this chat.")
    assert _tool_results(upstream, denied)[-1:] == [REJECTED]
    other.show()
    expect(conversation(other).get_by_role("button", name="Allow", exact=True)).to_have_count(1)


def test_full_access_runs_the_call_without_asking(tool_approval_on, chat_for, make_user, upstream):
    chat = chat_for(make_user())
    expect(chat_input(chat)).to_be_visible()
    question = _ask_for_the_time(chat, upstream, "Checked without asking.")

    expect_reply(chat, "Checked without asking.")
    expect(conversation(chat).get_by_role("button", name="Allow", exact=True)).to_have_count(0)
    [result] = _tool_results(upstream, question)
    assert result != REJECTED


def _picked_mode(page: Chat) -> Locator:
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
    tool_approval_on, page_for, chat_for, make_user, upstream
):
    account = make_user()
    set_tool_approval_mode(chat_for(account), "Ask for approval")
    with account.client() as client:
        assert wait_until(
            lambda: (
                client.get("/api/v1/users/user/settings")
                .json()["ui"]
                .get("params", {})
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
    auto_permissions_on, chat_for, make_user, upstream
):
    chat = chat_for(make_user())

    safe = _ask_for_the_time(chat, upstream, "Checked without asking.")
    expect_reply(chat, "Checked without asking.")
    expect(conversation(chat).get_by_role("button", name="Allow", exact=True)).to_have_count(0)
    assert _tool_results(upstream, safe) != [REJECTED]

    risky = _ask_for_the_time(chat, upstream, "Checked once allowed.", prompt="risky: what time?")
    _approval_button(chat, "Allow").click()
    expect_reply(chat, "Checked once allowed.")
    # the chat's earlier call and its result are sent again ahead of this one
    assert _tool_results(upstream, risky)[-1] != REJECTED
    reviewed = [
        json.loads(entry.json()["input"])["user_request"] for entry in auto_permissions_on.received
    ]
    assert len(reviewed) == 2, f"the reviewer was not asked once per call: {reviewed}"
    assert safe in reviewed[0] and risky in reviewed[1], reviewed
    expect(_picked_mode(chat)).to_contain_text("Auto")


@pytest.fixture
def model_review_on(admin, preserve) -> None:
    """Tool permissions with Auto on and the default engine, where the chat model marks calls."""
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_TOOL_PERMISSIONS": True,
                "ENABLE_TOOL_PERMISSIONS_AUTO": True,
                "TOOL_PERMISSIONS_AUTO_ENGINE": "",
            },
        )
    assert saved.status_code == 200, saved.text


def _ask_for_the_time_marked(page: Chat, upstream, required: bool, answer: str) -> str:
    question = f"what time is it? {uuid.uuid4().hex[:6]}"
    marked = {"__approval__": {"required": required}}
    upstream.queue(
        reply.tool_call("get_current_timestamp", marked, match=reply.answering(question)),
        reply.text(answer, match=reply.answering(question)),
    )
    send(page, question)
    return question


def test_auto_with_the_chat_models_review_runs_a_call_it_marks_safe_and_asks_for_one_it_marks_risky(
    model_review_on, chat_for, make_user, upstream
):
    chat = chat_for(make_user())

    _ask_for_the_time_marked(chat, upstream, required=False, answer="Checked without asking.")
    expect_reply(chat, "Checked without asking.")
    expect(conversation(chat).get_by_role("button", name="Allow", exact=True)).to_have_count(0)

    risky = _ask_for_the_time_marked(chat, upstream, required=True, answer="Checked once allowed.")
    _approval_button(chat, "Allow").click()
    expect_reply(chat, "Checked once allowed.")
    assert _tool_results(upstream, risky)[-1] != REJECTED
    expect(_picked_mode(chat)).to_contain_text("Auto")


def test_an_admin_picks_the_auto_review_engine_and_its_settings_are_kept(
    tool_approval_on, page_for, admin, listener
):
    page = page_for(admin, path="/admin/settings/interface")
    settings = page.get_by_role("dialog")
    auto = settings.get_by_role("switch", name="Auto permissions")
    expect(auto).not_to_be_checked()
    auto.click()
    engine = settings.get_by_label("Engine", exact=True)
    expect(engine).to_have_value("")
    expect(settings.get_by_label("Approval threshold")).to_have_count(0)
    engine.select_option(label="Decisions API")
    settings.get_by_label("API Base URL", exact=True).last.fill(f"{listener.base_url}/v1")
    settings.get_by_label("Model", exact=True).last.fill("reviewer-model")
    settings.get_by_label("Approval threshold").fill("0.4")
    with admin.client() as client:
        default_policy = client.get(CHAT_CONFIG[0]).json()["DEFAULT_TOOL_PERMISSIONS_AUTO_PROMPT"]
    # the policy field shows the default policy until the admin types one
    settings.get_by_placeholder(default_policy, exact=True).fill(
        "Ask before anything that leaves the harbour."
    )
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!")).to_be_visible()

    with admin.client() as client:
        stored = client.get(CHAT_CONFIG[0]).json()
    assert {key: stored[key] for key in AUTO_FIELDS} == {
        "ENABLE_TOOL_PERMISSIONS_AUTO": True,
        "TOOL_PERMISSIONS_AUTO_ENGINE": "decisions",
        "TOOL_PERMISSIONS_AUTO_DECISIONS_API_BASE_URL": f"{listener.base_url}/v1",
        "TOOL_PERMISSIONS_AUTO_DECISIONS_MODEL": "reviewer-model",
        "TOOL_PERMISSIONS_AUTO_DECISIONS_THRESHOLD": 0.4,
        "TOOL_PERMISSIONS_AUTO_PROMPT": "Ask before anything that leaves the harbour.",
    }
    reopened = page_for(admin, path="/admin/settings/interface").get_by_role("dialog")
    expect(reopened.get_by_label("Engine", exact=True)).to_have_value("decisions")
    expect(reopened.get_by_label("Approval threshold")).to_have_value("0.4")


AUTO_FIELDS = [
    "ENABLE_TOOL_PERMISSIONS_AUTO",
    "TOOL_PERMISSIONS_AUTO_ENGINE",
    "TOOL_PERMISSIONS_AUTO_DECISIONS_API_BASE_URL",
    "TOOL_PERMISSIONS_AUTO_DECISIONS_MODEL",
    "TOOL_PERMISSIONS_AUTO_DECISIONS_THRESHOLD",
    "TOOL_PERMISSIONS_AUTO_PROMPT",
]


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
