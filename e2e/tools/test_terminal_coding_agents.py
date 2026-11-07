"""Journey: a person picks a terminal, picks its coding agent as the model and chats with it.

New on dev for 0.13.1 (6ef26be29 and follow-ups, 1ef75436f, 72d52d2cd, titled `refac`). Picking
a terminal in the chat's Terminal menu adds that terminal's installed agents to the model
selector; the agent's answer shows in the chat and stays after a reload, and the reopened chat
continues with the same agent. A permission request shows Allow and Deny, a question shows the
question card, and an agent's error shows on the reply. Agent replies offer no Regenerate, no
Save As Copy and no Send on an edited message, since an agent conversation cannot branch. A reply
should keep the agent's name once its terminal is gone; on dev it shows the raw `agent:` id after
a reload, as the stored reply never carries the name (issue draft real-nf-agents-1, not filed
yet), so that test stays red. The agent is `harness/fake_acp_agent.py`, run by a real Open
Terminal.

Discriminates: the other tests pass on dev f659c0d12. In a backend copy whose agent discovery
returns no models every test fails at the model selector. The no-branching test fails on a build
with 1ef75436f's branch guards undone (Regenerate shows).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.coding_agent import FAST, connect_agent_terminal
from harness.fake_acp_agent import REFUSAL
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals
from utils.cached_chat import choose_terminal
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    conversation,
    expect_reply,
    last_reply,
    regenerate_buttons,
    send,
    set_tool_approval_mode,
)
from utils.model_selector import select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
AGENT_NAME = f"opencode/{FAST}"


@pytest.fixture
def agent_chat(page_for, make_user, admin, preserve, agent_terminal):
    """A chat page of a fresh account with the agent terminal picked and its agent selected."""
    preserve(TERMINAL_SERVERS_CONFIG)

    def open_chat(**fields) -> Page:
        person = make_user()
        name = f"Agent box {uuid.uuid4().hex[:6]}"
        connect_agent_terminal(admin, agent_terminal, person, name=name, **fields)
        page = page_for(person)
        choose_terminal(page, name)
        select_model(page, AGENT_NAME)
        expect(page.get_by_role("button", name=re.compile("Selected model"))).to_contain_text(
            AGENT_NAME
        )
        return page

    return open_chat


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True})
    assert saved.status_code == 200, saved.text


def _whole_message(page: Page, inside: Locator) -> Locator:
    """The last message holding `inside` (found within each message), with its name and buttons."""
    return conversation(page).get_by_role("listitem").filter(has=inside).last


def _wait_until_saved(page: Page) -> None:
    """The reply is on screen before it is stored; wait for the chat's own page to be done."""
    page.wait_for_url(re.compile(r"/c/[^/?#]+"), timeout=REPLY_TIMEOUT_MS)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0, timeout=REPLY_TIMEOUT_MS)


def test_the_picked_terminals_agent_answers_and_the_reopened_chat_continues_with_it(agent_chat):
    page = agent_chat()
    prompt = f"hello agent {uuid.uuid4().hex[:6]}"

    send(page, prompt)
    expect_reply(page, f"Agent heard: {prompt} [model {FAST},")
    _wait_until_saved(page)
    page.reload()
    expect(last_reply(page)).to_contain_text(f"Agent heard: {prompt}", timeout=REPLY_TIMEOUT_MS)
    expect(chat_input(page)).to_be_visible()
    send(page, "and once more")
    expect_reply(page, "Agent heard: and once more [model")


def test_an_agent_reply_offers_no_way_to_branch_the_conversation(agent_chat):
    page = agent_chat()

    send(page, "branch me")
    expect_reply(page, "Agent heard: branch me")
    _wait_until_saved(page)

    answer = _whole_message(page, page.locator(".chat-assistant"))
    expect(answer.get_by_label("Edit", exact=True)).to_have_count(1)
    expect(regenerate_buttons(page)).to_have_count(0)
    answer.hover()
    answer.get_by_label("Edit", exact=True).click()
    expect(answer.get_by_role("button", name="Save", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Save As Copy")).to_have_count(0)
    answer.get_by_role("button", name="Cancel", exact=True).click()

    # the question is the chat's first message; editing turns its text into a text box
    question = conversation(page).get_by_role("listitem").first
    question.hover()
    question.get_by_label("Edit", exact=True).click()
    expect(question.get_by_role("button", name="Save", exact=True)).to_be_visible()
    expect(question.get_by_role("button", name="Send", exact=True)).to_have_count(0)


def test_an_agent_reply_keeps_the_agents_name_once_its_terminal_is_gone(agent_chat, admin):
    """Red on dev f659c0d12: the reopened reply shows the raw `agent:` model id.

    The page keeps the name it showed only in its own memory; the stored reply never carries it.
    Issue draft: real-nf-agents-1 (title `issue: [dev] ...`), not filed yet.
    """
    page = agent_chat()

    send(page, "who are you?")
    expect_reply(page, "Agent heard: who are you?")
    _wait_until_saved(page)
    expect(
        _whole_message(page, page.locator(".chat-assistant")).get_by_role(
            "heading", name=AGENT_NAME
        )
    ).to_be_visible()
    with admin.client() as client:
        configure_terminals(client)
    page.reload()

    expect(last_reply(page)).to_contain_text("Agent heard: who are you?", timeout=REPLY_TIMEOUT_MS)
    answer = _whole_message(page, page.locator(".chat-assistant"))
    expect(answer.get_by_role("heading", name=AGENT_NAME, exact=True)).to_be_visible()


def test_an_agent_permission_request_is_allowed_from_the_chat(agent_chat, tool_approval_on):
    page = agent_chat()
    set_tool_approval_mode(page, "Ask for approval")

    send(page, "[permission] clean the build")
    allow = conversation(page).get_by_role("button", name="Allow", exact=True)
    expect(allow).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    allow.click()

    expect_reply(page, "Permission outcome: allow-it")


def test_an_agent_permission_request_can_be_denied_from_the_chat(agent_chat, tool_approval_on):
    page = agent_chat()
    set_tool_approval_mode(page, "Ask for approval")

    send(page, "[permission] clean the build")
    deny = conversation(page).get_by_role("button", name="Deny", exact=True)
    expect(deny).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
    deny.click()

    expect_reply(page, "Permission outcome: deny-it")


def test_an_agent_question_is_answered_on_its_card(agent_chat):
    page = agent_chat()

    send(page, "[ask] style the button")
    question = page.get_by_text("Which colour should the button be?")
    expect(question).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    card = page.locator("section", has=question)
    card.get_by_role("button", name=re.compile(r"^amber\b")).click()

    expect_reply(page, "Colour answer: amber")


def test_an_agent_error_shows_on_the_reply(agent_chat):
    page = agent_chat()

    send(page, "[refuse] do it")

    expect(last_reply(page)).to_contain_text(REFUSAL, timeout=REPLY_TIMEOUT_MS)
