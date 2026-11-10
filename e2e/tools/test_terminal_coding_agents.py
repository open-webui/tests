"""Journey: a person picks a terminal, picks its coding agent as the model and chats with it.

New on dev for 0.13.1 (6ef26be29 and follow-ups, 1ef75436f, 72d52d2cd, titled `refac`). Picking
a terminal in the chat's Terminal menu adds that terminal's installed agents to the model
selector; the agent's answer shows in the chat and stays after a reload, and the reopened chat
continues with the same agent. A permission request shows Allow and Deny, a question shows the
question card, and an agent's error shows on the reply. Agent replies offer no Regenerate, no
Save As Copy and no Send on an edited message, since an agent conversation cannot branch. A reply
should keep the agent's name once its terminal is gone; on dev it shows the raw `agent:` id after
a reload, as the stored reply never carries the name (open-webui/open-webui#33879), so that
test stays red until its fix (branch fix/agent-reply-keeps-name) merges. The agent is
`harness/fake_acp_agent.py`, run by a real Open Terminal.

Discriminates: the other tests pass on dev f659c0d12. In a backend copy whose agent discovery
returns no models every test fails at the model selector. The no-branching test fails on a build
with 1ef75436f's branch guards undone (Regenerate shows). The name test fails on dev 1fc5b8f6d
and passes on fix/agent-reply-keeps-name (the reloaded reply shows the raw id without the fix).

Once the agent's terminal is gone, its chat and a new chat take the default model, also when the
person had the agent as their default model, and the next message goes to it (#33843, #33860).
That test passes on dev e8c0ad85e and fails on a frontend build of it with both fixes undone,
where no model is selected.

Three more journeys. A Codex model brings its reasoning efforts as a model control: the chat
input shows it with the model's default, and a pick reaches the agent's turn (the agent is
`harness/fake_native_agent.py`). A new chat in a folder with a default terminal and working
directory offers the agents the terminal lists for that directory, where a chat outside the
folder gets those of the terminal's own (the terminal is the fake of `harness/agent_carrier.py`,
which lists a different agent per directory). And an agent model cannot be edited or built on:
its entry in the model selector has no Edit, and the workspace's model editor does not offer it
as a base model, while it offers and edits the provider's model.
Discriminates: these three pass on the dev c15f28649 build. The effort test fails in a backend
copy that builds no model controls from the catalog (no Model controls row); the folder test
fails in a backend copy whose `/api/models` does not pass the directory on (the folder's chat is
offered the agent of the terminal's own directory); the edit and base model test fails on a
frontend build with the `terminal` checks of the model item menu and the base model list undone.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.agent_carrier import agent, catalog, serving_agents
from harness.coding_agent import FAST, connect_agent_terminal
from harness.fake_acp_agent import REFUSAL
from harness.plain_http import loopback_url
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import MOCK_MODEL_ID
from utils.cached_chat import choose_terminal
from utils.chat_controls import choose_model_control, model_control_row
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
from utils.model_selector import model_options, model_search, select_model

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
    open-webui/open-webui#33879; fixed on branch fix/agent-reply-keeps-name.
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


def _selected_model(page: Page) -> Locator:
    return page.get_by_role("button", name=re.compile("Selected model"))


def test_once_the_agents_terminal_is_gone_its_chat_and_a_new_chat_take_the_default_model(
    agent_chat, admin, upstream
):
    page = agent_chat()
    send(page, "remember me")
    expect_reply(page, "Agent heard: remember me")
    _wait_until_saved(page)
    token = page.evaluate("localStorage.token")
    stored = page.request.get(
        loopback_url(page.url, f"/api/v1/chats/{page.url.rsplit('/c/', 1)[1]}"),
        headers={"Authorization": f"Bearer {token}"},
    ).json()["chat"]
    [agent_model] = stored["models"]
    saved = page.request.post(
        loopback_url(page.url, "/api/v1/users/user/settings/update"),
        headers={"Authorization": f"Bearer {token}"},
        data={"ui": {"models": [agent_model]}},
    )
    assert saved.ok, saved.text()
    with admin.client() as client:
        configure_terminals(client)

    page.reload()
    expect(last_reply(page)).to_contain_text("Agent heard: remember me", timeout=REPLY_TIMEOUT_MS)
    expect(_selected_model(page)).to_contain_text(MOCK_MODEL_ID)
    expect(_selected_model(page)).not_to_contain_text(AGENT_NAME)

    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    expect(_selected_model(page)).to_contain_text(MOCK_MODEL_ID)
    question = f"hello again {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Back on the default model.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Back on the default model.")


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


def test_a_codex_models_reasoning_effort_is_picked_in_the_chat_and_reaches_the_agent(
    page_for, make_user, admin, preserve, native_agent_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    name = f"Agent box {uuid.uuid4().hex[:6]}"
    connect_agent_terminal(admin, native_agent_terminal, person, name=name)
    page = page_for(person)
    choose_terminal(page, name)
    select_model(page, "codex/codex-fake")
    prompt = f"think harder {uuid.uuid4().hex[:6]}"

    page.get_by_label("Model controls", exact=True).click()
    expect(model_control_row(page, "Reasoning effort")).to_contain_text("medium")
    page.keyboard.press("Escape")
    choose_model_control(page, "Reasoning effort", "high")
    send(page, prompt)
    expect_reply(page, f"codex heard: {prompt} [model codex-fake,")

    turns = [
        message["params"]
        for message in native_agent_terminal.messages("codex")
        if message.get("method") == "turn/start" and prompt in str(message["params"])
    ]
    assert [turn.get("effort") for turn in turns] == ["high"], (
        f"the picked effort did not reach the agent's turn: {turns}"
    )


def test_a_folders_new_chat_offers_the_agents_of_its_terminal_and_directory(
    page_for, make_user, admin, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    name = f"Agent box {uuid.uuid4().hex[:6]}"
    with serving_terminal() as terminal:
        agents = serving_agents(terminal)
        agents.catalog = lambda folder: catalog(
            agent("opencode", {"id": f"model-of-{(folder or 'nowhere').rsplit('/', 1)[-1]}"})
        )
        connection = terminal.connection(name=name)
        connection["config"]["access_grants"] = [read_grant(person.id)]
        with admin.client() as client:
            configure_terminals(client, connection)
        with person.client() as client:
            created = client.post(
                "/api/v1/folders/",
                json={
                    "name": f"Atlas {uuid.uuid4().hex[:6]}",
                    "data": {"terminal_id": connection["id"], "working_directory": "/srv/atlas"},
                },
            )
        assert created.status_code == 200, f"creating the folder failed: {created.text}"

        outside = page_for(person)
        choose_terminal(outside, name)
        expect(model_options(outside, "opencode/model-of-work")).to_be_visible()
        model_search(outside).fill("opencode/model-of-atlas")
        expect(model_options(outside, "opencode/model-of-atlas")).to_have_count(0)

        inside = page_for(person, path=f"/folders/{created.json()['id']}")
        expect(chat_input(inside)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        select_model(inside, "opencode/model-of-atlas")
        send(inside, "map the coast")
        expect_reply(inside, "Scripted agent heard: map the coast")
        assert agents.starts[-1].options["cwd"] == "/srv/atlas", (
            f"the folder chat's agent did not start in the folder's directory: {agents.starts[-1]}"
        )


def test_an_agent_model_has_no_edit_entry_and_is_not_offered_as_a_base_model(
    page_for, make_user, admin, preserve, agent_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    owner = make_user(role="admin")
    name = f"Agent box {uuid.uuid4().hex[:6]}"
    connect_agent_terminal(admin, agent_terminal, owner, name=name)
    page = page_for(owner)
    choose_terminal(page, name)

    def offers_edit(model_name: str) -> bool:
        """Whether the model's More Options in the model selector hold Edit."""
        row = model_options(page, model_name)
        row.hover()
        row.get_by_label("More Options", exact=True).click()
        menu = page.get_by_role("menu")
        expect(menu.get_by_role("menuitem", name="Copy Link", exact=True)).to_be_visible()
        offered = menu.get_by_role("menuitem", name="Edit", exact=True).count() == 1
        page.keyboard.press("Escape")
        expect(menu).to_have_count(0)
        # the first Escape may leave the selector open
        if model_search(page).count():
            page.keyboard.press("Escape")
        expect(model_search(page)).to_have_count(0)
        return offered

    assert offers_edit(MOCK_MODEL_ID), "the provider's model offers an admin no Edit"
    assert not offers_edit(AGENT_NAME), "an agent model offers Edit"

    page.get_by_role("link", name="Workspace", exact=True).click()
    page.get_by_role("button", name="Create", exact=True).click()
    page.wait_for_url(re.compile(r"/workspace/models/create"))
    page.get_by_role("main").get_by_role("button", name="Select a base model", exact=True).click()
    bases = page.get_by_role("listbox", name="Available models")
    model_search(page).fill(MOCK_MODEL_ID)
    expect(bases.get_by_role("option", name=f"Select {MOCK_MODEL_ID} model")).to_be_visible()
    model_search(page).fill("opencode")
    expect(bases.get_by_role("option")).to_have_count(0)
