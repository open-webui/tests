"""Journey: a long chat in the browser on the Prompt Caching page's setup only appends.

The Prompt Caching docs page promises that with its cache-optimal setup the request Open WebUI
sends to the provider keeps its start: every request repeats the tool list and every message of
the one before it byte for byte and only adds to the end. Its integration twin checks each
feature family over HTTP; here a person drives chats through the web client. One goes through a
file attached in the chat input and searched with the file tools, a reply with reasoning, the
model's knowledge searched, a native tool call, a page reload, the chat continued from a second
browser and a while passing between turns. Another asks about an image attached in the chat
input across a reload. Others have a real Open Terminal picked for the chat from the first turn,
with a file written, a command run and the file read back across a reload, and with every tool
the terminal offers called, its processes included. Then a chat started in a folder with its own
prompt and knowledge, the last reply regenerated, the temperature changed in Controls and the
model's description edited mid-chat, a workspace tool picked before the first turn, a workspace
skill loaded on demand and a sub-agent. Every consecutive pair of the provider's requests is
checked.

Five tests stay red, on what the page does not name as a breaker. Two rewrite the tool list at
the very start of the prefix: opening a folder in the terminal's file browser moves the working
directory written into the run_command tool's description (open-webui/open-webui#32026), and a
reload with the terminal's shell open closes it, which drops the two user shell tools
(open-webui/open-webui#31590, fix PR #31602 open). Two rewrite the system message: the turn a
timer or a background sub-agent's report starts is sent the chat's finished system prompt with
the model's system prompt and knowledge added again (open-webui/open-webui#31568, fix PR #31579
open). The Create skill command is sent as the skill authoring prompt on its own turn and as the
typed command on the next, so that turn rewrites an earlier user message
(open-webui/open-webui#31591, fix PR #31598 open).

Discriminates: passes on dev 30f3f6a8f apart from those five, which fail there. In backend
copies, a clock value added to the model's system prompt turned every other test red; with no
stored system prompt handed to a timer or a report and the Create skill command left as typed,
the timer, report and Create skill tests passed. Earlier, on dev 176d31d1d, the tool list
shuffled per request turned the first tests red, and with no working directory in the tool
description and the shell tools offered whether or not the shell is open, the two terminal
tests passed.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.mcp_server import SNAPSHOT_PNG
from harness.prompt_caching import (
    assert_append_only,
    cache_optimal_model,
    first_break,
    turn_off_memory_system_context,
)
from harness.python_tools import EVERYONE_READS, python_tool
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import (
    ask,
    attach,
    called_tools,
    calling,
    calling_together,
    chat_requests,
    pick_terminal,
    tool_results,
    turn_on_tool,
)
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TIMER_PROMPT = "Time to leave for the ferry."
TIDE_TOOL = """
class Tools:
    def tide_time(self, harbour: str) -> str:
        \"\"\"
        The next high tide in a harbour.
        :param harbour: The harbour's name
        \"\"\"
        return f"High tide in {harbour} is at noon."
"""
NOTES = ("berths.txt", "Berth 3 is reserved for the pilot boat.\nBerth 5 is free on Mondays.\n")


@pytest.fixture
def cached_setup(admin, preserve):
    """The page's setup: its model, and the memory system context switched off."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin) as model:
        yield model


def test_a_long_chat_in_the_browser_only_appends(page_for, cached_setup, make_user, upstream):
    person = make_user()
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    attach(page, *NOTES)
    ask(
        page,
        upstream,
        "which berth is free?",
        calling("query_chat_files", {"query": "free berth"}),
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
        calling("query_knowledge_files", {"query": "ferry"}),
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
        calling("get_current_timestamp", {}),
        reply.text("Just after six."),
    )
    ask(second, upstream, "thanks, that is all", reply.text("Safe travels."))

    requests = chat_requests(upstream)
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

    requests = chat_requests(upstream)
    first_question = requests[0]["messages"][1]["content"]
    assert any(part.get("type") == "image_url" for part in first_question), first_question
    assert_append_only(requests)


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
        calling("write_file", {"path": "ferry.txt", "content": "06:40 from pier 7\n"}),
        reply.text("Written."),
    )
    ask(
        page,
        upstream,
        "list my files",
        calling("run_command", {"command": "ls && cat ferry.txt", "wait": 10}),
        reply.text("ferry.txt is there."),
    )
    page.reload()
    ask(
        page,
        upstream,
        "read it back",
        calling("read_file", {"path": "ferry.txt"}),
        reply.text("06:40 from pier 7."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
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

    requests = chat_requests(upstream)
    broken = first_break(requests)
    assert broken is None, (
        "#32026: opening a folder in the terminal's file browser rewrote the run_command tool "
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

    requests = chat_requests(upstream)
    broken = first_break(requests)
    assert broken is None, (
        "the reload closed the terminal's shell, which dropped the two user shell tools from "
        f"the tool list at the start of the cached prefix: {broken}"
    )


def test_every_terminal_tool_only_appends(terminal_chat, open_terminal, upstream):
    page = terminal_chat
    ask(
        page,
        upstream,
        "what is this machine?",
        calling_together(("get_info", {}), ("list_files", {"directory": "."})),
        reply.text("A small machine."),
    )
    ask(
        page,
        upstream,
        "keep the tide times in a file",
        calling("write_file", {"path": "tides.txt", "content": "high 12:00\nlow 18:00\n"}),
        calling(
            "replace_file_content",
            {"path": "tides.txt", "replacements": [{"target": "18:00", "replacement": "18:15"}]},
        ),
        reply.text("Kept."),
    )
    ask(
        page,
        upstream,
        "find it again",
        calling_together(
            ("grep_search", {"query": "high", "path": "."}),
            ("glob_search", {"pattern": "*.txt"}),
            ("search_files", {"query": "tides"}),
            ("match_files", {"query": "low 18:15"}),
        ),
        calling("display_file", {"path": str(open_terminal.home / "tides.txt")}),
        reply.text("Found it."),
    )
    ask(
        page,
        upstream,
        "start a counter",
        calling("run_command", {"command": "read line && echo got $line", "wait": 0}),
        reply.text("Started."),
    )
    process_id = json.loads(chat_requests(upstream)[-1]["messages"][-1]["content"])["id"]
    ask(
        page,
        upstream,
        "feed it and stop it",
        calling("list_processes", {}),
        calling("send_process_input", {"process_id": process_id, "input": "tide\n"}),
        calling("get_process_status", {"process_id": process_id, "wait": 5}),
        calling("kill_process", {"process_id": process_id}),
        reply.text("Done."),
    )
    page.reload()
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    every_tool = {
        "get_info",
        "list_files",
        "write_file",
        "replace_file_content",
        "grep_search",
        "glob_search",
        "search_files",
        "match_files",
        "display_file",
        "run_command",
        "list_processes",
        "send_process_input",
        "get_process_status",
        "kill_process",
    }
    assert every_tool <= called_tools(requests[-1]), every_tool - called_tools(requests[-1])
    assert (open_terminal.home / "tides.txt").read_text() == "high 12:00\nlow 18:15\n"
    assert "got tide" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def test_a_chat_in_a_folder_with_its_own_prompt_and_knowledge_only_appends(
    page_for, cached_setup, make_user, upstream
):
    person = make_user()
    folder = {
        "model_ids": [cached_setup.id],
        "system_prompt": "Answer as the harbour master.",
        "files": [{"type": "collection", "id": cached_setup.knowledge_id, "name": "Handbook"}],
    }
    with person.client() as client:
        created = client.post("/api/v1/folders/", json={"name": "Harbour", "data": folder})
    assert created.status_code == 200, created.text
    page = page_for(person)
    page.goto(f"/folders/{created.json()['id']}")

    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(
        page,
        upstream,
        "when does the ferry leave?",
        calling("query_knowledge_files", {"query": "ferry"}),
        reply.text("At 06:40 (handbook.txt)."),
    )
    page.reload()
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert "Answer as the harbour master." in requests[0]["messages"][0]["content"]
    assert_append_only(requests)


def test_regenerating_the_last_reply_only_appends(page_for, cached_setup, make_user, upstream):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(page, upstream, "when does the ferry leave?", reply.text("Soon."))

    upstream.queue(reply.text("At 06:40.", match=reply.answering("when does the ferry leave?")))
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Regenerate").last.click()
    page.get_by_text("Try Again").click()
    expect_reply(page, "At 06:40.")
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert requests[2] == requests[1], "the regenerated reply was asked for differently"
    assert "Soon." not in json.dumps(requests[3]["messages"])
    assert_append_only(requests)


def test_a_temperature_and_model_description_change_mid_chat_only_append(
    page_for, cached_setup, admin, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))

    page.get_by_role("navigation").get_by_role("button", name="Controls").click()
    expect(page.get_by_role("textbox", name="Enter system prompt")).to_be_visible()
    temperature = page.get_by_text("Temperature", exact=True)
    temperature.locator("xpath=following-sibling::button").click()
    page.get_by_role("spinbutton", name="Temperature").fill("0.2")
    with admin.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": cached_setup.id}).json()
        stored["meta"]["description"] = "Answers questions about the harbour"
        updated = client.post("/api/v1/models/model/update", json=stored)
        assert updated.status_code == 200, updated.text
    ask(page, upstream, "when does the ferry leave?", reply.text("At 06:40."))
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert requests[1].get("temperature") == 0.2, requests[1].get("temperature")
    assert_append_only(requests)


@pytest.fixture
def harbour_skill(admin):
    """An active skill every account may read, there before the chat starts."""
    skill_id = f"tide-tables-{uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": skill_id,
                "name": "Tide tables",
                "description": "Reading tide tables",
                "content": "Read the high tide column first.",
                "meta": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield skill_id
        client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def test_a_skill_loaded_on_demand_only_appends(
    page_for, cached_setup, harbour_skill, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(
        page,
        upstream,
        "when is high tide?",
        calling("view_skill", {"id": harbour_skill}),
        reply.text("At noon."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert "<available_skills>" in requests[0]["messages"][0]["content"]
    assert "Read the high tide column first." in tool_results(requests[-1])
    assert_append_only(requests)


@pytest.fixture
def subagents_on(admin, preserve):
    """`subagents_on(**settings)` switches sub-agents and timers on for this test."""
    preserve(SUBAGENTS)

    def switch_on(**settings) -> None:
        with admin.client() as client:
            current = client.get(SUBAGENTS[0]).json()
            body = {**current, "ENABLE_SUBAGENTS": True, **settings}
            client.post(SUBAGENTS[1], json=body).raise_for_status()

    return switch_on


def requests_of_chat(upstream, opening: str) -> list[dict]:
    """The chat's own requests to the provider, without a sub-agent's."""
    return [
        body
        for body in chat_requests(upstream)
        if opening in str(next(m for m in body["messages"] if m["role"] == "user")["content"])
    ]


def test_a_sub_agent_only_appends(page_for, cached_setup, make_user, subagents_on, upstream):
    subagents_on()
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    upstream.queue(reply.text("It leaves at 06:40.", match=reply.answering("find the ferry time")))
    ask(
        page,
        upstream,
        "ask a helper about the ferry",
        calling("delegate_task", {"task": "find the ferry time"}),
        reply.text("The helper says 06:40."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = requests_of_chat(upstream, "good morning")
    assert "It leaves at 06:40." in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


@pytest.mark.regression
def test_a_timer_firing_into_the_open_chat_keeps_the_prefix(
    page_for, cached_setup, make_user, subagents_on, upstream
):
    subagents_on()
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    upstream.queue(reply.text("Time to go.", match=reply.answering(TIMER_PROMPT)))
    ask(
        page,
        upstream,
        "remind me to leave",
        calling("timer", {"prompt": TIMER_PROMPT, "at": "2s"}),
        reply.text("Timer set."),
    )
    expect_reply(page, "Time to go.")
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = requests_of_chat(upstream, "remind me to leave")
    assert reply.answering(TIMER_PROMPT)(requests[2]), "the timer's turn is not the third request"
    broken = first_break(requests)
    assert broken is None, (
        "#31568: the turn a timer started was sent the chat's finished system prompt with the "
        f"model's system prompt and attached knowledge added to it a second time: {broken}"
    )


@pytest.mark.regression
def test_a_background_sub_agent_report_keeps_the_prefix(
    page_for, cached_setup, make_user, subagents_on, upstream
):
    subagents_on(SUBAGENTS_BACKGROUND_ENABLED=True)
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    task = f"find the ferry time {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("It leaves at 06:40.", match=reply.answering(task), delay=3),
        reply.text("The helper says 06:40.", match=reply.answering("[ASYNC SUBAGENT COMPLETE")),
    )
    ask(
        page,
        upstream,
        "ask a helper about the ferry",
        calling("delegate_task", {"task": task, "background": True}),
        reply.text("A helper is on it."),
    )
    expect_reply(page, "The helper says 06:40.")
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = requests_of_chat(upstream, "ask a helper about the ferry")
    broken = first_break(requests)
    assert broken is None, (
        "#31568: the turn a background sub-agent's report started was sent the chat's finished "
        f"system prompt with the model's system prompt added to it a second time: {broken}"
    )


@pytest.fixture
def terminal_skill(open_terminal):
    """A skill saved in the terminal before the chat starts, so the skill list never changes."""
    directory = open_terminal.home / ".agents" / "skills" / "tide-tables"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: tide-tables\ndescription: Reading tide tables\n---\nRead high tide first.\n"
    )
    yield
    shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.regression
def test_the_create_skill_command_keeps_the_prefix(terminal_skill, terminal_chat, upstream):
    page = terminal_chat
    ask(page, upstream, "high tide is read from the first column", reply.text("Understood."))
    upstream.queue(reply.text("The skill is up to date.", match=reply.answering("skill")))
    chat_input(page).click()
    page.keyboard.type("/skills")
    page.get_by_role("button", name=re.compile("^Create skill")).click()
    expect_reply(page, "The skill is up to date.")
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert "view_skill" in {tool["function"]["name"] for tool in requests[0]["tools"]}
    broken = first_break(requests)
    assert broken is None, (
        "#31591: the Create skill command's message went to the model as the skill authoring "
        "prompt on "
        f"its own turn and as the typed command on the next, rewriting it: {broken}"
    )


def test_a_workspace_tool_picked_before_the_first_turn_only_appends(
    page_for, cached_setup, admin, make_user, upstream
):
    with python_tool(admin, TIDE_TOOL, name="Tide clock") as tool_id:
        page = page_for(make_user())
        page.goto(f"/?models={cached_setup.id}")
        turn_on_tool(page, "Tide clock")
        ask(page, upstream, "good morning", reply.text("Morning."))
        ask(
            page,
            upstream,
            "when is high tide here and in Oban?",
            calling_together(
                ("tide_time", {"harbour": "Portree"}), ("tide_time", {"harbour": "Oban"})
            ),
            reply.text("At noon in both."),
        )
        # the chat keeps its tool choice in a draft saved half a second after each send
        page.wait_for_function(
            "toolId => Object.keys(sessionStorage).some((key) =>"
            " key.startsWith('chat-input-') && sessionStorage[key].includes(toolId))",
            arg=tool_id,
        )
        page.reload()
        ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert "tide_time" in called_tools(requests[-1])
    assert "High tide in Oban is at noon." in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)
