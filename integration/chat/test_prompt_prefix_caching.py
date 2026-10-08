"""Journey: with the page's cache-optimal setup, every request to the provider only appends.

The Prompt Caching docs page promises that a model set up as its checklist says (a static system
prompt, File Upload on, File Context and Citations off, builtin tools on, native function
calling, the memory system context off) sends requests whose start never changes: each request
to the provider repeats the tool list and every message of the one before it byte for byte and
only adds to the end, so the provider's prefix cache is never invalidated.

Each test drives one chat through a feature family over several turns, the way the web client
sends them, and checks every consecutive pair of requests the provider received, tool rounds
included: plain turns, single and multi-step tool rounds, parallel tool calls, reasoning, the
chat file tools, knowledge bases, notes and chats attached to the chat, the model's knowledge,
the memory tools, unchanged memories kept in the system prompt, skills, a sub-agent, a
background sub-agent's report, a timer, web search and fetch, an image a tool returned, a real
Open Terminal connected from the first turn, time passing and a change to settings outside the
prefix. The controls at the end take the breakers
the page names (Citations on, File Context on, the memory system context with a memory
changing, web search switched on mid-chat, an earlier message edited) and show the check goes
red on each.

Two tests stay red. The turn a timer or a background sub-agent's report starts is sent the
chat's finished system prompt as its own and has the model's system prompt and attached knowledge
added to it again, so the system message changes on that turn and changes back on the next;
issue open-webui/open-webui#31568 reports the same reuse for the sub-agent's own prompt (fix PR
#31579 open). A model that called a second tool straight after the first, with no text in
between, had both calls folded into one assistant message on the next round, so that round
rewrote the assistant message the previous request ended with; PR #31593 fixed that
(open-webui/open-webui#31588).

`test_a_sub_agent_only_appends` is red on dev 93fc3fcb7: since de73bb830 a chat request whose reply
message is already stored in the chat, the way automations, sub-agents and timers prepare their
reply, is refused with 409 and the reply is never written (open-webui/open-webui#32066).

Discriminates: passes on dev 015dbc861 apart from those two, which fail there; the multi-step
tool loop test fails on dev a5bc78300, before PR #31593. In backend copies, a clock value added
to the model's system prompt and the tool list shuffled per request each turned all twenty
promise tests red; with no stored system prompt handed to a timer or report, all of them passed.
The controls pass on all four.
"""

from __future__ import annotations

import itertools
import time
import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.listener import text_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.prompt_caching import (
    ADMIN_CONFIG,
    CAPABILITIES,
    assert_append_only,
    cache_optimal_model,
    prefix_break,
    turn_off_memory_system_context,
)
from harness.python_tools import EVERYONE_READS
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.upstream import Reply
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CALL_NUMBERS = itertools.count(1)
NOTES = ("berths.txt", "Berth 3 is reserved for the pilot boat.\nBerth 5 is free on Mondays.\n")
SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TIMER_PROMPT = "Time to leave for the ferry."


@pytest.fixture
def cached_setup(admin, preserve):
    """The page's setup: its model, and the memory system context switched off."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin) as model:
        yield model


def turn_on_memory_system_context(admin) -> None:
    """Step 5's other option: memories stay in the system prompt and do not change."""
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, "ENABLE_MEMORY_SYSTEM_CONTEXT": True})
    assert saved.status_code == 200, saved.text


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


def calling(name: str, arguments: dict, **options) -> Reply:
    return reply.tool_call(name, arguments, call_id=f"call_{next(CALL_NUMBERS)}", **options)


def calling_together(*calls: tuple[str, dict]) -> Reply:
    return Reply(tool_calls=[calling(name, arguments).tool_calls[0] for name, arguments in calls])


class Conversation:
    """One chat on a model, continued turn after turn the way the web client continues it."""

    def __init__(self, client, upstream, model_id: str, **options):
        self.client, self.upstream, self.model_id = client, upstream, model_id
        self.options = options
        self.chat_id: str | None = None
        self.parent_id: str | None = None
        self.opening: str | None = None

    def say(self, text: str, *replies: Reply, **options) -> dict:
        prompt = f"{text} [{uuid.uuid4().hex[:6]}]"
        self.opening = self.opening or prompt
        for scripted in replies or (reply.text(f"reply to {text}"),):
            scripted.match = reply.answering(prompt)
            self.upstream.queue(scripted)
        turn, message = ask(
            self.client,
            prompt,
            model=self.model_id,
            chat_id=self.chat_id,
            parent_id=self.parent_id,
            **{**self.options, **options},
        )
        assert message.get("done") and not message.get("error"), message
        self.chat_id, self.parent_id = turn.chat_id, turn.assistant_message_id
        return message

    def requests(self) -> list[dict]:
        """The chat's own requests to the provider, without a sub-agent's."""
        return [
            body
            for body in self.upstream.chat_requests()
            if body.get("stream") and self._opens_with_this_chat(body)
        ]

    def _opens_with_this_chat(self, body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        return bool(users) and self.opening in str(users[0].get("content"))


def upload(client, name: str, text: str) -> dict:
    """Upload a file as the chat input does and return the item the client attaches."""
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (name, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    stored = uploaded.json()
    return {
        "type": "file",
        "id": stored["id"],
        "url": f"/api/v1/files/{stored['id']}",
        "name": name,
        "content_type": "text/plain",
        "size": len(text),
        "status": "uploaded",
    }


def wait_for_answer_to(client, chat_id: str, prompt: str, timeout: float = 30.0) -> str:
    """The id of the finished reply to the internal message that starts with `prompt`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        for message in messages.values():
            if message.get("role") == "user" and str(message.get("content")).startswith(prompt):
                answer = messages.get((message.get("childrenIds") or [""])[0], {})
                if answer.get("done"):
                    return answer["id"]
        time.sleep(0.3)
    raise AssertionError(f"nothing ever answered {prompt!r} in the chat")


def offered_tools(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def tool_results(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "tool"
    )


def first_break(requests: list[dict]) -> str | None:
    for earlier, later in zip(requests, requests[1:]):
        broken = prefix_break(earlier, later)
        if broken:
            return broken
    return None


# --- the promise, one feature family at a time -------------------------------------------


def test_plain_turns_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        for text in ("good morning", "when does the ferry leave?", "and on sundays?", "thanks"):
            chat.say(text)

    assert_append_only(chat.requests())


def test_single_tool_rounds_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("hello")
        chat.say("what time is it?", calling("get_current_timestamp", {}), reply.text("Early."))
        chat.say(
            "and a week ago?",
            calling("calculate_timestamp", {"weeks_ago": 1}),
            reply.text("Last week."),
        )
        chat.say("thanks")

    assert_append_only(chat.requests())


def test_a_multi_step_tool_loop_only_appends(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("hello")
        chat.say(
            "how many days since last friday?",
            calling("get_current_timestamp", {}),
            calling("calculate_timestamp", {"days_ago": 3}),
            reply.text("Three days."),
        )
        chat.say("thanks")

    broken = first_break(chat.requests())
    assert broken is None, (
        "the second tool round folded its call into the assistant message the first round ended "
        f"with, rewriting it (#31588): {broken}"
    )


def test_tool_rounds_with_text_between_them_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say(
            "how many days since last friday?",
            calling("get_current_timestamp", {}, content="Checking the clock."),
            calling("calculate_timestamp", {"days_ago": 3}, content="Now the date."),
            reply.text("Three days."),
        )
        chat.say("thanks")

    assert_append_only(chat.requests())


def test_parallel_tool_calls_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say(
            "now and a week ago?",
            calling_together(
                ("get_current_timestamp", {}), ("calculate_timestamp", {"weeks_ago": 1})
            ),
            reply.text("Both done."),
        )
        chat.say("thanks")

    assert_append_only(chat.requests())


def test_reasoning_replies_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("think first", reply.text("Thought it over.", reasoning="weighing it up"))
        chat.say(
            "check the clock",
            calling("get_current_timestamp", {}, reasoning="I need the time"),
            reply.text("Done.", reasoning="the clock says so"),
        )
        chat.say("thanks")

    assert_append_only(chat.requests())


def test_the_chat_file_tools_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        attached = upload(client, *NOTES)
        chat = Conversation(client, upstream, cached_setup.id, files=[attached])
        chat.say(
            "which files do I have?",
            calling("list_chat_files", {}),
            reply.text("berths.txt"),
            files=None,
            chat_files=[attached],
        )
        chat.say(
            "which berth is free?",
            calling("query_chat_files", {"query": "free berth"}),
            reply.text("Berth 5 (berths.txt)."),
        )
        chat.say(
            "find the pilot boat",
            calling("grep_chat_files", {"pattern": "pilot"}),
            reply.text("Berth 3 (berths.txt)."),
        )
        chat.say(
            "read me the file",
            calling("view_file", {"file_id": attached["id"]}),
            reply.text("Read it (berths.txt)."),
        )

    requests = chat.requests()
    assert {"list_chat_files", "query_chat_files", "grep_chat_files", "view_file"} <= (
        offered_tools(requests[0])
    )
    assert tool_results(requests[-1]).count("Berth 5 is free") >= 2, tool_results(requests[-1])
    assert_append_only(requests)


def test_knowledge_tools_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say(
            "when does the ferry leave?",
            calling("query_knowledge_files", {"query": "ferry"}),
            reply.text("06:40 (handbook.txt)."),
        )
        chat.say(
            "show me the handbook",
            calling("view_knowledge_file", {"file_id": cached_setup.handbook_file_id}),
            reply.text("Here it is (handbook.txt)."),
        )
        chat.say(
            "anything about piers?",
            calling("grep_knowledge_files", {"pattern": "pier"}),
            reply.text("Pier 7 (handbook.txt)."),
        )

    assert tool_results(chat.requests()[-1]).count("pier 7") >= 3, tool_results(chat.requests()[-1])
    assert_append_only(chat.requests())


def test_chat_attached_knowledge_notes_and_chats_only_append(cached_setup, make_user, upstream):
    person = make_user(role="admin")  # a user may not create knowledge by default
    with person.client() as client, knowledge_base(client, "Timetables") as knowledge_id:
        add_text_file(client, knowledge_id, "buses.txt", "The harbour bus leaves at 07:15.\n")
        note = client.post(
            "/api/v1/notes/create",
            json={"title": "Packing list", "data": {"content": {"md": "rain jacket, tickets"}}},
        )
        assert note.status_code == 200, note.text
        earlier_chat, _ = ask(client, "an earlier trip", model=cached_setup.id)
        attachments = [
            {"type": "collection", "id": knowledge_id, "name": "Timetables"},
            {"type": "note", "id": note.json()["id"], "name": "Packing list"},
            {"type": "chat", "id": earlier_chat.chat_id, "name": "An earlier trip"},
        ]
        chat = Conversation(client, upstream, cached_setup.id, files=attachments)
        chat.say(
            "when does the bus leave?",
            calling("query_knowledge_files", {"query": "bus"}),
            reply.text("07:15 (buses.txt)."),
            files=None,
            chat_files=attachments,
        )
        chat.say(
            "what should I pack?",
            calling("view_note", {"note_id": note.json()["id"]}),
            reply.text("A rain jacket."),
        )
        chat.say(
            "what did we say last time?",
            calling("view_chat", {"chat_id": earlier_chat.chat_id}),
            reply.text("We talked about a trip."),
        )

    requests = chat.requests()
    results = tool_results(requests[-1])
    assert "07:15" in results and "rain jacket" in results and "earlier trip" in results, results
    assert_append_only(requests)


def test_memory_tools_only_append(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id, features={"memory": True})
        chat.say("hello")
        chat.say(
            "remember that I take the early ferry",
            calling("add_memory", {"content": "takes the early ferry"}),
            reply.text("Noted."),
        )
        chat.say(
            "what do you know about me?",
            calling("search_memories", {"query": "ferry"}),
            reply.text("You take the early ferry."),
        )
        chat.say("thanks")

    assert {"add_memory", "search_memories"} <= offered_tools(chat.requests()[0])
    assert "takes the early ferry" in tool_results(chat.requests()[-1])
    assert_append_only(chat.requests())


def test_unchanged_memories_in_the_system_context_only_append(
    cached_setup, admin, make_user, upstream
):
    turn_on_memory_system_context(admin)
    with make_user().client() as client:
        for memory in ("lives by the harbour", "takes the early ferry"):
            client.post("/api/v1/memories/add", json={"content": memory}).raise_for_status()
        chat = Conversation(client, upstream, cached_setup.id, features={"memory": True})
        for text in ("good morning", "what about the weather?", "and the tides?"):
            chat.say(text)

    requests = chat.requests()
    assert "takes the early ferry" in requests[0]["messages"][0]["content"]
    assert_append_only(requests)


def test_skills_only_append(cached_setup, make_user, upstream):
    author = make_user(role="admin")
    skill_id = f"tide-tables-{uuid.uuid4().hex[:6]}"
    with author.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": skill_id,
                "name": "Tide tables",
                "description": "Reading tide tables",
                "content": "Read the high tide column first.",
            },
        )
        assert created.status_code == 200, created.text
        chat = Conversation(client, upstream, cached_setup.id, skill_ids=[skill_id])
        chat.say("hello")
        chat.say(
            "when is high tide?",
            calling("view_skill", {"id": skill_id}),
            reply.text("At noon."),
        )
        chat.say("thanks")

    assert "view_skill" in offered_tools(chat.requests()[0])
    assert "Read the high tide column first." in tool_results(chat.requests()[-1])
    assert_append_only(chat.requests())


def test_a_sub_agent_only_appends(cached_setup, make_user, subagents_on, upstream):
    subagents_on()
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("hello")
        upstream.queue(
            reply.text("It leaves at 06:40.", match=reply.answering("find the ferry time"))
        )
        chat.say(
            "ask a helper about the ferry",
            calling("delegate_task", {"task": "find the ferry time"}),
            reply.text("The helper says 06:40."),
        )
        chat.say("thanks")

    assert "delegate_task" in offered_tools(chat.requests()[0])
    assert "It leaves at 06:40." in tool_results(chat.requests()[-1])
    assert_append_only(chat.requests())


@pytest.mark.slow
def test_web_search_and_fetch_only_append(instance_with, preserve, listener):
    fetching = instance_with(LOCAL_WEB_FETCH)
    preserve(RETRIEVAL_CONFIG, "admin_config", on=fetching)
    page = f"{listener.base_url}/tides"
    listener.route(
        "GET", "/tides", text_answer("<html><body><p>High tide at noon.</p></body></html>")
    )
    fetching_admin = admin_of(fetching)
    with fetching_admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, [page]))
    turn_off_memory_system_context(fetching_admin)
    with (
        cache_optimal_model(fetching_admin) as model,
        create_user(fetching).client() as client,
    ):
        chat = Conversation(client, fetching.upstream, model.id, features={"web_search": True})
        chat.say("hello")
        chat.say(
            "search the tides", calling("search_web", {"query": "tides"}), reply.text("Found.")
        )
        chat.say("read that page", calling("fetch_url", {"url": page}), reply.text("Noon."))
        chat.say("thanks")

    requests = chat.requests()
    assert {"search_web", "fetch_url"} <= offered_tools(requests[0])
    assert "High tide at noon." in tool_results(requests[-1])
    assert_append_only(requests)


def test_a_tool_image_only_appends(cached_setup, admin, make_user, preserve, upstream):
    preserve(TOOL_SERVERS)
    person = make_user()
    server_id = f"camera_{uuid.uuid4().hex[:8]}"
    with serving_mcp(media=True) as url, person.client() as client:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as admin_client:
            saved = admin_client.post(
                TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]}
            )
            assert saved.status_code == 200, saved.text
        chat = Conversation(client, upstream, cached_setup.id, tool_ids=[f"server:mcp:{server_id}"])
        chat.say("hello")
        chat.say(
            "show me the harbour camera",
            calling(f"{server_id}_snapshot", {}),
            reply.text("Two boats."),
        )
        chat.say("thanks")

    assert "image_url" in str(chat.requests()[2]["messages"][-1]), (
        "the image never reached the model"
    )
    assert_append_only(chat.requests())


def test_a_timer_only_appends(cached_setup, make_user, subagents_on, upstream):
    subagents_on()
    upstream.queue(reply.text("Time to go.", match=reply.answering(TIMER_PROMPT)))
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say(
            "remind me to leave",
            calling("timer", {"prompt": TIMER_PROMPT, "at": "2s"}),
            reply.text("Timer set."),
        )
        chat.parent_id = wait_for_answer_to(client, chat.chat_id, TIMER_PROMPT)
        chat.say("thanks")

    requests = chat.requests()
    assert reply.answering(TIMER_PROMPT)(requests[2]), "the timer's turn is not the third request"
    broken = first_break(requests)
    assert broken is None, (
        "the turn a timer started was sent the chat's finished system prompt with the model's "
        f"system prompt and attached knowledge added to it a second time: {broken}"
    )


def test_a_background_sub_agent_report_only_appends(
    cached_setup, make_user, subagents_on, upstream
):
    subagents_on(SUBAGENTS_BACKGROUND_ENABLED=True)
    task = f"find the ferry time {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("It leaves at 06:40.", match=reply.answering(task)),
        reply.text("The helper says 06:40.", match=reply.answering("[ASYNC SUBAGENT COMPLETE")),
    )
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say(
            "ask a helper about the ferry",
            calling("delegate_task", {"task": task, "background": True}),
            reply.text("A helper is on it."),
        )
        chat.parent_id = wait_for_answer_to(client, chat.chat_id, "[ASYNC SUBAGENT COMPLETE")
        chat.say("thanks")

    requests = chat.requests()
    assert reply.answering("[ASYNC SUBAGENT COMPLETE")(requests[2]), [
        str(request["messages"][-1])[:200] for request in requests
    ]
    broken = first_break(requests)
    assert broken is None, (
        "the turn a background sub-agent's report started was sent the chat's finished system "
        f"prompt with the model's system prompt added to it a second time: {broken}"
    )


def test_an_open_terminal_only_appends(
    cached_setup, admin, make_user, preserve, open_terminal, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    with person.client() as client:
        chat = Conversation(client, upstream, cached_setup.id, terminal_id=connection["id"])
        chat.say("hello")
        chat.say(
            "note the ferry time",
            calling("write_file", {"path": "ferry.txt", "content": "06:40 from pier 7\n"}),
            reply.text("Written."),
        )
        chat.say(
            "list my files",
            calling("run_command", {"command": "ls && cat ferry.txt", "wait": 10}),
            reply.text("ferry.txt is there."),
        )
        chat.say(
            "read it back",
            calling("read_file", {"path": "ferry.txt"}),
            reply.text("06:40 from pier 7."),
        )
        chat.say("thanks")

    requests = chat.requests()
    assert {"run_command", "write_file", "read_file"} <= offered_tools(requests[0])
    assert "You have access to a computer" in requests[0]["messages"][0]["content"]
    assert tool_results(requests[-1]).count("06:40 from pier 7") >= 2, tool_results(requests[-1])
    assert_append_only(requests)


def test_time_passing_between_turns_changes_nothing(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("first")
        time.sleep(1.5)  # a clock value in the prompt would move on between the turns
        chat.say("second")

    assert_append_only(chat.requests())


def test_a_change_outside_the_prefix_changes_nothing(cached_setup, admin, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("first")
        chat.say("second", params={"temperature": 0.2})
        with admin.client() as admin_client:
            stored = admin_client.get("/api/v1/models/model", params={"id": cached_setup.id}).json()
            stored["meta"]["description"] = "Answers questions about the harbour"
            updated = admin_client.post("/api/v1/models/model/update", json=stored)
            assert updated.status_code == 200, updated.text
        chat.say("third")

    requests = chat.requests()
    assert requests[1].get("temperature") == 0.2
    assert_append_only(requests)


# --- controls: the breakers the page names do turn the check red -------------------------


def test_the_prefix_check_names_the_first_changed_bytes():
    earlier = {"model": "m", "messages": [{"role": "system", "content": "stable"}]}
    later = {"model": "m", "messages": [{"role": "system", "content": "stab1e"}]}

    assert prefix_break(earlier, earlier) is None
    assert prefix_break(earlier, {**earlier, "messages": [*earlier["messages"], {}]}) is None
    broken = prefix_break(earlier, later)
    assert "messages[0] (system)" in broken and "stab1e" in broken


@pytest.fixture
def model_with(admin):
    """`model_with(**capabilities)` is a model set up like the page's, those capabilities apart."""
    created = []

    def build(**capabilities) -> str:
        model_id = f"breaker-{uuid.uuid4().hex[:8]}"
        form = {
            "id": model_id,
            "base_model_id": reply.MOCK_MODEL_ID,
            "name": model_id,
            "meta": {"capabilities": {**CAPABILITIES, **capabilities}},
            "params": {"system": "A static prompt.", "function_calling": "native"},
            "access_grants": [EVERYONE_READS],
        }
        with admin.client() as client:
            response = client.post("/api/v1/models/create", json=form)
            assert response.status_code == 200, response.text
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        created.append(model_id)
        return model_id

    yield build
    with admin.client() as client:
        for model_id in created:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_citations_on_rewrite_the_prefix(cached_setup, model_with, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, model_with(citations=True))
        chat.say(
            "read the handbook",
            calling("view_knowledge_file", {"file_id": cached_setup.handbook_file_id}),
            reply.text("Read."),
        )

    assert first_break(chat.requests()) is not None


def test_file_context_on_rewrites_the_prefix(model_with, make_user, upstream):
    with make_user().client() as client:
        attached = upload(client, *NOTES)
        chat = Conversation(client, upstream, model_with(file_context=True), files=[attached])
        chat.say("which berth is free?", files=None, chat_files=[attached])
        chat.say("and the pilot boat?")

    assert first_break(chat.requests()) is not None


def test_a_memory_change_with_the_system_context_on_rewrites_the_prefix(
    cached_setup, admin, make_user, upstream
):
    turn_on_memory_system_context(admin)
    with make_user().client() as client:
        client.post("/api/v1/memories/add", json={"content": "lives by the harbour"})
        chat = Conversation(client, upstream, cached_setup.id, features={"memory": True})
        chat.say("hello")
        client.post("/api/v1/memories/add", json={"content": "takes the early ferry"})
        chat.say("again")

    assert first_break(chat.requests()) is not None


def test_switching_web_search_on_mid_chat_rewrites_the_prefix(
    cached_setup, admin, preserve, make_user, upstream
):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, ENABLE_WEB_SEARCH=True)
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("hello")
        chat.say("look it up", features={"web_search": True})

    assert first_break(chat.requests()) is not None


def test_editing_an_earlier_message_rewrites_the_prefix(cached_setup, make_user, upstream):
    with make_user().client() as client:
        chat = Conversation(client, upstream, cached_setup.id)
        chat.say("hello")
        branch_point = chat.parent_id
        chat.say("when does the ferry leave?")
        chat.parent_id = branch_point  # the edit is a new version of the second message
        chat.say("when does the bus leave?")

    assert first_break(chat.requests()) is not None
