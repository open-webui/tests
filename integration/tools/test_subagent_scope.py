"""Journey: what a sub-agent can reach is what the chat that started it can reach, and no more.

A sub-agent runs as the account that delegated, on the same model, and is offered what the parent
was: the workspace tools and external tool servers the chat uses, the skills the account may
read, the terminal of the chat, the knowledge attached to the model and the memories to read
(never to write). It leads with the model's own system prompt, runs the filters the person
switched on, and reads only the knowledge bases the account may read. It gets none of the
conversation before the task and no chat file unless the delegation names it. A second account's
stop request leaves a running sub-agent alone.

The scripted model delegates, the sub-agent's provider requests show what it was offered and
sent, and where a tool matters the sub-agent calls it and the test reads the result it got back.

A sub-agent's system prompt named each skill twice, since the parent's already-assembled
prompt carried the skill list the sub-agent builds again (open-webui/open-webui#31568); dev
ecbbff8af fixed that by handing the sub-agent the chat's own system messages to build on. A
sub-agent started in a folder chat was offered the tools that browse every knowledge base where
its parent has the folder's scoped ones; PR #31574 fixed that (open-webui/open-webui#31569). A
temporary chat that delegated left the sub-agent's chat stored on the server, where `Temporary
Chat` keeps nothing (the task tools are withheld from it for that reason, dev d2936c880); PR
#31573 fixed that (open-webui/open-webui#31567).

Every test here but the temporary chat one is red on dev 1c010b438: since de73bb830 a chat request
whose reply message is already stored in the chat, the way automations, sub-agents and timers
prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066). Apart from the skill test they pass on de73bb830^. With that refusal
removed in a backend copy of dev 1c010b438 the skill test passes three of three, and fails three
of three with ecbbff8af reverted as well.

Discriminates: passes on dev 015dbc861 apart from the skill test, which turns green in a backend
copy that strips the skill list from the parent's prompt; the folder knowledge test fails on dev
a5bc78300, before PR #31574, and the temporary chat test on dev 176d31d1d, before PR #31573. In
backend copies the rest turn red with their edit:
the sub-agent's tool ids, skill tool, terminal, filters or model dropped, the parent's system
prompt replaced (with the model dropped as well for the model's own prompt), the parent's prompt
put in its system prompt, all memory tools removed, every chat file passed on (or none), the
sub-agent run as the instance owner or as an admin, and the owner check of the chat stop endpoint
removed.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import time
import uuid

import pytest

from harness import backends
from harness import upstream as reply
from harness.chat import ask, send_message
from harness.knowledge_bases import add_text_file, knowledge_base, model_with_knowledge
from harness.listener import json_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS, python_tool
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
DELEGATION = {"delegate_task", "timer"}
KNOWLEDGE_TOOLS = {
    "list_knowledge",
    "list_knowledge_bases",
    "search_knowledge_bases",
    "query_knowledge_bases",
    "grep_knowledge_files",
    "search_knowledge_files",
    "query_knowledge_files",
    "view_knowledge_file",
    "view_file",
}
MEMORY_READERS = {"search_memories", "list_memories", "list_memory_paths", "read_memory_path"}
MEMORY_WRITERS = {"add_memory", "update_memory", "replace_memory_content", "delete_memory"}
DEFAULT_PERMISSIONS = ("/api/v1/users/default/permissions", "/api/v1/users/default/permissions")
SHOUTING_TOOL = '''
class Tools:
    def shout(self, text: str) -> str:
        """Shout the text."""
        return text.upper()
'''
TOGGLE_FILTER = """
class Filter:
    def __init__(self):
        self.toggle = True

    async def inlet(self, body):
        body["messages"][-1]["content"] += " FILTER-MARK"
        return body
"""


@pytest.fixture
def subagents_on(admin, preserve):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def sub_requests(upstream, task: str) -> list[dict]:
    return [r for r in upstream.chat_requests() if reply.answering(task)(r)]


def system_text(request: dict) -> str:
    return "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")


def delegating(upstream, prompt: str, task: str, *sub_steps: reply.Reply, **arguments) -> None:
    """Script the delegation, the sub-agent's steps (the last one an answer) and the wrap-up."""
    upstream.queue(
        reply.tool_call(
            "delegate_task", {"task": task, **arguments}, match=reply.answering(prompt)
        ),
        *sub_steps,
        reply.text("Handed back.", match=reply.answering(prompt)),
    )


def sub_calls(task: str, name: str, arguments: dict) -> list[reply.Reply]:
    """The sub-agent calls one tool, then answers."""
    return [
        reply.tool_call(name, arguments, "sub_call", match=reply.answering(task)),
        reply.text("Finished.", match=reply.answering(task)),
    ]


def sub_answer(task: str) -> reply.Reply:
    return reply.text("Finished.", match=reply.answering(task))


def tool_result(request: dict) -> str:
    last = request["messages"][-1]
    assert last["role"] == "tool", last
    return last["content"]


def test_a_workspace_tool_the_chat_uses_runs_in_the_subagent(
    subagents_on, admin, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("shout the word")
    delegating(upstream, prompt, task, *sub_calls(task, "shout", {"text": "ahoy"}))
    with python_tool(admin, SHOUTING_TOOL) as tool_id, make_user().client() as client:
        ask(client, prompt, tool_ids=[tool_id])

    first, *_, last = sub_requests(upstream, task)
    assert "shout" in offered(first), "the sub-agent was not offered the chat's workspace tool"
    assert tool_result(last) == "AHOY"


def test_an_mcp_tool_server_the_chat_uses_runs_in_the_subagent(
    subagents_on, admin, make_user, preserve, upstream
):
    preserve(TOOL_SERVERS)
    person, server_id = make_user(), f"harbour_{secrets.token_hex(4)}"
    prompt, task = unique("hand this over"), unique("echo the word")
    delegating(upstream, prompt, task, *sub_calls(task, f"{server_id}_echo", {"text": "ahoy"}))
    with serving_mcp() as url, admin.client() as client:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        with person.client() as chat:
            ask(chat, prompt, tool_ids=[f"server:mcp:{server_id}"])

    first, *_, last = sub_requests(upstream, task)
    assert f"{server_id}_echo" in offered(first)
    assert "ahoy" in tool_result(last)


@pytest.fixture
def skills(admin):
    """`skills(grants)` creates a skill with those access grants; deleted again afterwards."""
    created: list[str] = []

    def create(grants: list[dict], content: str = "Tie a bowline.") -> str:
        skill_id = f"skill_{uuid.uuid4().hex[:8]}"
        with admin.client() as client:
            response = client.post(
                "/api/v1/skills/create",
                json={
                    "id": skill_id,
                    "name": f"Knots {skill_id}",
                    "description": "How to tie knots",
                    "content": content,
                    "meta": {},
                    "access_grants": grants,
                },
            )
        assert response.status_code == 200, response.text
        created.append(skill_id)
        return skill_id

    yield create
    with admin.client() as client:
        for skill_id in created:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def test_a_skill_the_account_may_read_is_listed_to_the_subagent_and_loads(
    subagents_on, skills, make_user, upstream
):
    instructions = unique("Tie a bowline")
    skill_id, hidden_id = skills([EVERYONE_READS], instructions), skills([], instructions)
    prompt, task = unique("hand this over"), unique("tie the boat up")
    delegating(upstream, prompt, task, *sub_calls(task, "view_skill", {"id": skill_id}))
    with make_user().client() as chat:
        ask(chat, prompt)

    first, *_, last = sub_requests(upstream, task)
    assert "view_skill" in offered(first)
    assert f"<id>{skill_id}</id>" in system_text(first)
    assert f"<id>{hidden_id}</id>" not in system_text(first)
    assert instructions in tool_result(last)


def test_a_subagent_is_told_of_each_skill_once(subagents_on, skills, make_user, upstream):
    skill_id = skills([EVERYONE_READS])
    prompt, task = unique("hand this over"), unique("tie the boat up")
    delegating(upstream, prompt, task, sub_answer(task))
    with make_user().client() as chat:
        ask(chat, prompt)

    [sub] = sub_requests(upstream, task)
    assert system_text(sub).count(f"<id>{skill_id}</id>") == 1, (
        "the skill list of the parent's system prompt and the sub-agent's own both name the skill"
    )


@contextlib.contextmanager
def preset(admin, **fields):
    """A model on the scripted provider that every account may chat with; yields its id."""
    model_id = f"preset-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Scope model",
        "meta": fields.pop("meta", {}),
        "params": fields.pop("params", {}),
        "access_grants": [EVERYONE_READS],
        **fields,
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        try:
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_model_with_attached_knowledge_gives_the_subagent_the_same_knowledge_tools(
    subagents_on, admin, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("look it up")
    delegating(upstream, prompt, task, sub_answer(task))
    with admin.client() as client, knowledge_base(client, unique("Notes")) as knowledge_id:
        add_text_file(client, knowledge_id, "notes.txt", "Herons nest in colonies.")
        with model_with_knowledge(client, knowledge_id) as model_id:
            with make_user().client() as chat:
                ask(chat, prompt, model=model_id)

    [parent] = [r for r in upstream.chat_requests() if reply.answering(prompt)(r)][:1]
    [sub] = sub_requests(upstream, task)
    assert offered(parent) & KNOWLEDGE_TOOLS
    assert "list_knowledge_bases" not in offered(parent)
    assert offered(sub) & KNOWLEDGE_TOOLS == offered(parent) & KNOWLEDGE_TOOLS


def test_a_subagent_lists_only_the_knowledge_bases_its_account_may_read(
    subagents_on, admin, make_user, upstream
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    prompt, task = unique("hand this over"), unique("list the libraries")
    delegating(upstream, prompt, task, *sub_calls(task, "list_knowledge_bases", {"count": 100}))
    with admin.client() as client:
        with (
            knowledge_base(client, f"Guide {tag}", [read_grant(person.id)]),
            knowledge_base(client, f"Ledger {tag}"),
            person.client() as chat,
        ):
            ask(chat, prompt)

    listed = {entry["name"] for entry in json.loads(tool_result(sub_requests(upstream, task)[-1]))}
    assert f"Guide {tag}" in listed
    assert f"Ledger {tag}" not in listed


def test_the_models_own_system_prompt_leads_the_subagents(subagents_on, admin, make_user, upstream):
    persona = unique("You are the lighthouse keeper")
    prompt, task = unique("hand this over"), unique("trim the lamp")
    delegating(upstream, prompt, task, sub_answer(task))
    with preset(admin, params={"system": persona}) as model_id, make_user().client() as chat:
        ask(chat, prompt, model=model_id)

    [sub] = sub_requests(upstream, task)
    assert system_text(sub).startswith(persona)


def test_a_filter_the_person_switched_on_runs_on_the_subagent_too(
    subagents_on, admin, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("check the tide")
    plain_prompt, plain_task = unique("hand this over"), unique("check the wind")
    delegating(upstream, prompt, task, sub_answer(task))
    delegating(upstream, plain_prompt, plain_task, sub_answer(plain_task))
    with installed_function(admin, TOGGLE_FILTER, is_global=True) as filter_id:
        with make_user().client() as chat:
            ask(chat, prompt, filter_ids=[filter_id])
            ask(chat, plain_prompt)

    [sub], [plain_sub] = sub_requests(upstream, task), sub_requests(upstream, plain_task)
    assert sub["messages"][-1]["content"].endswith("FILTER-MARK"), "the filter did not run"
    assert "FILTER-MARK" not in plain_sub["messages"][-1]["content"]


def test_a_subagent_reads_the_memories_but_cannot_write_them(subagents_on, make_user, upstream):
    person = make_user()
    memory = unique("prefers tea over coffee")
    prompt, task = unique("hand this over"), unique("recall the preference")
    delegating(upstream, prompt, task, *sub_calls(task, "list_memories", {}))
    with person.client() as chat:
        chat.post("/api/v1/memories/add", json={"content": memory}).raise_for_status()
        ask(chat, prompt, features={"memory": True})

    first, *_, last = sub_requests(upstream, task)
    assert MEMORY_READERS <= offered(first)
    assert not MEMORY_WRITERS & offered(first)
    assert memory in tool_result(last)


def _chat_file_model(admin):
    capabilities = {"file_upload": True, "file_context": False}
    return preset(admin, meta={"capabilities": capabilities})


@pytest.fixture
def two_files(admin):
    """(first id, second id): two text files the chat holds, both readable by everyone."""
    tag = uuid.uuid4().hex[:8]
    with admin.client() as client, knowledge_base(client, f"Files {tag}") as knowledge_id:
        first = add_text_file(client, knowledge_id, f"tides-{tag}.txt", "High tide at noon.")
        second = add_text_file(client, knowledge_id, f"winds-{tag}.txt", "Wind from the north.")
        yield first, second


def _chat_files(*file_ids: str) -> list[dict]:
    return [{"type": "file", "id": file_id, "name": file_id} for file_id in file_ids]


def test_only_the_files_a_delegation_names_reach_the_subagent(
    subagents_on, admin, two_files, upstream
):
    first, second = two_files
    prompt, task = unique("hand this over"), unique("read the tide table")
    delegating(upstream, prompt, task, *sub_calls(task, "list_chat_files", {}), file_ids=[first])
    with _chat_file_model(admin) as model_id, admin.client() as chat:
        ask(chat, prompt, model=model_id, chat_files=_chat_files(first, second))

    listed = json.loads(tool_result(sub_requests(upstream, task)[-1]))
    assert [entry["id"] for entry in listed] == [first]


def test_a_delegation_naming_no_file_gives_the_subagent_none(
    subagents_on, admin, two_files, upstream
):
    first, second = two_files
    prompt, task = unique("hand this over"), unique("answer from memory")
    delegating(upstream, prompt, task, sub_answer(task))
    with _chat_file_model(admin) as model_id, admin.client() as chat:
        ask(chat, prompt, model=model_id, chat_files=_chat_files(first, second))

    [parent] = [r for r in upstream.chat_requests() if reply.answering(prompt)(r)][:1]
    [sub] = sub_requests(upstream, task)
    assert "list_chat_files" in offered(parent)
    assert "list_chat_files" not in offered(sub)


def test_the_subagent_gets_none_of_the_conversation_before_its_task(
    subagents_on, make_user, upstream
):
    earlier, prompt, task = unique("earlier question"), unique("hand this over"), unique("do it")
    delegating(upstream, prompt, task, sub_answer(task))
    history = [
        {"role": "user", "content": earlier},
        {"role": "assistant", "content": unique("earlier answer")},
    ]
    with make_user().client() as chat:
        ask(chat, prompt, history=history)

    [sub] = sub_requests(upstream, task)
    assert [m["role"] for m in sub["messages"]] == ["system", "user"]
    assert earlier not in json.dumps(sub["messages"])
    assert prompt not in json.dumps(sub["messages"])


TERMINAL_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/files/view": {
            "post": {
                "operationId": "read_file",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}


def test_the_terminal_of_the_chat_is_offered_to_the_subagent_and_answers_it(
    subagents_on, make_user, preserve, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user(role="admin")
    prompt, task = unique("hand this over"), unique("read the log")
    delegating(upstream, prompt, task, *sub_calls(task, "read_file", {"path": "/work/log.txt"}))
    with serving_terminal() as server, person.client() as chat:
        server.route("GET", "/openapi.json", json_answer(TERMINAL_SPEC))
        server.route("POST", "/files/view", json_answer({"content": "LOG-LINE"}))
        connection = server.connection()
        configure_terminals(chat, connection)
        ask(chat, prompt, terminal_id=connection["id"])
        calls = server.requests_to("/files/view")

    first, *_, last = sub_requests(upstream, task)
    assert "read_file" in offered(first)
    assert len(calls) == 1
    assert "LOG-LINE" in tool_result(last)


def test_a_subagent_has_no_tool_the_account_may_not_use(
    subagents_on, admin, make_user, preserve, upstream
):
    preserve(DEFAULT_PERMISSIONS)
    with admin.client() as client:
        permissions = client.get(DEFAULT_PERMISSIONS[0]).json()
        permissions["features"]["notes"] = False
        client.post(DEFAULT_PERMISSIONS[1], json=permissions).raise_for_status()
    prompt, task = unique("hand this over"), unique("jot it down")
    delegating(upstream, prompt, task, sub_answer(task))
    with make_user().client() as chat:
        ask(chat, prompt)

    [parent] = [r for r in upstream.chat_requests() if reply.answering(prompt)(r)][:1]
    [sub] = sub_requests(upstream, task)
    assert "search_notes" not in offered(parent)
    assert "search_notes" not in offered(sub)


def _dispatch_handle(message: dict) -> dict:
    [output] = [item for item in message["output"] if item["type"] == "function_call_output"]
    return json.loads(output["output"][0]["text"])


def test_a_second_account_can_neither_open_nor_stop_the_subagent_of_another(
    admin, preserve, make_user, upstream
):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        body = {**current, "ENABLE_SUBAGENTS": True, "SUBAGENTS_BACKGROUND_ENABLED": True}
        client.post(SUBAGENTS[1], json=body).raise_for_status()
    owner, other = make_user(), make_user()
    prompt, task = unique("hand this over"), unique("watch the tide")
    upstream.queue(
        reply.tool_call(
            "delegate_task",
            {"task": task, "background": True},
            match=reply.answering(prompt),
        ),
        reply.text("Tide watched.", delay=3.0, match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("Report read.", match=lambda body: "ASYNC SUBAGENT COMPLETE" in str(body)),
    )
    with owner.client() as mine, other.client() as theirs:
        _, stored = ask(mine, prompt)
        sub_id = _dispatch_handle(stored)["subagent_chat_id"]

        opened = theirs.get(f"/api/v1/chats/{sub_id}")
        stopped = theirs.post(f"/api/tasks/chat/{sub_id}/stop")
        listed = theirs.get(f"/api/tasks/chat/{sub_id}").json()["task_ids"]
        still_running = mine.get(f"/api/tasks/chat/{sub_id}").json()["task_ids"]
        owners_view = mine.get(f"/api/v1/chats/{sub_id}")

        finished = wait_for_sub_answer(mine, sub_id)

    assert opened.status_code in (401, 404)
    assert stopped.status_code == 404
    assert listed == []
    assert still_running != [], "the other account's stop request ended the sub-agent"
    assert owners_view.status_code == 200
    assert finished["content"] == "Tide watched."


def wait_for_sub_answer(client, chat_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        answers = [m for m in messages.values() if m["role"] == "assistant" and m.get("done")]
        if answers:
            return answers[0]
        time.sleep(0.1)
    raise AssertionError("the sub-agent never finished")


def test_a_subagent_in_a_folder_chat_leads_with_the_folders_system_prompt(
    subagents_on, make_user, upstream
):
    folder_prompt = unique("Answer as the harbour master")
    prompt, task = unique("hand this over"), unique("log the arrivals")
    delegating(upstream, prompt, task, sub_answer(task))
    with make_user().client() as chat:
        folder = chat.post(
            "/api/v1/folders/",
            json={"name": unique("Harbour"), "data": {"system_prompt": folder_prompt}},
        )
        assert folder.status_code == 200, folder.text
        ask(chat, prompt, folder_id=folder.json()["id"])

    [sub] = sub_requests(upstream, task)
    assert system_text(sub).startswith(folder_prompt)


def test_a_subagent_in_a_folder_chat_gets_the_folders_knowledge_tools_its_parent_has(
    subagents_on, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("look it up")
    delegating(upstream, prompt, task, sub_answer(task))
    with make_user(role="admin").client() as chat:
        with knowledge_base(chat, unique("Harbour notes")) as knowledge_id:
            add_text_file(chat, knowledge_id, "notes.txt", "Herons nest in colonies.")
            attached = [{"type": "collection", "id": knowledge_id, "name": "Harbour notes"}]
            folder = chat.post(
                "/api/v1/folders/", json={"name": unique("Harbour"), "data": {"files": attached}}
            )
            assert folder.status_code == 200, folder.text
            ask(chat, prompt, folder_id=folder.json()["id"])

    [parent] = [r for r in upstream.chat_requests() if reply.answering(prompt)(r)][:1]
    [sub] = sub_requests(upstream, task)
    assert "list_knowledge_bases" not in offered(parent)
    assert offered(sub) & KNOWLEDGE_TOOLS == offered(parent) & KNOWLEDGE_TOOLS, (
        "the sub-agent was offered the tools that browse every knowledge base, not the folder's"
    )


def _stored_subagent_chats(instance, parent_chat_id: str) -> list[dict]:
    rows = backends.read_rows(
        instance,
        "SELECT CAST(chat AS TEXT) AS chat, CAST(meta AS TEXT) AS meta FROM chat "
        "WHERE CAST(meta AS TEXT) LIKE '%subagent%'",
    )
    return [
        json.loads(row["chat"])
        for row in rows
        if json.loads(row["meta"]).get("parent_chat_id") == parent_chat_id
    ]


def _delegate_from(client, upstream, chat_id: str | None, task: str, answer: str) -> str:
    """Send a delegating message and return once the parent has the sub-agent's result."""
    prompt = unique("hand this over")
    delegating(upstream, prompt, task, reply.text(answer, match=reply.answering(task)))
    turn = send_message(client, prompt, chat_id=chat_id)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        requests = [r for r in upstream.chat_requests() if reply.answering(prompt)(r)]
        if any(r["messages"][-1]["role"] == "tool" for r in requests):
            return turn.chat_id
        time.sleep(0.1)
    raise AssertionError("the delegation never came back to the parent")


def test_a_saved_chat_keeps_the_task_and_answer_of_its_subagent(
    subagents_on, instance, make_user, upstream
):
    task, answer = unique("look it up"), unique("Found it")
    with make_user().client() as client:
        chat_id = _delegate_from(client, upstream, None, task, answer)

    [stored] = _stored_subagent_chats(instance, chat_id)
    assert task in json.dumps(stored)
    assert answer in json.dumps(stored)


def test_a_temporary_chat_leaves_no_stored_subagent_chat_behind(
    subagents_on, instance, make_user, upstream
):
    task, answer = unique("look it up"), unique("Found it")
    chat_id = f"temporary:{uuid.uuid4().hex[:12]}"
    with make_user().client() as client:
        _delegate_from(client, upstream, chat_id, task, answer)

    assert _stored_subagent_chats(instance, chat_id) == [], (
        "a temporary chat's delegation left the task and the answer stored on the server"
    )
