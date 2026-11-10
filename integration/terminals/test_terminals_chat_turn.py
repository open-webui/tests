"""Journey: a chat's model works in the machine the picked terminal gives that chat.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). A person picks a terminal of
a Terminals orchestrator in the chat input and sends a message. The model is offered the tools
of the machine the terminal's scope gives this chat (one for everyone, one per user, per chat or
per model), started by this first use, and is told about that machine. Every command it calls
runs there. The terminal is checked again before each call, so a terminal unshared while the
reply is still running refuses the next command. A new chat starts in the working directory
chosen for it or set as its folder's default. The machine's own AGENTS.md and skills reach the
model, and only that machine's. A call the machine refuses is handed to the model as an error
and the reply goes on. A terminal with a machine per chat refuses a temporary chat, which has
none. An HTML file's preview gets a signed link that needs no sign-in, serves the machine's
files and nothing else, and cannot write.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7; each
mutation below, made in a backend copy, turns its test red. The machine's system prompt no
longer read fails the four scope tests (the model is not told about its machine), and the chat
and model left out of what the orchestrator is sent fails the per-chat and per-model ones; the
check before each call removed fails the unshare test (the second command runs); the chat's
directory never set on the machine fails the chosen-directory and missing-directory tests, and a
folder's defaults read as empty the folder test; AGENTS.md no longer read, or the machine's
skills dropped from the list, fails the AGENTS.md and skills test; the machine's answer replaced
by a fixed text fails the refused-call test; the saved-chat check removed fails the
temporary-chat test (the orchestrator's own refusal shows); the write check or the path check
removed from the preview link fails the preview test.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.backends import wait_until
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.socket_client import connected
from harness.terminals import execute, instance_of, output_of, share_template
from harness.upstream import MOCK_MODEL_ID, conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HOME = "/home/user"
CWD_LINE = "The current working directory is: "
SAVED_CHAT_NEEDED = "This terminal requires a saved chat"


@dataclasses.dataclass
class Exchange:
    """One message and what the provider was sent for it, request by request."""

    turn: ChatTurn
    message: dict
    requests: list[dict]

    def offered(self) -> dict[str, dict]:
        """The tools the model was offered with the message, by name."""
        return {tool["function"]["name"]: tool["function"] for tool in self.requests[0]["tools"]}

    def results(self) -> list[dict]:
        """What each of the model's tool calls gave it, oldest first."""
        last = conversation(self.requests[-1])
        return [json.loads(entry["content"]) for entry in last if entry["role"] == "tool"]

    def sent(self, role: str) -> list[str]:
        """What the model was sent under `role` with the message, context rounds included."""
        return [
            str(entry["content"]) for entry in self.requests[0]["messages"] if entry["role"] == role
        ]


def _scripted(upstream, *steps: reply.Reply) -> tuple[str, list]:
    """Queue the model's `steps` for a prompt of this test's own; returns the prompt."""
    prompt = f"harbour watch {uuid.uuid4().hex[:8]}"
    answering = reply.answering(prompt)
    upstream.queue(*(dataclasses.replace(step, match=answering) for step in steps))
    return prompt


def _requests_for(upstream, prompt: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]


def _ask(person, upstream, *steps: reply.Reply, **options) -> Exchange:
    """`person` sends a message, the model takes `steps`; a new chat unless `options` name one."""
    prompt = _scripted(upstream, *steps)
    with person.client() as client:
        turn = send_message(client, prompt, **options)
        message = wait_for_reply(client, turn)
    return Exchange(turn, message, _requests_for(upstream, prompt))


def _run(command: str) -> reply.Reply:
    return reply.tool_call("run_command", {"command": command, "wait": 10})


def _printed(result: dict) -> str:
    assert "output" in result, f"the command did not run: {result}"
    return "".join(entry["data"] for entry in result["output"])


def _prepare(person, terminal_id: str, command: str) -> None:
    """Run `command` in the person's machine, as they would in its shell."""
    with person.client() as client:
        done = execute(client, terminal_id, command)
    output_of(done)
    assert done.json()["exit_code"] == 0, f"preparing the machine failed: {done.json()}"


def _machine_of(person, terminal_id: str, **context) -> dict:
    with person.client() as client:
        return instance_of(client, terminal_id, **context)


@pytest.mark.parametrize("scope", [None, "user", "chat", "model"])
def test_the_models_command_runs_in_the_machine_the_terminals_scope_gives_the_chat(
    scope, make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person, scope=scope)
    tag = uuid.uuid4().hex[:8]
    name = f"tide-{tag}.txt"

    asked = _ask(
        person,
        upstream,
        _run(f"echo high-{tag} > ~/{name} && cat ~/{name}"),
        reply.text("Noted."),
        terminal_id=terminal_id,
    )

    context = {
        None: "default",
        "user": f"user:{person.id}",
        "chat": f"chat:{asked.turn.chat_id}",
        "model": f"model:{MOCK_MODEL_ID}",
    }[scope]
    assert "run_command" in asked.offered(), (
        f"the model was not offered the terminal's tools: {sorted(asked.offered())}"
    )
    assert asked.message.get("content") == "Noted.", f"the reply did not finish: {asked.message}"
    (result,) = asked.results()
    assert _printed(result).strip() == f"high-{tag}", (
        f"the model was told the command printed {_printed(result)!r}"
    )
    machines = terminals.instances()
    assert [machine["context_id"] for machine in machines] == [context], (
        f"the chat did not start the one machine its terminal gives it ({context}): {machines}"
    )
    assert terminals.holding(name) == {context: f"high-{tag}\n"}, (
        f"the command's file is in these machines: {terminals.holding(name)}"
    )
    host = terminals.hostname(machines[0]["id"])
    assert any(f'on host "{host}"' in system for system in asked.sent("system")), (
        f"the model was not told about the machine it works in ({host}): {asked.sent('system')}"
    )


def test_a_terminal_unshared_while_the_reply_runs_refuses_the_models_next_command(
    admin, make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]
    before, after = f"before-{tag}.txt", f"after-{tag}.txt"
    unshared = threading.Event()
    prompt = _scripted(
        upstream,
        _run(f"echo first > ~/{before}"),
        dataclasses.replace(_run(f"echo second > ~/{after}"), hold=unshared),
        reply.text("Stopped."),
    )

    with person.client() as client:
        turn = send_message(client, prompt, terminal_id=terminal_id)
        try:
            first_ran = wait_until(lambda: len(_requests_for(upstream, prompt)) == 2, 30)
            with admin.client() as sharing:
                share_template(sharing, terminal_id)
        finally:
            unshared.set()
        message = wait_for_reply(client, turn)

    assert first_ran, "the model's first command never came back to it"
    assert terminals.holding(before) == {f"user:{person.id}": "first\n"}, (
        f"the command before the unshare did not run in the person's machine: {message}"
    )
    first, second = Exchange(turn, message, _requests_for(upstream, prompt)).results()
    assert "error" not in first, f"the command before the unshare failed: {first}"
    assert "Terminal access denied" in str(second.get("error")), (
        f"the model was not told the terminal is no longer its to use: {second}"
    )
    assert terminals.holding(after) == {}, (
        f"a command ran in a terminal unshared before it: {terminals.holding(after)}"
    )
    assert message.get("content") == "Stopped.", f"the reply did not go on: {message}"


def test_a_new_chat_starts_in_the_working_directory_chosen_for_it(
    make_user, shared_terminal, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    chosen = f"{HOME}/project-{uuid.uuid4().hex[:8]}"
    _prepare(person, terminal_id, f"mkdir {chosen}")

    asked = _ask(
        person,
        upstream,
        _run("pwd"),
        reply.text("Looked."),
        terminal_id=terminal_id,
        terminal_cwd=chosen,
    )

    (result,) = asked.results()
    assert _printed(result).strip() == chosen, (
        f"the model's command ran in {_printed(result)!r}, not the chosen directory"
    )
    description = asked.offered()["run_command"]["description"]
    assert description.endswith(f"{CWD_LINE}{chosen}"), (
        f"the model was not told it works in the chosen directory: {description[-200:]!r}"
    )
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{asked.turn.chat_id}").json()["chat"]
    assert (stored.get("terminal_id"), stored.get("working_directory")) == (terminal_id, chosen), (
        f"the chat does not keep its terminal and directory: {stored.get('terminal_id')}, "
        f"{stored.get('working_directory')}"
    )


def test_a_working_directory_the_machine_does_not_have_refuses_the_new_chat(
    make_user, shared_terminal, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    _prepare(person, terminal_id, "true")

    asked = _ask(
        person,
        upstream,
        reply.text("Looked."),
        terminal_id=terminal_id,
        terminal_cwd=f"{HOME}/missing-{uuid.uuid4().hex[:8]}",
    )

    error = str((asked.message.get("error") or {}).get("content"))
    assert "Working directory is unavailable" in error, (
        f"a chat in a missing directory did not say so: {asked.message}"
    )
    assert asked.requests == [], "the model was asked although the chat was refused"


def test_a_folders_terminal_and_working_directory_apply_to_its_new_chat(
    make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]
    default = f"{HOME}/repo-{tag}"
    _prepare(person, terminal_id, f"mkdir {default}")
    with person.client() as client:
        folder = client.post(
            "/api/v1/folders/",
            json={
                "name": f"Repo {tag}",
                "data": {"terminal_id": terminal_id, "working_directory": default},
            },
        )
    assert folder.status_code == 200, f"creating the folder failed: {folder.text}"

    asked = _ask(
        person, upstream, _run("pwd"), reply.text("Looked."), folder_id=folder.json()["id"]
    )

    (result,) = asked.results()
    assert _printed(result).strip() == default, (
        f"the folder's new chat ran in {_printed(result)!r}, not the folder's directory"
    )
    assert [machine["context_id"] for machine in terminals.instances()] == [f"user:{person.id}"], (
        f"the folder's chat did not use the person's own machine: {terminals.instances()}"
    )


def test_the_machines_own_agents_md_and_skills_reach_the_model(
    make_user, shared_terminal, upstream
):
    keeper, other = make_user(), make_user()
    terminal_id = shared_terminal(keeper, other)
    tag = uuid.uuid4().hex[:8]
    rule, skill = f"Greet with ahoy-{tag}", f"knots-{tag}"
    manual = f"---\\nname: {skill}\\ndescription: Tying knots {tag}\\n---\\nUse a bowline.\\n"
    _prepare(
        keeper,
        terminal_id,
        f"echo '{rule}' > ~/AGENTS.md && mkdir -p ~/.agents/skills/{skill} && "
        f"printf -- '{manual}' > ~/.agents/skills/{skill}/SKILL.md",
    )

    kept = _ask(keeper, upstream, reply.text("Ahoy."), terminal_id=terminal_id)
    others = _ask(other, upstream, reply.text("Hello."), terminal_id=terminal_id)

    assert f"# AGENTS.md\n\n{rule}\n" in kept.sent("user"), (
        f"the machine's AGENTS.md did not reach the model: {kept.sent('user')}"
    )
    listed = "".join(kept.sent("tool"))
    assert f"<id>terminal:{skill}</id>" in listed and f"Tying knots {tag}" in listed, (
        f"the machine's skill was not listed for the model: {listed}"
    )
    assert f"{HOME}/.agents/skills/{skill}/SKILL.md" in listed, (
        f"the model was not told where the skill is in the machine: {listed}"
    )
    assert "run_command" in others.offered(), (
        f"the other person's chat got no terminal: {sorted(others.offered())}"
    )
    seen = json.dumps(others.requests[0]["messages"])
    assert rule not in seen and skill not in seen, (
        f"another person's chat was sent the keeper's AGENTS.md or skill: {seen}"
    )


def test_a_call_the_machine_refuses_is_handed_to_the_model_and_the_reply_goes_on(
    make_user, shared_terminal, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]

    asked = _ask(
        person,
        upstream,
        reply.tool_call("read_file", {"path": f"{HOME}/missing-{tag}.txt"}),
        _run(f"echo still-here-{tag}"),
        reply.text("Carried on."),
        terminal_id=terminal_id,
    )

    refused, ran = asked.results()
    assert "404" in str(refused.get("error")) and "File not found" in str(refused.get("error")), (
        f"the model was not told the file is missing: {refused}"
    )
    assert _printed(ran).strip() == f"still-here-{tag}", (
        f"the command after the refused call printed {_printed(ran)!r}"
    )
    assert asked.message.get("content") == "Carried on.", (
        f"the reply did not go on after the refused call: {asked.message}"
    )
    assert not asked.message.get("error"), f"the refused call failed the reply: {asked.message}"


def test_a_per_chat_terminal_refuses_a_temporary_chat_and_serves_a_saved_one(
    make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="chat")
    unsaved = _scripted(upstream, reply.text("Should not be asked."))

    with connected(person) as tab, person.client() as client:
        chat_id = tab.temporary_chat_id()
        send_message(client, unsaved, terminal_id=terminal_id, chat_id=chat_id, session_id=tab.sid)
        refused = tab.wait_for(chat_id, "chat:message:error")
    started_for_it = terminals.instances()
    saved = _ask(person, upstream, _run("echo saved"), reply.text("Ran."), terminal_id=terminal_id)

    assert refused["data"]["error"] == {"content": SAVED_CHAT_NEEDED}, (
        f"the temporary chat was not told why its terminal cannot be used: {refused}"
    )
    assert _requests_for(upstream, unsaved) == [], "the model was asked for the refused chat"
    assert started_for_it == [], f"a temporary chat started a machine: {started_for_it}"
    assert _printed(saved.results()[0]).strip() == "saved", (
        f"the same terminal did not serve a saved chat: {saved.message}"
    )
    assert [machine["context_id"] for machine in terminals.instances()] == [
        f"chat:{saved.turn.chat_id}"
    ], f"the saved chat did not get its own machine: {terminals.instances()}"


def test_a_file_preview_link_serves_the_machines_files_without_sign_in_and_read_only(
    instance, make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]
    page = f"<h1>Tide table {tag}</h1>"
    _prepare(person, terminal_id, f"echo '{page}' > ~/tides-{tag}.html")

    with person.client() as client:
        preview = client.post(f"/api/v1/terminals/{terminal_id}/files/preview")
    assert preview.status_code == 200, f"no preview link: {preview.status_code} {preview.text}"
    link = preview.json()["url"]
    file_link = f"{link}home/user/tides-{tag}.html"
    with httpx.Client(base_url=instance.base_url, timeout=60.0) as signed_out:
        served = signed_out.get(file_link)
        overwritten = signed_out.put(file_link, content=b"<h1>Overwritten</h1>")
        deleted = signed_out.delete(file_link)
        outside = signed_out.get(f"{link}%2e%2e/%2e%2e/etc/hostname")
        still_served = signed_out.get(file_link)

    assert served.status_code == 200, f"the link did not serve the file: {served.status_code}"
    assert served.text.strip() == page, f"the link served something else: {served.text!r}"
    for write in (overwritten, deleted):
        assert (write.status_code, write.json().get("detail")) == (
            405,
            "File preview tokens are read-only",
        ), f"the link took a {write.request.method}: {write.status_code} {write.text}"
    assert (outside.status_code, outside.json().get("detail")) == (400, "Invalid file path"), (
        f"the link reached outside the files it serves: {outside.status_code} {outside.text}"
    )
    assert still_served.text.strip() == page, (
        f"the file changed through a read-only link: {still_served.text!r}"
    )
    machine = _machine_of(person, terminal_id)
    assert terminals.read_file(machine["id"], f"tides-{tag}.html") == f"{page}\n", (
        "the machine's file is not what the link served"
    )
