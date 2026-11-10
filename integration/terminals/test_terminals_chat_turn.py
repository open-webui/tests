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
files and nothing else, and cannot write. The model gets the two tools for the person's own
shell only while a shell is open in the chat, and they read and type in that shell. A picture
the model draws is copied into the machine. With the orchestrator down a message fails with the
reason, and one sent while the machine is still starting, past `AIOHTTP_CLIENT_TIMEOUT_TERMINAL`,
says so and works when sent again. A chat keeps the terminal of its first message: a message
naming another one is refused. A file uploaded for a chat is in the machine its model works in.

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
removed from the preview link fails the preview test; the shell tools offered whether or not a
shell is open fails the shell test; the copy to the terminal skipped fails the picture test; a
failed read of the machine's skills taken for no skills fails the orchestrator-down test (the
model answers without the terminal); the timeout no longer turned into its message fails the
still-starting test; the check of the chat's stored terminal removed fails the other-terminal
test (the command runs in the other terminal's machine); the chat left out of what the
orchestrator is sent fails the upload test (the upload is refused).

`test_the_still_starting_message_names_the_terminal_the_person_picked` is red on dev c15f28649:
the message names the admin's connection (finding
open-webui-terminal-timeout-names-the-connection.md, not filed yet).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import threading
import time
import uuid
from typing import Callable, Iterator

import httpx
import pytest
from websockets.sync.client import ClientConnection, connect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.backends import wait_until
from harness.channel_chat import serve_openai_images
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.image_engines import IMAGES_CONFIG, save_image_settings
from harness.socket_client import connected
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.terminals import (
    create_template,
    execute,
    instance_of,
    output_of,
    serving_terminals,
    share_template,
)
from harness.upstream import MOCK_MODEL_ID, conversation

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HOME = "/home/user"
CWD_LINE = "The current working directory is: "
SAVED_CHAT_NEEDED = "This terminal requires a saved chat"
SHELL_TOOLS = {"read_user_terminal", "send_user_terminal_input"}
STILL_STARTING = "timed out. It may still be starting. Please try again."
# a machine takes longer than this one second to start, and every later call far less
SHORT_TERMINAL_TIMEOUT = {"AIOHTTP_CLIENT_TIMEOUT_TERMINAL": "1"}


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


Matching = Callable[[str], Callable[[dict], bool]]


def _scripted(upstream, *steps: reply.Reply, matching: Matching = reply.answering) -> str:
    """Queue the model's `steps` for a prompt of this test's own; returns the prompt."""
    prompt = f"harbour watch {uuid.uuid4().hex[:8]}"
    upstream.queue(*(dataclasses.replace(step, match=matching(prompt)) for step in steps))
    return prompt


def _requests_for(upstream, prompt: str, matching: Matching = reply.answering) -> list[dict]:
    return [request for request in upstream.chat_requests() if matching(prompt)(request)]


def _ask(
    person, upstream, *steps: reply.Reply, matching: Matching = reply.answering, **options
) -> Exchange:
    """`person` sends a message, the model takes `steps`; a new chat unless `options` name one."""
    prompt = _scripted(upstream, *steps, matching=matching)
    with person.client() as client:
        turn = send_message(client, prompt, **options)
        message = wait_for_reply(client, turn)
    return Exchange(turn, message, _requests_for(upstream, prompt, matching))


def _after(person, chat_id: str) -> dict:
    """The options that send the next message of the chat `chat_id`."""
    with person.client() as client:
        history = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]
    return {"chat_id": chat_id, "parent_id": history["currentId"]}


def _error_of(message: dict) -> str:
    return str((message.get("error") or {}).get("content"))


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

    assert "Working directory is unavailable" in _error_of(asked.message), (
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


@contextlib.contextmanager
def _open_shell(instance, person, terminal_id: str, chat_id: str) -> Iterator[ClientConnection]:
    """The shell a person opens in the chat's Terminal tab, through the instance."""
    with person.client() as client:
        created = client.post(
            f"/api/v1/terminals/{terminal_id}/api/terminals", headers={"X-Session-Id": chat_id}
        )
    assert created.status_code == 200, f"opening a shell failed: {created.text}"
    address = instance.base_url.replace("http://", "ws://", 1)
    path = f"/api/v1/terminals/{terminal_id}/api/terminals/{created.json()['id']}"
    with connect(address + path, proxy=None, open_timeout=10, close_timeout=2) as shell:
        shell.send(json.dumps({"type": "auth", "token": person.token, "chat_id": chat_id}))
        yield shell


def _shell_shows(shell: ClientConnection, expected: str, timeout: float = 30.0) -> str:
    """What the shell printed until `expected` showed, or until the time ran out."""
    shown = b""
    deadline = time.monotonic() + timeout
    while expected.encode() not in shown and (remaining := deadline - time.monotonic()) > 0:
        try:
            frame = shell.recv(timeout=remaining)
        except TimeoutError:
            break
        shown += frame if isinstance(frame, bytes) else frame.encode()
    return shown.decode(errors="replace")


def test_the_model_reads_and_types_in_the_shell_the_person_has_open_in_the_chat(
    instance, make_user, shared_terminal, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]

    without_a_shell = _ask(person, upstream, reply.text("Hello."), terminal_id=terminal_id)
    chat_id = without_a_shell.turn.chat_id
    with _open_shell(instance, person, terminal_id, chat_id) as shell:
        # the sums show only in what the shell prints, never in what was typed
        shell.send(f"echo $((40+2))-{tag}\n".encode())
        typed_by_the_person = _shell_shows(shell, f"42-{tag}")
        with_a_shell = _ask(
            person,
            upstream,
            reply.tool_call("read_user_terminal", {}),
            reply.tool_call("send_user_terminal_input", {"input": f"echo $((50+5))-{tag}\n"}),
            reply.text("Typed."),
            terminal_id=terminal_id,
            **_after(person, chat_id),
        )
        typed_by_the_model = _shell_shows(shell, f"55-{tag}")

    assert "run_command" in without_a_shell.offered(), (
        f"the chat got no terminal: {sorted(without_a_shell.offered())}"
    )
    assert not SHELL_TOOLS & set(without_a_shell.offered()), (
        f"the model was offered a shell nobody has open: {sorted(without_a_shell.offered())}"
    )
    assert f"42-{tag}" in typed_by_the_person, (
        f"the person's shell did not run their command: {typed_by_the_person!r}"
    )
    assert SHELL_TOOLS <= set(with_a_shell.offered()), (
        f"the model was not offered the person's open shell: {sorted(with_a_shell.offered())}"
    )
    read, sent = with_a_shell.results()
    assert f"42-{tag}" in str(read.get("output")), (
        f"the model did not read what the person's shell shows: {read}"
    )
    assert "error" not in sent, f"the model's input was refused: {sent}"
    assert f"55-{tag}" in typed_by_the_model, (
        f"what the model typed did not run in the person's shell: {typed_by_the_model!r}"
    )


def test_a_picture_the_model_draws_is_copied_into_the_chats_machine(
    admin, listener, preserve, make_user, shared_terminal, terminals, upstream
):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))
    person = make_user()
    terminal_id = shared_terminal(person)

    asked = _ask(
        person,
        upstream,
        reply.tool_call("generate_image", {"prompt": "a lighthouse"}),
        _run("sha256sum ~/*.png"),
        reply.text("Drawn."),
        matching=reply.mentioning,
        terminal_id=terminal_id,
        features={"image_generation": True},
    )

    drawn, hashed = asked.results()
    assert len(drawn.get("images") or []) == 1, f"the model was told {drawn}"
    told = drawn["images"][0]
    copy = f"{HOME}/{told['id']}.png"
    assert told.get("terminal_path") == copy, f"the model was not told where the copy is: {told}"
    with person.client() as client:
        picture = client.get(told["url"])
    assert picture.status_code == 200, f"the chat's picture could not be read: {picture.text}"
    assert _printed(hashed).split() == [hashlib.sha256(picture.content).hexdigest(), copy], (
        f"the machine does not hold the picture the chat shows: {_printed(hashed)!r}"
    )
    machine = _machine_of(person, terminal_id)
    assert machine["context_id"] == f"user:{person.id}", f"the copy went to {machine}"
    assert (terminals.home(machine["id"]) / f"{told['id']}.png").exists(), (
        "the copy is not in the person's own machine"
    )


@pytest.mark.slow
def test_a_message_sent_while_the_orchestrator_is_down_fails_with_the_reason(
    admin, preserve, make_user, upstream
):
    person = make_user()
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminals() as stack:
        connection = stack.connection(config={"access_grants": [read_grant(person.id)]})
        with admin.client() as client:
            configure_terminals(client, connection)
            terminal_id = create_template(client, connection)["terminal_id"]
            share_template(client, terminal_id, read_grant(person.id))
        while_up = _ask(
            person, upstream, _run("echo up"), reply.text("Ran."), terminal_id=terminal_id
        )

    while_down = _ask(
        person,
        upstream,
        reply.text("Answered without the terminal."),
        terminal_id=terminal_id,
        **_after(person, while_up.turn.chat_id),
    )

    assert _printed(while_up.results()[0]).strip() == "up", (
        f"the terminal did not work while its orchestrator ran: {while_up.message}"
    )
    assert _error_of(while_down.message) == "Terminals is unavailable", (
        f"the message was not refused with the reason: {while_down.message}"
    )
    assert while_down.requests == [], "the model was asked without the chat's terminal"


def _reader_on_a_short_timeout(instance_with, preserve, terminals, name: str):
    """An instance that gives a terminal one second, with `name` shared to a person on it."""
    slow = instance_with(SHORT_TERMINAL_TIMEOUT)
    preserve(TERMINAL_SERVERS_CONFIG, on=slow)
    person = create_user(slow)
    connection = terminals.connection(config={"access_grants": [read_grant(person.id)]})
    with admin_of(slow).client() as client:
        configure_terminals(client, connection)
        terminal_id = create_template(client, connection, name=name)["terminal_id"]
        share_template(client, terminal_id, read_grant(person.id))
    return slow, person, terminal_id


@pytest.mark.slow
def test_a_message_sent_while_the_machine_is_still_starting_says_so_and_works_when_sent_again(
    instance_with, preserve, terminals
):
    slow, person, terminal_id = _reader_on_a_short_timeout(
        instance_with, preserve, terminals, f"Workbench {uuid.uuid4().hex[:8]}"
    )

    first = _ask(person, slow.upstream, reply.text("Too early."), terminal_id=terminal_id)
    again = _ask(
        person, slow.upstream, _run("echo started"), reply.text("Ran."), terminal_id=terminal_id
    )

    assert _error_of(first.message).endswith(STILL_STARTING), (
        f"the person was not told the terminal is still starting: {first.message}"
    )
    assert first.requests == [], "the model was asked without the chat's terminal"
    assert _printed(again.results()[0]).strip() == "started", (
        f"the message sent again did not reach the started machine: {again.message}"
    )
    assert [machine["context_id"] for machine in terminals.instances()] == [f"user:{person.id}"], (
        f"sending again started another machine: {terminals.instances()}"
    )


@pytest.mark.slow
def test_the_still_starting_message_names_the_terminal_the_person_picked(
    instance_with, preserve, terminals
):
    name = f"Workbench {uuid.uuid4().hex[:8]}"
    slow, person, terminal_id = _reader_on_a_short_timeout(instance_with, preserve, terminals, name)
    with person.client() as client:
        listed = {entry["id"]: entry["name"] for entry in client.get("/api/v1/terminals/").json()}
    assert listed == {terminal_id: name}, f"the person's terminal menu shows {listed}"

    first = _ask(person, slow.upstream, reply.text("Too early."), terminal_id=terminal_id)

    assert _error_of(first.message) == f"Terminal '{name}' {STILL_STARTING}", (
        f"the message names something other than the terminal the person picked ({name}): "
        f"{_error_of(first.message)}"
    )


def test_a_message_naming_another_terminal_than_the_chats_own_is_refused(
    admin, make_user, connect_terminals, terminals, upstream
):
    person = make_user()
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        first, other = (create_template(client, connection)["terminal_id"] for _ in range(2))
        for terminal_id in (first, other):
            share_template(client, terminal_id, read_grant(person.id))
    tag = uuid.uuid4().hex[:8]

    started = _ask(
        person, upstream, _run(f"echo first > ~/{tag}.txt"), reply.text("Ran."), terminal_id=first
    )
    chat = _after(person, started.turn.chat_id)
    switched = _ask(
        person,
        upstream,
        _run(f"echo other > ~/{tag}.txt"),
        reply.text("Ran."),
        terminal_id=other,
        **chat,
    )

    assert "error" not in started.results()[0], f"the chat's own terminal failed: {started.message}"
    assert _error_of(switched.message) == "Chat terminal selection changed; refresh the chat", (
        f"a message naming another terminal was not refused with the reason: {switched.message}"
    )
    assert switched.requests == [], "the model was asked on a terminal the chat does not have"
    assert terminals.holding(f"{tag}.txt") == {f"user:{person.id}": "first\n"}, (
        f"a command ran in the other terminal's machine: {terminals.holding(f'{tag}.txt')}"
    )
    assert len(terminals.instances()) == 1, (
        f"the refused message started the other terminal's machine: {terminals.instances()}"
    )


def test_a_file_uploaded_for_a_chat_is_where_the_chats_model_works(
    make_user, shared_terminal, terminals, upstream
):
    person = make_user()
    terminal_id = shared_terminal(person, scope="chat")
    tag = uuid.uuid4().hex[:8]
    name, content = f"manifest-{tag}.txt", f"twelve crates {tag}\n"

    started = _ask(person, upstream, reply.text("Hello."), terminal_id=terminal_id)
    chat_id = started.turn.chat_id
    base, session = f"/api/v1/terminals/{terminal_id}", {"X-Session-Id": chat_id}
    with person.client() as client:
        cwd = client.get(f"{base}/files/cwd", headers=session)
        assert cwd.status_code == 200, f"the chat's working directory could not be read: {cwd.text}"
        uploaded = client.post(
            f"{base}/files/upload",
            params={"directory": cwd.json()["cwd"]},
            headers=session,
            files={"file": (name, content.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, f"the upload failed: {uploaded.text}"
    read = _ask(
        person,
        upstream,
        _run(f"cat {uploaded.json()['path']}"),
        reply.text("Read."),
        terminal_id=terminal_id,
        **_after(person, chat_id),
    )

    assert _printed(read.results()[0]) == content.replace("\n", "\r\n"), (
        f"the model did not find the chat's upload: {read.results()[0]}"
    )
    assert terminals.holding(name) == {f"chat:{chat_id}": content}, (
        f"the upload is not in the chat's own machine alone: {terminals.holding(name)}"
    )
