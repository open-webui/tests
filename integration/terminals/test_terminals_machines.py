"""Journey: the machines of a terminal on a Terminals orchestrator, and who may manage them.

New on dev after 0.13.3 (titled `refac`, 6f2fc7bc7 to c15f28649). A terminal on an orchestrator
gives each person, chat or model it is scoped to a machine of its own. The terminal's settings
list its machines with whose they are, and let a person who may write to the connection start,
stop, restart, reset and delete each of them, their own or someone else's. A person the terminal
is only shared with uses their machine and manages none. A stopped machine keeps its files and
starts again with the next command; a restart keeps the files and nothing else; a reset clears
the files; deleting the machine removes it while the terminal stays. The terminal's start
folder, setup script and image reach the machine, and what the orchestrator refuses (an image it
does not have or does not allow, a scope change while machines exist) reaches the person with
the orchestrator's reason.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Each
mutation below, made in a backend copy, turns its tests red. The machine actions sent to the
orchestrator as other actions (stop as start, start as stop, restart as reset, reset as restart,
delete as stop) fails the stop, restart, reset, delete, never-started, someone-else's-machine,
scope, missing-image, scope-change and running-command tests. The action always naming the caller
as the machine's user fails the someone-else's-machine test (the manager's own machine stops) and
the manager's stop in the may-only-use test. Write access to the connection no longer asked of the
management routes fails the may-only-use test (the reader's Start is answered 200). The names left
out of the machine list fails the list test. The chat and the model left out of the action fails
the scope test (409 for a named chat). `env` dropped from a new terminal fails the start folder
test. The orchestrator's reason replaced by a fixed text fails the never-started, missing-image,
not-allowed-image and scope-change tests. The allowed-image and idle tests need settings of the
service: without `TERMINALS_ALLOWED_IMAGES` the other image is accepted (201), and with
`TERMINALS_IDLE_TIMEOUT_MINUTES` at 0 the machine never stops.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import httpx
import pytest

from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    write_grant,
)
from harness.terminals import (
    OPEN_TERMINAL_IMAGE_ENV,
    act_on_instance,
    create_template,
    execute,
    instance_of,
    list_instances,
    output_of,
    serving_terminals,
    share_template,
    start_instance,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

NEVER_STARTED = "Environment has not been created"
WRITE_ACCESS_REQUIRED = "Terminal connection write access required"
# the idle sweep runs once a minute, after one idle minute
IDLE_STOP_SECONDS = 240


def _managed_terminal(connect_terminals, manager, *people, **data) -> str:
    """A terminal `manager` created on a connection they may write to, shared with `people`."""
    grants = [read_grant(person.id) for person in people]
    connection = connect_terminals(write_grant(manager.id), *grants)
    with manager.client() as client:
        terminal_id = create_template(client, connection, **data)["terminal_id"]
        share_template(client, terminal_id, *grants)
    return terminal_id


def _new_chat(client: httpx.Client, title: str) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": {"title": title}})
    assert created.status_code == 200, f"creating the chat failed: {created.text}"
    return created.json()["id"]


def _reason(answer: httpx.Response) -> tuple[int, object]:
    return answer.status_code, answer.json().get("detail")


def _menu(client: httpx.Client) -> dict[str, dict]:
    listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry for entry in listed.json()}


def test_a_stopped_machine_keeps_its_files_and_starts_again_with_the_next_command(
    make_user, connect_terminals, terminals
):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager)
    line = f"kept {uuid.uuid4().hex[:8]}"

    with manager.client() as client:
        output_of(execute(client, terminal_id, f"echo {line} > ~/kept.txt"))
        before = instance_of(client, terminal_id)
        stopped = act_on_instance(client, terminal_id, "stop")
        after = instance_of(client, terminal_id)
        running_containers = terminals.containers(running_only=True)
        system_info = client.get(f"/api/v1/terminals/{terminal_id}/system/info")
        still_stopped = instance_of(client, terminal_id)
        printed = output_of(execute(client, terminal_id, "cat ~/kept.txt"))
        again = instance_of(client, terminal_id)

    assert before["status"] == "running", f"the command did not start the machine: {before}"
    assert stopped.status_code == 200, f"stopping the machine failed: {stopped.text}"
    assert (after["status"], after["id"]) == ("stopped", before["id"]), (
        f"the stopped machine is shown as {after}"
    )
    assert running_containers == [], f"a stopped machine still runs: {running_containers}"
    assert _reason(system_info) == (409, "Terminal instance is not running"), (
        f"System Info of a stopped machine answered {system_info.status_code} {system_info.text}"
    )
    assert still_stopped["status"] == "stopped", (
        f"reading System Info started the stopped machine: {still_stopped}"
    )
    assert line in printed, f"the machine lost its file while stopped: {printed!r}"
    assert (again["status"], again["id"]) == ("running", before["id"]), (
        f"the next command did not start the stopped machine again: {again}"
    )


def test_restarting_a_machine_keeps_its_files_and_nothing_else(
    make_user, connect_terminals, terminals
):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager)
    tag = uuid.uuid4().hex[:8]
    kept, scratch = f"kept-{tag}", f"/tmp/scratch-{tag}"
    look = f"ls ~/{kept} {scratch}"

    with manager.client() as client:
        before = output_of(execute(client, terminal_id, f"touch ~/{kept} {scratch}; {look}"))
        restarted = act_on_instance(client, terminal_id, "restart")
        machine = instance_of(client, terminal_id)
        after = output_of(execute(client, terminal_id, look))

    assert kept in before and scratch in before, f"the machine holds neither file: {before!r}"
    assert restarted.status_code == 200, f"restarting the machine failed: {restarted.text}"
    assert machine["status"] == "running", f"the restarted machine is not running: {machine}"
    assert f"/home/user/{kept}" in after.split(), f"the restart lost the home's file: {after!r}"
    assert scratch not in after.split(), (
        f"a file outside the home outlived the restart, so nothing restarted: {after!r}"
    )


def test_resetting_a_machine_clears_its_files(make_user, connect_terminals, terminals):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager)
    tag = uuid.uuid4().hex[:8]
    old, new = f"old-{tag}.txt", f"new-{tag}.txt"

    with manager.client() as client:
        output_of(execute(client, terminal_id, f"echo before the reset > ~/{old}"))
        held_before = terminals.holding(old)
        reset = act_on_instance(client, terminal_id, "reset")
        machine = instance_of(client, terminal_id)
        listed = output_of(execute(client, terminal_id, f"touch ~/{new}; ls -a ~")).split()

    assert held_before == {f"user:{manager.id}": "before the reset\n"}, (
        f"the machine did not hold the file before the reset: {held_before}"
    )
    assert reset.status_code == 200, f"resetting the machine failed: {reset.text}"
    assert machine["status"] == "stopped", f"a reset machine is shown as {machine}"
    assert new in listed, f"the home was not listed after the reset: {listed}"
    assert old not in listed, f"the file is still there after the reset: {listed}"
    assert terminals.holding(old) == {}, "on disk the file outlived the reset"


def test_deleting_a_machine_removes_it_and_its_files_and_keeps_the_terminal(
    make_user, connect_terminals, terminals
):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager)
    name = f"gone-{uuid.uuid4().hex[:8]}.txt"

    with manager.client() as client:
        output_of(execute(client, terminal_id, f"echo here > ~/{name}"))
        machine = instance_of(client, terminal_id)
        home = terminals.home(machine["id"])
        home_before = home.exists()
        deleted = act_on_instance(client, terminal_id, "delete")
        after = instance_of(client, terminal_id)
        containers = terminals.containers()
        home_after = home.exists()
        listed_machines = list_instances(client, terminal_id)
        menu = _menu(client)
        fresh = output_of(execute(client, terminal_id, f"touch ~/fresh; ls -a ~; cat ~/{name}"))

    assert machine["status"] == "running" and home_before, (
        f"the machine or its home was not there before the delete: {machine}"
    )
    assert deleted.status_code == 200, f"deleting the machine failed: {deleted.text}"
    assert after["status"] == "not_created", f"the deleted machine is shown as {after}"
    assert containers == [], f"the machine's container outlived it: {containers}"
    assert not home_after, "the machine's files outlived it"
    assert listed_machines == {}, f"the deleted machine is still listed: {listed_machines}"
    assert terminal_id in menu, f"deleting the machine took the terminal away: {list(menu)}"
    assert "fresh" in fresh.split() and name not in fresh.split(), (
        f"the next command did not get a new, empty machine: {fresh!r}"
    )


def test_a_machine_that_was_never_started_can_only_be_started(
    make_user, connect_terminals, terminals
):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager)

    with manager.client() as client:
        refused = {
            action: act_on_instance(client, terminal_id, action)
            for action in ("stop", "restart", "reset")
        }
        untouched = instance_of(client, terminal_id)
        started = start_instance(client, terminal_id)
        stopped = act_on_instance(client, terminal_id, "stop")

    for action, answer in refused.items():
        assert _reason(answer) == (404, NEVER_STARTED), (
            f"{action} of a machine that was never started answered "
            f"{answer.status_code} {answer.text}"
        )
    assert untouched["status"] == "not_created", (
        f"a refused action left a machine behind: {untouched}"
    )
    assert started["status"] == "running", f"Start did not start the machine: {started}"
    assert stopped.status_code == 200, (
        f"the started machine could not be stopped: {stopped.status_code} {stopped.text}"
    )


def test_a_person_who_may_only_use_a_terminal_manages_no_machine(
    make_user, connect_terminals, terminals
):
    manager, reader = make_user(), make_user()
    terminal_id = _managed_terminal(connect_terminals, manager, reader)
    name = f"mine-{uuid.uuid4().hex[:8]}.txt"

    with reader.client() as client:
        output_of(execute(client, terminal_id, f"echo mine > ~/{name}"))
        machine = instance_of(client, terminal_id)
        refused = {
            action: act_on_instance(client, terminal_id, action)
            for action in ("start", "stop", "restart", "reset", "delete")
        }
        machines = client.get(f"/api/v1/terminals/templates/{terminal_id}/instances")
        after = instance_of(client, terminal_id)
    with manager.client() as client:
        managers_view = list_instances(client, terminal_id)
        stopped = act_on_instance(client, terminal_id, "stop", user_id=reader.id)

    assert machine["status"] == "running", f"the reader's command started no machine: {machine}"
    for action, answer in refused.items():
        assert _reason(answer) == (403, WRITE_ACCESS_REQUIRED), (
            f"a person the terminal is only shared with could {action} a machine: "
            f"{answer.status_code} {answer.text}"
        )
    assert _reason(machines) == (403, WRITE_ACCESS_REQUIRED), (
        f"a person the terminal is only shared with read its machine list: {machines.text}"
    )
    assert after["status"] == "running", f"a refused action changed the machine: {after}"
    assert terminals.holding(name) == {f"user:{reader.id}": "mine\n"}, (
        "a refused action touched the reader's files"
    )
    assert set(managers_view) == {f"user:{reader.id}"}, (
        f"the manager's machine list shows {set(managers_view)}"
    )
    assert stopped.status_code == 200, (
        f"the manager could not stop the same machine: {stopped.status_code} {stopped.text}"
    )


def test_a_manager_stops_and_deletes_someone_elses_machine_and_not_their_own(
    make_user, connect_terminals, terminals
):
    manager, person = make_user(), make_user()
    terminal_id = _managed_terminal(connect_terminals, manager, person)
    name = f"notes-{uuid.uuid4().hex[:8]}.txt"

    with person.client() as client:
        output_of(execute(client, terminal_id, f"echo person > ~/{name}"))
    with manager.client() as client:
        output_of(execute(client, terminal_id, f"echo manager > ~/{name}"))
        stopped = act_on_instance(client, terminal_id, "stop", user_id=person.id)
        after_stop = {
            context: machine["status"]
            for context, machine in list_instances(client, terminal_id).items()
        }
        deleted = act_on_instance(client, terminal_id, "delete", user_id=person.id)
        after_delete = list_instances(client, terminal_id)
    with person.client() as client:
        persons_machine = instance_of(client, terminal_id)

    assert stopped.status_code == 200, f"stopping the person's machine failed: {stopped.text}"
    assert after_stop == {f"user:{person.id}": "stopped", f"user:{manager.id}": "running"}, (
        f"after stopping the person's machine the list shows {after_stop}"
    )
    assert deleted.status_code == 200, f"deleting the person's machine failed: {deleted.text}"
    assert set(after_delete) == {f"user:{manager.id}"}, (
        f"after deleting the person's machine the list shows {set(after_delete)}"
    )
    assert persons_machine["status"] == "not_created", (
        f"the person still has the deleted machine: {persons_machine}"
    )
    assert terminals.holding(name) == {f"user:{manager.id}": "manager\n"}, (
        f"on disk the file is now in these machines: {terminals.holding(name)}"
    )


def test_the_machine_list_names_whose_machine_each_is(
    admin, make_user, connect_terminals, terminals
):
    manager, person = make_user(), make_user()
    grants = [read_grant(person.id)]
    connection = connect_terminals(write_grant(manager.id), *grants)
    title = f"Harbour plans {uuid.uuid4().hex[:8]}"
    with manager.client() as client:
        per_user = create_template(client, connection, scope="user")["terminal_id"]
        per_chat = create_template(client, connection, scope="chat")["terminal_id"]
        share_template(client, per_user, *grants)
        share_template(client, per_chat, *grants)

    with person.client() as client:
        chat_id = _new_chat(client, title)
        output_of(execute(client, per_user, "true"))
        output_of(execute(client, per_chat, "true", chat_id))
    with manager.client() as client:
        output_of(execute(client, per_user, "true"))
        users = list_instances(client, per_user)
        chats = list_instances(client, per_chat)
    with person.client() as client:
        assert client.delete(f"/api/v1/chats/{chat_id}").status_code == 200
    with admin.client() as client:
        assert client.delete(f"/api/v1/users/{person.id}").status_code == 200
    with manager.client() as client:
        users_after = list_instances(client, per_user)
        chats_after = list_instances(client, per_chat)

    named = {context: machine.get("label") for context, machine in users.items()}
    assert named == {f"user:{person.id}": person.name, f"user:{manager.id}": manager.name}, (
        f"the per-user terminal's machines are named {named}"
    )
    assert users[f"user:{person.id}"]["context"] == {"user_id": person.id}, (
        f"the person's machine does not say whose it is: {users[f'user:{person.id}']}"
    )
    assert set(chats) == {f"chat:{chat_id}"}, f"the per-chat terminal lists {set(chats)}"
    chat_machine = chats[f"chat:{chat_id}"]
    assert (chat_machine.get("label"), chat_machine.get("user_name")) == (title, person.name), (
        f"the chat's machine is not named by its chat and the chat's owner: {chat_machine}"
    )
    assert chat_machine["context"] == {"chat_id": chat_id}, chat_machine
    gone_chat = chats_after.get(f"chat:{chat_id}")
    assert gone_chat is not None, f"the machine of a deleted chat is not listed: {chats_after}"
    assert (gone_chat.get("label"), gone_chat.get("user_name")) == (None, None), (
        f"the machine of a deleted chat still carries a name: {gone_chat}"
    )
    gone_user = users_after.get(f"user:{person.id}")
    assert gone_user is not None, f"the machine of a deleted account is not listed: {users_after}"
    assert gone_user.get("label") is None, (
        f"the machine of a deleted account still carries a name: {gone_user}"
    )
    assert users_after[f"user:{manager.id}"].get("label") == manager.name, (
        f"the manager's machine lost its name: {users_after}"
    )


def test_a_manager_starts_a_per_chat_or_per_model_machine_only_by_naming_the_chat_or_model(
    make_user, connect_terminals, terminals
):
    manager = make_user()
    connection = connect_terminals(write_grant(manager.id))

    with manager.client() as client:
        per_chat = create_template(client, connection, scope="chat")["terminal_id"]
        per_model = create_template(client, connection, scope="model")["terminal_id"]
        chat_id = _new_chat(client, "Machines")
        no_chat = act_on_instance(client, per_chat, "start")
        unsaved_chat = act_on_instance(client, per_chat, "start", chat_id="temporary:unsaved")
        no_model = act_on_instance(client, per_model, "start")
        started_by_refused = terminals.instances()
        chat_started = act_on_instance(client, per_chat, "start", chat_id=chat_id)
        model_started = act_on_instance(client, per_model, "start", model_id=MOCK_MODEL_ID)
        chat_machine = instance_of(client, per_chat, chat_id=chat_id)
        model_machine = instance_of(client, per_model, model_id=MOCK_MODEL_ID)

    for label, answer in (("no chat", no_chat), ("an unsaved chat", unsaved_chat)):
        assert _reason(answer) == (409, "This terminal requires a saved chat"), (
            f"starting a per-chat machine for {label} answered {answer.status_code} {answer.text}"
        )
    assert _reason(no_model) == (409, "This terminal requires a model"), (
        f"starting a per-model machine for no model answered {no_model.status_code} {no_model.text}"
    )
    assert started_by_refused == [], f"a refused start left a machine: {started_by_refused}"
    assert chat_started.status_code == 200, (
        f"starting the chat's machine failed: {chat_started.text}"
    )
    assert model_started.status_code == 200, (
        f"starting the model's machine failed: {model_started.text}"
    )
    assert (chat_machine["status"], chat_machine["context_id"]) == ("running", f"chat:{chat_id}"), (
        f"the named chat's machine is {chat_machine}"
    )
    assert (model_machine["status"], model_machine["context_id"]) == (
        "running",
        f"model:{MOCK_MODEL_ID}",
    ), f"the named model's machine is {model_machine}"


def test_a_terminals_start_folder_and_setup_script_reach_its_machine(
    admin, connect_terminals, terminals
):
    connection = connect_terminals()
    tag = uuid.uuid4().hex[:8]
    folder, line = f"/home/user/work-{tag}", f"prepared {tag}"
    env = {
        "OPEN_TERMINAL_START_DIR": folder,
        "OPEN_TERMINAL_SETUP_SCRIPT": f"echo {line} > /home/user/setup-{tag}.txt",
    }

    with admin.client() as client:
        terminal_id = create_template(client, connection, env=env)["terminal_id"]
        printed = output_of(execute(client, terminal_id, f"pwd; cat ~/setup-{tag}.txt"))
        cwd = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")
        shown = client.get(f"/api/v1/terminals/templates/{terminal_id}")

    assert shown.status_code == 200, f"the terminal's settings could not be read: {shown.text}"
    assert shown.json()["data"].get("env") == env, (
        f"the terminal's settings show another start folder or setup script: {shown.json()}"
    )
    assert printed.split("\r\n")[:2] == [folder, line], (
        f"the command did not start in the start folder after the setup script ran: {printed!r}"
    )
    assert cwd.status_code == 200 and cwd.json()["cwd"] == folder, (
        f"the file browser would open in {cwd.text}, not in the start folder"
    )


def test_a_terminal_whose_image_the_orchestrator_does_not_have_says_so_on_first_use(
    admin, connect_terminals, terminals
):
    connection = connect_terminals()
    absent = f"example.com/absent-{uuid.uuid4().hex[:8]}:1"
    reason = f"Image '{absent}' is not present and the pull policy is Never"

    with admin.client() as client:
        terminal_id = create_template(client, connection, image=absent)["terminal_id"]
        command = execute(client, terminal_id, "echo ran")
        started = act_on_instance(client, terminal_id, "start")
        machine = instance_of(client, terminal_id)
        containers = terminals.containers()
        changed = client.patch(
            f"/api/v1/terminals/templates/{terminal_id}",
            json={"image": os.environ[OPEN_TERMINAL_IMAGE_ENV]},
        )
        printed = output_of(execute(client, terminal_id, "echo ran"))

    assert _reason(command) == (503, reason), (
        f"a command on a terminal without its image answered {command.status_code} {command.text}"
    )
    assert _reason(started) == (503, reason), (
        f"starting a terminal without its image answered {started.status_code} {started.text}"
    )
    assert machine["status"] != "running" and containers == [], (
        f"a machine runs without its image: {machine} {containers}"
    )
    assert changed.status_code == 200, f"changing the terminal's image failed: {changed.text}"
    assert printed.strip() == "ran", f"with an image that is there the command printed {printed!r}"


def test_a_terminals_scope_changes_only_once_its_machines_are_deleted(
    make_user, connect_terminals, terminals
):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager, scope="user")
    settings = f"/api/v1/terminals/templates/{terminal_id}"

    with manager.client() as client:
        output_of(execute(client, terminal_id, "true"))
        refused = client.patch(settings, json={"scope": "chat"})
        kept = _menu(client)[terminal_id]["scope"]
        deleted = act_on_instance(client, terminal_id, "delete")
        changed = client.patch(settings, json={"scope": "chat"})
        now = _menu(client)[terminal_id]["scope"]
        chat_id = _new_chat(client, "Scope")
        output_of(execute(client, terminal_id, "true", chat_id))
        machines = set(list_instances(client, terminal_id))

    assert _reason(refused) == (
        409,
        "Delete existing environments before changing scope or storage mode",
    ), f"changing the scope beside a machine answered {refused.status_code} {refused.text}"
    assert kept == "user", f"a refused scope change shows in the terminal menu as {kept}"
    assert deleted.status_code == 200, f"deleting the machine failed: {deleted.text}"
    assert changed.status_code == 200, (
        f"changing the scope without machines failed: {changed.status_code} {changed.text}"
    )
    assert now == "chat", f"the terminal menu shows the scope {now} after the change"
    assert machines == {f"chat:{chat_id}"}, (
        f"after the change to per chat the terminal's machines are {machines}"
    )


def test_stopping_a_machine_ends_the_command_running_in_it(make_user, connect_terminals, terminals):
    manager = make_user()
    terminal_id = _managed_terminal(connect_terminals, manager)
    tag = uuid.uuid4().hex[:8]
    begun, finished = f"begun-{tag}", f"finished-{tag}"
    answers = {}

    def run_long_command() -> None:
        with manager.client() as client:
            answers["command"] = execute(
                client, terminal_id, f"touch ~/{begun}; sleep 25; touch ~/{finished}"
            )

    running = threading.Thread(target=run_long_command, daemon=True)
    running.start()
    deadline = time.monotonic() + 30
    while not terminals.holding(begun) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert terminals.holding(begun), "the long command never began"
    with manager.client() as client:
        stopped = act_on_instance(client, terminal_id, "stop")
        running.join(timeout=20)
        machine = instance_of(client, terminal_id)
        listed = output_of(execute(client, terminal_id, "ls ~")).split()

    assert stopped.status_code == 200, f"stopping the machine failed: {stopped.text}"
    assert not running.is_alive(), "the command's request was still open 20 s after the stop"
    command = answers["command"]
    assert (command.status_code, command.json()) == (
        502,
        {"error": "Upstream connection failed"},
    ), f"the command cut off by the stop answered {command.status_code} {command.text}"
    assert machine["status"] == "stopped", f"the stopped machine is shown as {machine}"
    assert begun in listed and finished not in listed, (
        f"the stopped command ran on to its end: {listed}"
    )


@pytest.fixture(scope="module")
def strict_terminals_service():
    """A Terminals service that allows one image only and stops a machine idle for a minute."""
    settings = {
        "TERMINALS_ALLOWED_IMAGES": os.getenv(OPEN_TERMINAL_IMAGE_ENV, ""),
        "TERMINALS_IDLE_TIMEOUT_MINUTES": "1",
    }
    with serving_terminals(settings) as service:
        yield service


@pytest.fixture
def strict_connection(strict_terminals_service, admin, preserve):
    """The admin's connection to the strict service; its terminals go after the test."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = strict_terminals_service.connection()
    with admin.client() as client:
        configure_terminals(client, connection)
    yield connection
    with admin.client() as client:
        for template in strict_terminals_service.templates():
            client.delete(
                f"/api/v1/terminals/templates/template:{connection['id']}:{template['id']}"
            )
    strict_terminals_service.reset()


@pytest.mark.slow
def test_an_image_the_orchestrator_does_not_allow_is_refused_with_its_reason(
    admin, strict_connection, strict_terminals_service
):
    allowed = os.environ[OPEN_TERMINAL_IMAGE_ENV]
    other = f"example.com/other-{uuid.uuid4().hex[:8]}:1"
    reason = f"Image '{other}' not in allowed list"

    with admin.client() as client:
        refused = client.post(
            "/api/v1/terminals/templates",
            json={
                "id": str(uuid.uuid4()),
                "connection_id": strict_connection["id"],
                "data": {"name": "Other image", "scope": "user", "image": other},
            },
        )
        menu_after_refusal = _menu(client)
        terminal_id = create_template(client, strict_connection, image=allowed)["terminal_id"]
        settings = f"/api/v1/terminals/templates/{terminal_id}"
        changed = client.patch(settings, json={"image": other})
        shown = client.get(settings)

    assert _reason(refused) == (400, reason), (
        f"a terminal on an image that is not allowed answered {refused.status_code} {refused.text}"
    )
    assert menu_after_refusal == {}, f"the refused terminal is in the menu: {menu_after_refusal}"
    assert _reason(changed) == (400, reason), (
        f"changing a terminal to an image that is not allowed answered "
        f"{changed.status_code} {changed.text}"
    )
    assert shown.json()["data"].get("image") == allowed, (
        f"the terminal's settings show the refused image: {shown.json()}"
    )


@pytest.mark.slow
def test_a_machine_left_idle_is_stopped_and_starts_again_with_the_next_command(
    admin, strict_connection, strict_terminals_service
):
    line = f"before the idle stop {uuid.uuid4().hex[:8]}"

    with admin.client() as client:
        terminal_id = create_template(client, strict_connection)["terminal_id"]
        output_of(execute(client, terminal_id, f"echo {line} > ~/kept.txt"))
        started = instance_of(client, terminal_id)
        deadline = time.monotonic() + IDLE_STOP_SECONDS
        idle = started
        while idle["status"] == "running" and time.monotonic() < deadline:
            time.sleep(5)
            idle = instance_of(client, terminal_id)
        containers = strict_terminals_service.containers(running_only=True)
        printed = output_of(execute(client, terminal_id, "cat ~/kept.txt"))
        again = instance_of(client, terminal_id)

    assert started["status"] == "running", f"the command did not start the machine: {started}"
    assert idle["status"] == "stopped", (
        f"the machine was still {idle['status']} after {IDLE_STOP_SECONDS} s without use"
    )
    assert containers == [], f"the idle machine's container still runs: {containers}"
    assert line in printed, f"the machine lost its file to the idle stop: {printed!r}"
    assert again["status"] == "running", f"the next command did not start it again: {again}"
