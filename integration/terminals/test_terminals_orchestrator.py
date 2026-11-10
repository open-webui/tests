"""Journey: one connection to a Terminals orchestrator gives each person a machine of their own.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). An admin connects a Terminals
orchestrator once under Settings > Integrations. On that connection a terminal is a template the
orchestrator keeps: an admin, or a person who may write to the connection, creates one from the
terminal menu's New terminal, and it stays hidden from the people who may only use the connection
until it is shared with them. A terminal scoped to the user gives every person it is shared with a
machine of their own, started by their first use and held to the terminal's CPU and memory; a file
one of them writes is in no other's machine. Deleting the terminal removes its machines and their
files.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Each
mutation below, made in a backend copy, turns its test red. The connection check no longer
asking the orchestrator for its templates fails the connecting test (the check answers 400), as
does the orchestrator's CPU and memory caps blanked on their way to New terminal; the terminal
list no longer narrowed to the reader's own grants fails the own-terminal test (the other person
sees it); the CPU and memory limits dropped from a new terminal on its way to the orchestrator
fails the limits test (the terminal's settings show none); deleting a terminal without telling
the orchestrator fails the deletion test (the container and its files stay), as does the
terminal list leaving out what was shared with a reader (the terminal is not in their menu).
The template header left out of what the orchestrator is sent fails the command test (400,
Missing X-Terminal-Template-Id), and one fixed user named to the orchestrator fails the isolation
test (the reader's machine lists and prints the writer's file); these two also fail the other
tests that run a command.
"""

from __future__ import annotations

import uuid

import pytest

from harness.terminal_server import read_grant, write_grant
from harness.terminals import (
    MAX_CPU,
    MAX_MEMORY,
    create_template,
    execute,
    instance_of,
    output_of,
    start_instance,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

MEBIBYTE = 2**20


def _listed_terminals(actor) -> dict[str, dict]:
    """The terminals the account's terminal menu offers, by id."""
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry for entry in listed.json()}


def test_an_admin_connects_the_orchestrator_with_its_key(admin, connect_terminals):
    connection = connect_terminals()
    with admin.client() as client:
        verify = "/api/v1/configs/terminal_servers/verify"
        verified = client.post(verify, json=connection)
        wrong_key = client.post(verify, json={**connection, "key": "sk-not-the-key"})
        offered = client.get("/api/v1/terminals/templates/connections")
        status = client.get(f"/api/v1/terminals/templates/connections/{connection['id']}/status")

    assert verified.status_code == 200, f"the connection check failed: {verified.text}"
    assert verified.json() == {"status": True, "type": "orchestrator"}, (
        f"the connection check took the orchestrator for something else: {verified.json()}"
    )
    assert wrong_key.status_code == 400, (
        f"a wrong key passed the connection check: {wrong_key.status_code} {wrong_key.text}"
    )
    assert offered.status_code == 200, offered.text
    assert {"id": connection["id"], "name": connection["name"]} in offered.json(), (
        f"New terminal does not offer the connection: {offered.json()}"
    )
    assert status.status_code == 200, f"the orchestrator's limits could not be read: {status.text}"
    limits = status.json()
    assert (limits["max_cpu"], limits["max_memory"]) == (MAX_CPU, MAX_MEMORY), (
        f"New terminal would show other limits than the orchestrator's: {limits}"
    )


def test_a_person_creates_a_terminal_of_their_own_and_gets_its_machine(
    make_user, connect_terminals, terminals
):
    owner, other = make_user(), make_user()
    connection = connect_terminals(write_grant(owner.id), read_grant(other.id))
    name = f"Workbench {uuid.uuid4().hex[:8]}"

    with owner.client() as client:
        terminal_id = create_template(client, connection, name=name)["terminal_id"]
        before = instance_of(client, terminal_id)
        machine = start_instance(client, terminal_id)

    listed = _listed_terminals(owner)
    assert list(listed) == [terminal_id], f"the owner's terminal menu shows {list(listed)}"
    assert (listed[terminal_id]["name"], listed[terminal_id]["scope"]) == (name, "user"), (
        f"the owner's terminal is listed as {listed[terminal_id]}"
    )
    assert _listed_terminals(other) == {}, "a person's own terminal shows for someone else"
    assert before["status"] == "not_created", f"the machine existed before its first use: {before}"
    assert (machine["status"], machine["context_id"]) == ("running", f"user:{owner.id}"), (
        f"starting the terminal did not give its owner a running machine: {machine}"
    )
    assert terminals.containers(running_only=True) == [f"terminal-{machine['id']}"], (
        f"the owner's machine is not the one container running: {terminals.containers()}"
    )


def test_a_terminals_machine_runs_under_its_cpu_and_memory_limits(
    admin, connect_terminals, terminals
):
    connection = connect_terminals()

    with admin.client() as client:
        created = create_template(client, connection, cpu_limit="0.5", memory_limit="256Mi")
        machine = start_instance(client, created["terminal_id"])
        shown = client.get(f"/api/v1/terminals/templates/{created['terminal_id']}")

    assert shown.status_code == 200, f"the terminal's settings could not be read: {shown.text}"
    settings = shown.json()["data"]
    assert (settings.get("cpu_limit"), settings.get("memory_limit")) == ("0.5", "256Mi"), (
        f"the terminal's settings show other limits than the ones it was created with: {settings}"
    )
    memory, cpus = terminals.limits(machine["id"])
    assert (memory, cpus) == (256 * MEBIBYTE, 500_000_000), (
        f"the machine runs with {memory} bytes of memory and {cpus} billionths of a CPU, "
        "not the terminal's 256Mi and half a CPU"
    )


def test_a_command_runs_in_the_persons_machine_and_returns_its_output(
    make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]

    with person.client() as client:
        ran = execute(client, terminal_id, f"echo ran-{tag} && hostname")
        machine = instance_of(client, terminal_id)

    printed = output_of(ran).split()
    assert ran.json()["exit_code"] == 0, f"the command failed: {ran.json()}"
    assert printed[0] == f"ran-{tag}", f"the command printed {printed}"
    assert (machine["status"], machine["context_id"]) == ("running", f"user:{person.id}"), (
        f"the command did not start the person's own machine: {machine}"
    )
    assert printed[1] == terminals.hostname(machine["id"]), (
        f"the command ran on {printed[1]}, somewhere other than the person's machine"
    )


def test_a_file_one_person_writes_is_not_in_another_persons_machine(
    make_user, shared_terminal, terminals
):
    writer, reader = make_user(), make_user()
    terminal_id = shared_terminal(writer, reader)
    tag = uuid.uuid4().hex[:8]
    name, line, own = "notes.txt", f"harbour log {tag}", f"own-{tag}"

    with writer.client() as client:
        written = output_of(execute(client, terminal_id, f"echo {line} > ~/{name} && cat ~/{name}"))
    with reader.client() as client:
        seen = output_of(execute(client, terminal_id, f"touch ~/{own}; ls -a ~; cat ~/{name}"))

    assert line in written, f"the writer's own machine does not hold the file: {written}"
    assert own in seen.split(), f"the reader's home was not listed: {seen}"
    assert name not in seen.split() and line not in seen, (
        f"the reader's machine shows the writer's file: {seen}"
    )
    held = terminals.holding(name)
    assert held == {f"user:{writer.id}": f"{line}\n"}, (
        f"on disk the writer's file is in these machines' homes: {held}"
    )
    assert set(terminals.holding(own)) == {f"user:{reader.id}"}, (
        f"on disk the reader's own file is in these machines' homes: {terminals.holding(own)}"
    )


def test_deleting_a_terminal_removes_its_machine_and_files(
    admin, make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person)
    with person.client() as client:
        output_of(execute(client, terminal_id, "echo kept > ~/kept.txt"))
        machine = instance_of(client, terminal_id)
    home = terminals.home(machine["id"])
    assert terminal_id in _listed_terminals(person), (
        "the terminal shared with the person is not in their terminal menu"
    )
    assert terminals.containers() == [f"terminal-{machine['id']}"], (
        f"the person's machine is not the one container: {terminals.containers()}"
    )
    assert (home / "kept.txt").exists(), "the machine's home is not where the harness looks"

    with admin.client() as client:
        deleted = client.delete(f"/api/v1/terminals/templates/{terminal_id}")
    with person.client() as client:
        after = execute(client, terminal_id, "echo still here")

    assert deleted.status_code == 200, f"deleting the terminal failed: {deleted.text}"
    assert terminals.containers() == [], "the machine's container outlived its terminal"
    assert not home.exists(), "the machine's files outlived its terminal"
    assert terminals.templates() == [], (
        f"the orchestrator still holds the terminal: {terminals.templates()}"
    )
    assert terminal_id not in _listed_terminals(person), (
        "the deleted terminal is still in the person's terminal menu"
    )
    assert after.status_code == 404, (
        f"a deleted terminal still ran a command: {after.status_code} {after.text}"
    )
