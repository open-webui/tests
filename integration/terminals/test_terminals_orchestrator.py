"""Journey: one connection to a Terminals orchestrator gives each person a machine of their own.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). An admin connects a Terminals
orchestrator once under Settings > Integrations. On that connection a terminal is a template the
orchestrator keeps: an admin, or a person who may write to the connection, creates one from the
terminal menu's New terminal, and it is private until shared. A terminal scoped to the user gives
every person it is shared with a machine of their own, started by their first use; a file one of
them writes is in no other's machine. Deleting the terminal removes its machines and their files.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Each
mutation below, made in a backend copy, turns its test red. The connection check no longer
asking the orchestrator for its templates fails the connecting test (the check answers 400); the
terminal list no longer narrowed to the reader's own grants fails the own-terminal test (the
other person sees it); deleting a terminal without telling the orchestrator fails the deletion
test (the container and its files stay). The template header left out of what the orchestrator
is sent fails the command test (400, Missing X-Terminal-Template-Id), and one fixed user named
to the orchestrator fails the isolation test (the reader's machine lists and prints the writer's
file); these two also fail the other tests that run a command.
"""

from __future__ import annotations

import os
import subprocess
import uuid

import pytest

from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    write_grant,
)
from harness.terminals import (
    CPU_LIMIT,
    OPEN_TERMINAL_IMAGE_ENV,
    create_template,
    execute,
    instance_of,
    output_of,
    share_template,
    start_instance,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

GIBIBYTE = 2**30


def _shared_terminal(admin, connect_terminals, *people) -> str:
    """A terminal the admin made, one machine per user, shared with `people`; returns its id."""
    grants = [read_grant(person.id) for person in people]
    connection = connect_terminals(*grants)
    with admin.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
        share_template(client, terminal_id, *grants)
    return terminal_id


def _listed_terminals(actor) -> dict[str, dict]:
    """The terminals the account's terminal menu offers, by id."""
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry for entry in listed.json()}


def _container_limits(container: str) -> tuple[int, int]:
    """The memory in bytes and the CPUs in billionths Docker holds the container to."""
    limits = subprocess.run(
        ["docker", "inspect", "--format", "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}"]
        + [container],
        capture_output=True,
        text=True,
        check=True,
    )
    memory, cpus = limits.stdout.split()
    return int(memory), int(cpus)


def test_an_admin_connects_the_orchestrator_with_its_key(admin, preserve, terminals):
    connection = terminals.connection()
    with admin.client() as client:
        verify = "/api/v1/configs/terminal_servers/verify"
        verified = client.post(verify, json=connection)
        wrong_key = client.post(verify, json={**connection, "key": "sk-not-the-key"})
        preserve(TERMINAL_SERVERS_CONFIG)
        configure_terminals(client, connection)
        offered = client.get("/api/v1/terminals/templates/connections")
        status = client.get(f"/api/v1/terminals/templates/connections/{connection['id']}/status")

    assert verified.status_code == 200, f"the connection check failed: {verified.text}"
    assert verified.json() == {"status": True, "type": "orchestrator"}
    assert wrong_key.status_code == 400, (
        f"a wrong key passed the connection check: {wrong_key.status_code} {wrong_key.text}"
    )
    assert offered.status_code == 200, offered.text
    assert {"id": connection["id"], "name": connection["name"]} in offered.json(), (
        f"New terminal does not offer the connection: {offered.json()}"
    )
    assert status.status_code == 200, f"the orchestrator's status failed: {status.text}"
    assert (status.json()["backend"], status.json()["image"]) == (
        "docker",
        os.environ[OPEN_TERMINAL_IMAGE_ENV],
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
    assert (listed[terminal_id]["name"], listed[terminal_id]["scope"]) == (name, "user")
    assert _listed_terminals(other) == {}, "a person's own terminal shows for someone else"
    assert before["status"] == "not_created", f"the machine existed before its first use: {before}"
    assert (machine["status"], machine["context_id"]) == ("running", f"user:{owner.id}")
    container = f"terminal-{machine['id']}"
    assert terminals.containers(running_only=True) == [container]
    assert _container_limits(container) == (GIBIBYTE // 2, int(CPU_LIMIT) * 10**9), (
        "the container does not run under the terminal's CPU and memory limits"
    )


def test_a_command_runs_in_the_persons_machine_and_returns_its_output(
    admin, make_user, connect_terminals
):
    person = make_user()
    terminal_id = _shared_terminal(admin, connect_terminals, person)
    tag = uuid.uuid4().hex[:8]

    with person.client() as client:
        ran = execute(client, terminal_id, f"echo ran-{tag} && hostname")
        machine = instance_of(client, terminal_id)

    printed = output_of(ran).split()
    assert ran.json()["exit_code"] == 0, f"the command failed: {ran.json()}"
    assert printed[0] == f"ran-{tag}", f"the command printed {printed}"
    assert (machine["status"], machine["context_id"]) == ("running", f"user:{person.id}")
    container_id = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.Hostname}}", f"terminal-{machine['id']}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert printed[1] == container_id, "the command ran somewhere other than the person's machine"


def test_a_file_one_person_writes_is_not_in_another_persons_machine(
    admin, make_user, connect_terminals, terminals
):
    writer, reader = make_user(), make_user()
    terminal_id = _shared_terminal(admin, connect_terminals, writer, reader)
    name, line = "notes.txt", f"harbour log {uuid.uuid4().hex[:8]}"

    with writer.client() as client:
        written = output_of(execute(client, terminal_id, f"echo {line} > ~/{name} && cat ~/{name}"))
    with reader.client() as client:
        seen = output_of(execute(client, terminal_id, f"ls -a ~; cat ~/{name}"))

    assert line in written, f"the writer's own machine does not hold the file: {written}"
    assert ".bashrc" in seen, f"the reader's home was not listed: {seen}"
    assert name not in seen.split() and line not in seen, (
        f"the reader's machine shows the writer's file: {seen}"
    )
    homes = {
        machine["context_id"]: terminals.home(machine["id"]) for machine in terminals.instances()
    }
    assert set(homes) == {f"user:{writer.id}", f"user:{reader.id}"}, (
        f"the two people did not get a machine each: {sorted(homes)}"
    )
    assert (homes[f"user:{writer.id}"] / name).read_text().strip() == line
    assert not (homes[f"user:{reader.id}"] / name).exists()


def test_deleting_a_terminal_removes_its_machine_and_files(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    terminal_id = _shared_terminal(admin, connect_terminals, person)
    with person.client() as client:
        output_of(execute(client, terminal_id, "echo kept > ~/kept.txt"))
        machine = instance_of(client, terminal_id)
    home = terminals.home(machine["id"])
    assert terminals.containers() == [f"terminal-{machine['id']}"]
    assert (home / "kept.txt").exists(), "the machine's home is not where the harness looks"

    with admin.client() as client:
        deleted = client.delete(f"/api/v1/terminals/templates/{terminal_id}")
    with person.client() as client:
        after = execute(client, terminal_id, "echo still here")

    assert deleted.status_code == 200, f"deleting the terminal failed: {deleted.text}"
    assert terminals.containers() == [], "the machine's container outlived its terminal"
    assert not home.exists(), "the machine's files outlived its terminal"
    assert terminals.templates() == []
    assert terminal_id not in _listed_terminals(person)
    assert after.status_code == 404, (
        f"a deleted terminal still ran a command: {after.status_code} {after.text}"
    )
