"""Journey: who may create, change, see and delete the terminals of a Terminals connection.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). On a connection to a Terminals
orchestrator a terminal is a template the orchestrator keeps. New terminal offers the connection
to the people who may write to it (an admin, a write grant of their own or of their group) and
shows them the orchestrator's limits. A new terminal needs a UUID, is kept by the orchestrator and
is readable by its creator alone, who keeps it after losing write access. A person who may write
sees, renames, changes and deletes every terminal of the connection; a person who may only read
sees and uses the ones shared with them and is refused every management call. A deleted terminal
loses its shares. The terminal menu lists nothing of a switched-off connection, an id that names
no template is refused before the orchestrator is asked, and an orchestrator that refuses, lost
the terminal, redirects, has no key or is stopped is answered for with its own words or a plain
refusal while the other terminals stay listed. With the admin bypass off an admin needs a grant
like anyone else once the connection has grants, and with tool servers off every template call is
refused.

The real Terminals service (open-webui/terminals) runs on Docker (`harness.terminals`); without
its images or Docker the tests skip. The redirect test scripts a server that answers as an
orchestrator never does.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Each
mutation below, made in a backend copy, turns its test red. New terminal offering every
connection, or the limits read without write access, fails the offer test. The creator's read
grant not stored fails the creator test. Any text taken as a terminal's id, or creation without
the write check, fails the UUID test. A change answered without being sent to the orchestrator
fails the rename test. A share taken as enough to manage fails the refused reader test. The
menu narrowed to their shares for a writer too, or built without asking for access to the
connection, fails the menu test. Use without the share, or without access to the connection,
fails the use test. A deletion that keeps the shares fails the put-back test. A switched-off
connection still resolved fails the switched-off test. An id no longer compared with its
canonical form, or a connection of any type taken for an orchestrator, fails the id test. The
orchestrator's message replaced by a fixed one, or its status by 502, fails the refusal and the
lost terminal tests. Redirects followed, or the key left out, fails the redirect test. The key
check dropped fails the key test (the orchestrator's 401 comes back). An outage answered 502,
or raised out of the terminal list, fails the stopped test. An admin always writing fails the
no-bypass test, the bypass ignored for writing fails the bypass test and the tool server
switch ignored when a connection is resolved fails the tool servers test.
"""

from __future__ import annotations

import uuid
from typing import Callable, Iterator

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor, admin_of, create_user
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
    write_grant,
)
from harness.terminals import (
    MAX_CPU,
    MAX_MEMORY,
    Terminals,
    create_template,
    serving_terminals,
    share_template,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TEMPLATES = "/api/v1/terminals/templates"
WRITE_REQUIRED = (403, "Terminal connection write access required")
NO_CONNECTION_ACCESS = (403, "Terminal connection access denied")
NO_CONNECTION = (404, "Managed terminal connection not found")
# What the terminal's settings dialog and its machine menu call, the deletion last.
MANAGEMENT_CALLS = [
    ("GET", "", None),
    ("PATCH", "", {"name": "Taken over"}),
    ("GET", "/access", None),
    ("POST", "/access", {"access_grants": []}),
    ("GET", "/instances", None),
    ("POST", "/start", {}),
    ("POST", "/stop", {}),
    ("POST", "/restart", {}),
    ("POST", "/reset", {}),
    ("DELETE", "/instance", {}),
    ("DELETE", "", None),
]


def _menu(actor: Actor) -> dict[str, dict]:
    """The terminals the account's terminal menu offers, by id."""
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return {entry["id"]: entry for entry in listed.json()}


def _answer(answer: httpx.Response) -> tuple[int, str]:
    """The status and the message a refused call shows in the dialog."""
    body = answer.json() if answer.content else {}
    return answer.status_code, body.get("detail") if isinstance(body, dict) else body


def _offered(actor: Actor) -> list[dict]:
    """The connections New terminal lets the account pick."""
    with actor.client() as client:
        offered = client.get(f"{TEMPLATES}/connections")
    assert offered.status_code == 200, offered.text
    return offered.json()


def _limits(actor: Actor, connection_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"{TEMPLATES}/connections/{connection_id}/status")


def _settings(actor: Actor, terminal_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"{TEMPLATES}/{terminal_id}")


def _machine(actor: Actor, terminal_id: str) -> httpx.Response:
    """The account's machine of the terminal, as the terminal menu reads it before any use."""
    with actor.client() as client:
        return client.get(f"{TEMPLATES}/{terminal_id}/instance")


def _shares(actor: Actor, terminal_id: str) -> list[tuple[str, str, str]]:
    with actor.client() as client:
        shares = client.get(f"{TEMPLATES}/{terminal_id}/access")
    assert shares.status_code == 200, f"the terminal's shares could not be read: {shares.text}"
    return sorted(
        (share["principal_type"], share["principal_id"], share["permission"])
        for share in shares.json()
    )


def _new_terminal(actor: Actor, connection_id: str, template_id: str, **data) -> httpx.Response:
    with actor.client() as client:
        return client.post(
            TEMPLATES, json={"id": template_id, "connection_id": connection_id, "data": data}
        )


def _management_answers(actor: Actor, terminal_id: str) -> dict[str, tuple[int, str]]:
    with actor.client() as client:
        return {
            f"{method} {suffix or '/'}": _answer(
                client.request(method, f"{TEMPLATES}/{terminal_id}{suffix}", json=body)
            )
            for method, suffix, body in MANAGEMENT_CALLS
        }


def test_new_terminal_offers_the_connection_only_to_those_who_may_write_to_it(
    admin, make_user, connect_terminals
):
    writer, teammate, reader, stranger = make_user(), make_user(), make_user(), make_user()
    team = make_group(admin, [teammate])
    name = f"Workshop {uuid.uuid4().hex[:8]}"
    connection = connect_terminals(
        write_grant(writer.id), grant("group", team, "write"), read_grant(reader.id), name=name
    )
    choice = [{"id": connection["id"], "name": name}]

    offered = {
        role: _offered(actor)
        for role, actor in [("admin", admin), ("writer", writer), ("teammate", teammate)]
    }
    assert offered == {"admin": choice, "writer": choice, "teammate": choice}, (
        f"New terminal does not offer the connection to everyone who may write to it: {offered}"
    )
    kept_out = {"reader": _offered(reader), "stranger": _offered(stranger)}
    assert kept_out == {"reader": [], "stranger": []}, (
        f"New terminal offers the connection to people who may not create on it: {kept_out}"
    )

    for role, actor in [("writer", writer), ("teammate", teammate)]:
        limits = _limits(actor, connection["id"])
        assert limits.status_code == 200, f"the {role} cannot read the limits: {limits.text}"
        assert (limits.json()["max_cpu"], limits.json()["max_memory"]) == (MAX_CPU, MAX_MEMORY), (
            f"the {role} is shown other limits than the orchestrator's: {limits.json()}"
        )
    refused = {
        "reader": _answer(_limits(reader, connection["id"])),
        "stranger": _answer(_limits(stranger, connection["id"])),
        "unknown connection": _answer(_limits(writer, f"terminals-{uuid.uuid4().hex[:8]}")),
    }
    assert refused == {
        "reader": (403, "Creation is not allowed"),
        "stranger": NO_CONNECTION_ACCESS,
        "unknown connection": NO_CONNECTION,
    }, f"the orchestrator's limits were not refused as expected: {refused}"


def test_a_new_terminal_is_readable_by_its_creator_alone_who_keeps_it_without_write_access(
    admin, make_user, connect_terminals, terminals
):
    creator, colleague = make_user(), make_user()
    connection = connect_terminals(write_grant(creator.id), read_grant(colleague.id))
    template_id, name = str(uuid.uuid4()), f"Bench {uuid.uuid4().hex[:8]}"

    created = _new_terminal(creator, connection["id"], template_id, name=name, scope="user")

    assert created.status_code == 201, f"creating the terminal failed: {created.text}"
    terminal_id = created.json()["terminal_id"]
    assert terminal_id == f"template:{connection['id']}:{template_id}", (
        f"the new terminal is called by {terminal_id}"
    )
    assert (created.json()["id"], created.json()["data"]) == (
        template_id,
        {"name": name, "scope": "user"},
    ), f"the new terminal was answered as {created.json()}"
    kept = {template["id"]: template["data"] for template in terminals.templates()}
    assert kept == {template_id: {"name": name, "scope": "user"}}, (
        f"the orchestrator keeps these terminals: {kept}"
    )
    assert _shares(creator, terminal_id) == [("user", creator.id, "read")], (
        f"a new terminal is shared with {_shares(creator, terminal_id)}"
    )

    only_reading = {"access_grants": [read_grant(creator.id), read_grant(colleague.id)]}
    with admin.client() as client:
        configure_terminals(client, {**connection, "config": only_reading})

    assert _answer(_settings(creator, terminal_id)) == WRITE_REQUIRED, (
        "the creator still manages the terminal after losing write access"
    )
    assert list(_menu(creator)) == [terminal_id], (
        f"the creator lost their own terminal with their write access: {list(_menu(creator))}"
    )
    assert _machine(creator, terminal_id).status_code == 200, (
        f"the creator may no longer use their own terminal: {_machine(creator, terminal_id).text}"
    )
    assert _menu(colleague) == {}, f"someone else's terminal shows up: {list(_menu(colleague))}"


def test_a_terminal_is_created_only_with_a_uuid_and_write_access(
    make_user, connect_terminals, terminals
):
    writer, reader, stranger = make_user(), make_user(), make_user()
    connection = connect_terminals(write_grant(writer.id), read_grant(reader.id))
    fresh = str(uuid.uuid4())

    refused = {
        "no UUID": _new_terminal(writer, connection["id"], "bench-1", name="Bench").status_code,
        "reader": _answer(_new_terminal(reader, connection["id"], fresh, name="Bench")),
        "stranger": _answer(_new_terminal(stranger, connection["id"], fresh, name="Bench")),
        "unknown connection": _answer(_new_terminal(writer, "terminals-none", fresh, name="Bench")),
    }
    assert refused == {
        "no UUID": 422,
        "reader": (403, "Creation is not allowed"),
        "stranger": NO_CONNECTION_ACCESS,
        "unknown connection": NO_CONNECTION,
    }, f"creating a terminal was not refused as expected: {refused}"
    assert terminals.templates() == [], (
        f"a refused creation left a terminal on the orchestrator: {terminals.templates()}"
    )

    created = _new_terminal(writer, connection["id"], fresh, name="Bench")
    assert created.status_code == 201, f"the writer could not create the terminal: {created.text}"
    assert [template["id"] for template in terminals.templates()] == [fresh], (
        f"the orchestrator keeps these terminals: {terminals.templates()}"
    )


def test_a_writer_renames_and_changes_a_terminal_someone_else_created(
    make_user, connect_terminals, terminals
):
    creator, manager, reader = make_user(), make_user(), make_user()
    connection = connect_terminals(
        write_grant(creator.id), write_grant(manager.id), read_grant(reader.id)
    )
    tag = uuid.uuid4().hex[:8]
    with creator.client() as client:
        terminal_id = create_template(client, connection, name=f"Before {tag}")["terminal_id"]
        share_template(client, terminal_id, read_grant(creator.id), read_grant(reader.id))
    assert _menu(reader)[terminal_id]["name"] == f"Before {tag}", (
        f"the reader's menu shows the shared terminal as {_menu(reader)}"
    )

    with manager.client() as client:
        changed = client.patch(
            f"{TEMPLATES}/{terminal_id}", json={"name": f"After {tag}", "memory_limit": "256Mi"}
        )

    assert changed.status_code == 200, f"changing the terminal failed: {changed.text}"
    expected = {"name": f"After {tag}", "scope": "user", "cpu_limit": "1", "memory_limit": "256Mi"}
    assert changed.json()["data"] == expected, f"the change was answered as {changed.json()}"
    assert _settings(manager, terminal_id).json()["data"] == expected, (
        f"the terminal's settings read {_settings(manager, terminal_id).text} after the change"
    )
    assert _menu(reader)[terminal_id]["name"] == f"After {tag}", (
        f"the reader's menu still shows {_menu(reader)[terminal_id]['name']}"
    )
    assert [template["data"] for template in terminals.templates()] == [expected], (
        f"the orchestrator keeps the terminal as {terminals.templates()}"
    )


def test_a_reader_is_refused_every_management_call_a_writer_makes(
    make_user, connect_terminals, terminals
):
    writer, reader = make_user(), make_user()
    connection = connect_terminals(write_grant(writer.id), read_grant(reader.id))
    name = f"Bench {uuid.uuid4().hex[:8]}"
    shares = [("user", reader.id, "read"), ("user", writer.id, "read")]
    with writer.client() as client:
        terminal_id = create_template(client, connection, name=name)["terminal_id"]
        share_template(client, terminal_id, read_grant(writer.id), read_grant(reader.id))

    refused = _management_answers(reader, terminal_id)

    assert refused == {call: WRITE_REQUIRED for call in refused}, (
        f"a person the terminal is only shared with was not refused everywhere: {refused}"
    )
    assert _settings(writer, terminal_id).json()["data"]["name"] == name, (
        f"a refused call changed the terminal: {_settings(writer, terminal_id).text}"
    )
    assert sorted(_shares(writer, terminal_id)) == sorted(shares), (
        f"a refused call changed the shares: {_shares(writer, terminal_id)}"
    )
    assert terminals.instances() == [], f"a refused call started a machine: {terminals.instances()}"
    assert list(_menu(reader)) == [terminal_id], "the reader lost the terminal shared with them"

    allowed = _management_answers(writer, terminal_id)
    assert {call: status for call, (status, _) in allowed.items()} == {
        call: 200 for call in allowed
    }, f"the writer's own calls of the same kind failed: {allowed}"
    assert terminals.templates() == [], (
        f"the writer's deletion left the terminal: {terminals.templates()}"
    )


@pytest.fixture
def two_terminals(make_user, connect_terminals) -> dict:
    """A connection with a writer's two terminals, the first shared with a reader and with an
    outsider who has no access to the connection; an onlooker may read the connection only."""
    people = {
        role: make_user() for role in ("creator", "manager", "reader", "onlooker", "outsider")
    }
    connection = connect_terminals(
        write_grant(people["creator"].id),
        write_grant(people["manager"].id),
        read_grant(people["reader"].id),
        read_grant(people["onlooker"].id),
    )
    tag = uuid.uuid4().hex[:8]
    with people["creator"].client() as client:
        shared = create_template(client, connection, name=f"Shared {tag}")
        private = create_template(client, connection, name=f"Private {tag}", scope=None)
        share_template(
            client,
            shared["terminal_id"],
            read_grant(people["reader"].id),
            read_grant(people["outsider"].id),
        )
    return {**people, "shared": shared, "private": private}


def test_the_menu_shows_a_writer_every_terminal_and_a_reader_the_ones_shared_with_them(
    admin, two_terminals
):
    shared, private = two_terminals["shared"], two_terminals["private"]
    both = {shared["terminal_id"], private["terminal_id"]}

    menus = {
        role: _menu(actor)
        for role, actor in [*two_terminals.items(), ("admin", admin)]
        if isinstance(actor, Actor)
    }

    seen = {role: set(menu) for role, menu in menus.items()}
    assert seen == {
        "creator": both,
        "manager": both,
        "admin": both,
        "reader": {shared["terminal_id"]},
        "onlooker": set(),
        "outsider": set(),
    }, f"the terminal menus show {seen}"
    listed = {
        terminal_id: (entry["name"], entry["template_id"], entry["scope"])
        for terminal_id, entry in menus["manager"].items()
    }
    assert listed == {
        shared["terminal_id"]: (shared["data"]["name"], shared["id"], "user"),
        private["terminal_id"]: (private["data"]["name"], private["id"], None),
    }, f"the terminals are listed as {listed}"


def test_a_terminal_is_used_only_with_a_share_and_access_to_its_connection(two_terminals):
    shared = two_terminals["shared"]["terminal_id"]
    private = two_terminals["private"]["terminal_id"]

    own = _machine(two_terminals["reader"], shared)
    assert own.status_code == 200 and own.json()["status"] == "not_created", (
        f"the reader cannot use the terminal shared with them: {own.status_code} {own.text}"
    )
    unshared = _machine(two_terminals["manager"], private)
    assert unshared.status_code == 200, (
        f"a writer cannot use a terminal nobody shared with them: {unshared.text}"
    )
    refused = {
        "reader, not shared": _answer(_machine(two_terminals["reader"], private)),
        "onlooker": _answer(_machine(two_terminals["onlooker"], shared)),
        "outsider, shared": _answer(_machine(two_terminals["outsider"], shared)),
        "onlooker managing": _answer(_settings(two_terminals["onlooker"], shared)),
        "outsider managing": _answer(_settings(two_terminals["outsider"], shared)),
    }
    assert refused == {
        "reader, not shared": (403, "Terminal access denied"),
        "onlooker": (403, "Terminal access denied"),
        "outsider, shared": NO_CONNECTION_ACCESS,
        "onlooker managing": WRITE_REQUIRED,
        "outsider managing": NO_CONNECTION_ACCESS,
    }, f"the terminals were not refused as expected: {refused}"


def test_a_deleted_terminal_put_back_on_the_orchestrator_is_shared_with_nobody(
    make_user, connect_terminals, terminals
):
    writer, reader = make_user(), make_user()
    connection = connect_terminals(write_grant(writer.id), read_grant(reader.id))
    with writer.client() as client:
        created = create_template(client, connection)
        terminal_id = created["terminal_id"]
        share_template(client, terminal_id, read_grant(reader.id))
        assert list(_menu(reader)) == [terminal_id], "the shared terminal is not in the menu"
        deleted = client.delete(f"{TEMPLATES}/{terminal_id}")
    assert deleted.status_code == 200, f"deleting the terminal failed: {deleted.text}"

    with terminals.client() as service:
        restored = service.post(
            "/api/v1/terminals/templates", json={"id": created["id"], "data": created["data"]}
        )
    assert restored.status_code == 201, f"the orchestrator refused the terminal: {restored.text}"

    assert list(_menu(writer)) == [terminal_id], "the terminal is not back for the writer"
    assert _shares(writer, terminal_id) == [], (
        f"the deleted terminal's shares came back with it: {_shares(writer, terminal_id)}"
    )
    assert _menu(reader) == {}, f"the reader has the deleted terminal again: {_menu(reader)}"
    assert _answer(_machine(reader, terminal_id)) == (403, "Terminal access denied"), (
        "the reader may use the terminal again through the share it had before its deletion"
    )


def test_a_switched_off_connection_lists_and_serves_none_of_its_terminals(
    admin, make_user, connect_terminals
):
    writer, reader = make_user(), make_user()
    connection = connect_terminals(write_grant(writer.id), read_grant(reader.id))
    with writer.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
        share_template(client, terminal_id, read_grant(reader.id))
    people = {"admin": admin, "writer": writer, "reader": reader}
    shown = {role: list(_menu(actor)) for role, actor in people.items()}
    assert shown == {role: [terminal_id] for role in people}, f"the menus show {shown}"

    with admin.client() as client:
        configure_terminals(client, {**connection, "enabled": False})

    shown = {role: list(_menu(actor)) for role, actor in people.items()}
    assert shown == {role: [] for role in people}, (
        f"the terminals of a switched-off connection are still listed: {shown}"
    )
    assert _offered(writer) == [], (
        f"New terminal offers a switched-off connection: {_offered(writer)}"
    )
    answers = {
        "settings": _answer(_settings(writer, terminal_id)),
        "machine": _answer(_machine(reader, terminal_id)),
        "new terminal": _answer(_new_terminal(writer, connection["id"], str(uuid.uuid4()))),
    }
    assert answers == {call: NO_CONNECTION for call in answers}, (
        f"a switched-off connection still answers: {answers}"
    )

    with admin.client() as client:
        configure_terminals(client, connection)
    assert list(_menu(reader)) == [terminal_id], (
        f"the shared terminal did not come back with its connection: {list(_menu(reader))}"
    )


def test_an_id_that_names_no_template_is_refused_before_any_server_is_asked(
    admin, make_user, connect_terminals, preserve
):
    writer = make_user()
    with serving_terminal() as plain:
        connection = connect_terminals(write_grant(writer.id))
        plain_connection = plain.connection(config={"access_grants": [read_grant(writer.id)]})
        with admin.client() as client:
            configure_terminals(client, connection, plain_connection)
        with writer.client() as client:
            created = create_template(client, connection)
        template_id, terminal_id = created["id"], created["terminal_id"]
        assert _settings(writer, terminal_id).status_code == 200, "the terminal is not readable"
        plain.clear()
        malformed = {
            "no template": f"template:{connection['id']}",
            "empty template": f"template:{connection['id']}:",
            "no connection": f"template::{template_id}",
            "unencoded": f"template:{connection['id']}:{template_id} copy",
            "encoded twice": f"template:{connection['id']}:{template_id}%2541",
        }
        elsewhere = {
            "unknown connection": f"template:terminals-{uuid.uuid4().hex[:8]}:{template_id}",
            "plain terminal": f"template:{plain_connection['id']}:{template_id}",
        }

        with writer.client() as client:
            answers = {
                (case, call): _answer(client.get(path.format(id=bad_id)))
                for case, bad_id in {**malformed, **elsewhere}.items()
                for call, path in [
                    ("settings", TEMPLATES + "/{id}"),
                    ("machine", TEMPLATES + "/{id}/instance"),
                    ("files", "/api/v1/terminals/{id}/files/cwd"),
                ]
            }

        expected = {
            (case, call): (400, "Invalid template terminal ID")
            if case in malformed
            else NO_CONNECTION
            for case, call in answers
        }
        assert answers == expected, f"the ids were not refused as expected: {answers}"
        assert plain.received == [], (
            f"a plain terminal was asked for a template: {[r.path for r in plain.received]}"
        )


def test_the_orchestrators_own_refusal_reaches_the_person_in_its_words(
    make_user, connect_terminals, terminals
):
    writer = make_user()
    connection = connect_terminals(write_grant(writer.id))
    with writer.client() as client:
        created = create_template(client, connection)
    profile = f"lab-{uuid.uuid4().hex[:8]}"

    again = _new_terminal(writer, connection["id"], created["id"], name="Twin")
    unknown_network = _new_terminal(
        writer, connection["id"], str(uuid.uuid4()), name="Lab", network_profile=profile
    )
    with writer.client() as client:
        moved = client.patch(
            f"{TEMPLATES}/{created['terminal_id']}", json={"network_profile": profile}
        )

    answers = {
        "same id": _answer(again),
        "create": _answer(unknown_network),
        "change": _answer(moved),
    }
    assert answers == {
        "same id": (409, "Template already exists"),
        "create": (400, f"Unknown network profile: '{profile}'"),
        "change": (400, f"Unknown network profile: '{profile}'"),
    }, f"the orchestrator's refusals reached the person as {answers}"
    kept = [template["data"] for template in terminals.templates()]
    assert kept == [created["data"]], f"a refused call changed what the orchestrator keeps: {kept}"


def test_a_terminal_the_orchestrator_lost_is_gone_for_everyone(
    make_user, connect_terminals, terminals
):
    writer, reader = make_user(), make_user()
    connection = connect_terminals(write_grant(writer.id), read_grant(reader.id))
    with writer.client() as client:
        lost = create_template(client, connection)
        kept = create_template(client, connection)
        for terminal in (lost, kept):
            share_template(client, terminal["terminal_id"], read_grant(reader.id))
    assert set(_menu(reader)) == {lost["terminal_id"], kept["terminal_id"]}, (
        f"the reader's menu shows {list(_menu(reader))}"
    )

    with terminals.client() as service:
        removed = service.delete(f"/api/v1/terminals/templates/{lost['id']}")
    assert removed.status_code == 200, f"the orchestrator kept the terminal: {removed.text}"

    assert list(_menu(reader)) == [kept["terminal_id"]], (
        f"the reader's menu shows {list(_menu(reader))} after the orchestrator lost a terminal"
    )
    gone = (404, "Template not found")
    answers = {
        "settings": _answer(_settings(writer, lost["terminal_id"])),
        "machine": _answer(_machine(reader, lost["terminal_id"])),
    }
    with writer.client() as client:
        answers["deletion"] = _answer(client.delete(f"{TEMPLATES}/{lost['terminal_id']}"))
        answers["sharing"] = _answer(
            client.post(f"{TEMPLATES}/{lost['terminal_id']}/access", json={"access_grants": []})
        )
    assert answers == {call: gone for call in answers}, (
        f"a terminal the orchestrator no longer has is answered for with {answers}"
    )
    assert _machine(reader, kept["terminal_id"]).status_code == 200, "the other terminal broke"


def test_a_redirecting_orchestrator_is_asked_with_the_connections_key_and_not_followed(
    admin, make_user, preserve, listener
):
    writer = make_user()
    key = f"sk-{uuid.uuid4().hex}"
    listener.route("GET", "/elsewhere", json_answer({"status": True, "max_cpu": "64"}))
    with serving_terminal() as server:
        server.route("GET", "/api/v1/terminals/status", json_answer({"status": True}))
        connection = server.connection(
            server_type="orchestrator",
            key=key,
            auth_type="bearer",
            config={"access_grants": [write_grant(writer.id)]},
        )
        preserve(TERMINAL_SERVERS_CONFIG)
        with admin.client() as client:
            configure_terminals(client, connection)
        server.clear()

        answered = _limits(writer, connection["id"])
        server.redirect("/api/v1/terminals/status", f"{listener.base_url}/elsewhere")
        redirected = _limits(writer, connection["id"])

        assert _answer(answered) == (200, None), f"the limits were not read: {answered.text}"
        asked = server.requests_to("/api/v1/terminals/status")
        keys = [request.headers.get("authorization") for request in asked]
        assert keys == [f"Bearer {key}"] * 2, (
            f"the orchestrator was not asked twice with the connection's key: {keys}"
        )
        assert redirected.status_code == 502, (
            f"a redirect from the orchestrator was answered {redirected.status_code} "
            f"{redirected.text}"
        )
        assert listener.requests_to("/elsewhere") == [], (
            "the redirect was followed to another server"
        )


def test_a_connection_without_a_key_asks_the_orchestrator_nothing(
    admin, make_user, connect_terminals, terminals
):
    writer = make_user()
    connection = connect_terminals(write_grant(writer.id))
    with writer.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
    assert list(_menu(writer)) == [terminal_id], "the terminal is not in its creator's menu"
    assert _limits(writer, connection["id"]).status_code == 200, "the limits are not readable"

    with admin.client() as client:
        configure_terminals(client, {**connection, "key": ""})

    key_required = (503, "A Terminals server key is required")
    answers = {
        "limits": _answer(_limits(writer, connection["id"])),
        "settings": _answer(_settings(writer, terminal_id)),
        "machine": _answer(_machine(writer, terminal_id)),
        "new terminal": _answer(_new_terminal(writer, connection["id"], str(uuid.uuid4()))),
    }
    assert answers == {call: key_required for call in answers}, (
        f"a connection without a key was answered for with {answers}"
    )
    assert _menu(writer) == {}, f"the menu lists terminals nobody can reach: {_menu(writer)}"
    assert len(terminals.templates()) == 1, (
        f"a call without a key changed the orchestrator: {terminals.templates()}"
    )


@pytest.fixture
def stoppable_terminals() -> Iterator[Terminals]:
    """A Terminals service of the test's own, which it may stop."""
    with serving_terminals() as service:
        yield service


@pytest.mark.slow
def test_a_stopped_orchestrator_is_reported_while_the_other_terminals_stay_listed(
    admin, make_user, preserve, stoppable_terminals
):
    writer = make_user()
    grants = {"access_grants": [write_grant(writer.id)]}
    connection = stoppable_terminals.connection(config=grants)
    with serving_terminal() as plain:
        plain_connection = plain.connection(config={"access_grants": [read_grant(writer.id)]})
        preserve(TERMINAL_SERVERS_CONFIG)
        with admin.client() as client:
            configure_terminals(client, connection, plain_connection)
        with writer.client() as client:
            terminal_id = create_template(client, connection)["terminal_id"]
        assert set(_menu(writer)) == {terminal_id, plain_connection["id"]}, (
            f"the menu shows {list(_menu(writer))} while the orchestrator runs"
        )
        assert _settings(writer, terminal_id).status_code == 200, "the terminal is not readable"

        stoppable_terminals.stop()

        assert list(_menu(writer)) == [plain_connection["id"]], (
            f"the menu shows {list(_menu(writer))} while the orchestrator is stopped"
        )
        unavailable = (503, "Terminals is unavailable")
        answers = {
            "limits": _answer(_limits(writer, connection["id"])),
            "settings": _answer(_settings(writer, terminal_id)),
            "machine": _answer(_machine(writer, terminal_id)),
            "new terminal": _answer(_new_terminal(writer, connection["id"], str(uuid.uuid4()))),
        }
        with writer.client() as client:
            answers["deletion"] = _answer(client.delete(f"{TEMPLATES}/{terminal_id}"))
        assert answers == {call: unavailable for call in answers}, (
            f"a stopped orchestrator was answered for with {answers}"
        )
        assert _offered(writer) == [{"id": connection["id"], "name": connection["name"]}], (
            "New terminal no longer offers the connection of a stopped orchestrator"
        )


@pytest.fixture
def on_instance(instance_with, terminals) -> Callable[[dict], tuple]:
    """`on_instance(env)`: an instance with that setting, its admin and a way to save the
    Terminals connection there; the connections it had are put back afterwards."""
    restores = []

    def boot(env: dict[str, str]) -> tuple[Actor, Callable[..., Actor], Callable[..., dict]]:
        extra = instance_with(env)
        extra_admin = admin_of(extra)
        with extra_admin.client() as client:
            before = client.get(TERMINAL_SERVERS_CONFIG[0])
        before.raise_for_status()
        restores.append((extra_admin, before.json()))

        def connect(*grants: dict, **fields) -> dict:
            connection = terminals.connection(config={"access_grants": list(grants)}, **fields)
            with extra_admin.client() as client:
                configure_terminals(client, connection)
            return connection

        return extra_admin, lambda: create_user(extra), connect

    yield boot
    for extra_admin, before in restores:
        with extra_admin.client() as client:
            restored = client.post(TERMINAL_SERVERS_CONFIG[1], json=before)
        assert restored.status_code == 200, restored.text


@pytest.mark.slow
def test_without_the_bypass_an_admin_needs_a_grant_once_the_connection_has_grants(on_instance):
    strict_admin, make_strict_user, connect = on_instance({"BYPASS_ADMIN_ACCESS_CONTROL": "false"})
    writer = make_strict_user()

    ungranted = connect()
    choice = [{"id": ungranted["id"], "name": ungranted["name"]}]
    assert _offered(strict_admin) == choice, (
        f"an admin cannot create on a connection without grants: {_offered(strict_admin)}"
    )
    with strict_admin.client() as client:
        own = create_template(client, ungranted)["terminal_id"]
    assert list(_menu(strict_admin)) == [own], f"the admin's menu shows {list(_menu(strict_admin))}"

    granted = connect(write_grant(writer.id), id=ungranted["id"])
    with writer.client() as client:
        theirs = create_template(client, granted)["terminal_id"]
    assert set(_menu(writer)) == {own, theirs}, f"the writer's menu shows {list(_menu(writer))}"

    assert _offered(strict_admin) == [], (
        f"an admin without the bypass is offered the connection: {_offered(strict_admin)}"
    )
    assert _menu(strict_admin) == {}, (
        f"an admin without the bypass sees terminals of a connection granted to someone else: "
        f"{list(_menu(strict_admin))}"
    )
    answers = {
        "settings": _answer(_settings(strict_admin, theirs)),
        "own terminal": _answer(_machine(strict_admin, own)),
        "limits": _answer(_limits(strict_admin, granted["id"])),
        "new terminal": _answer(_new_terminal(strict_admin, granted["id"], str(uuid.uuid4()))),
    }
    assert answers == {call: NO_CONNECTION_ACCESS for call in answers}, (
        f"an admin without the bypass was answered {answers}"
    )

    connect(write_grant(writer.id), read_grant(strict_admin.id), id=ungranted["id"])
    assert list(_menu(strict_admin)) == [own], (
        f"with a read grant the admin's menu shows {list(_menu(strict_admin))}"
    )
    assert _answer(_settings(strict_admin, theirs)) == WRITE_REQUIRED, (
        "an admin with a read grant and no bypass manages someone else's terminal"
    )


def test_with_the_bypass_an_admin_manages_a_connection_granted_to_someone_else(
    admin, make_user, connect_terminals
):
    writer = make_user()
    connection = connect_terminals(write_grant(writer.id))
    with writer.client() as client:
        theirs = create_template(client, connection)["terminal_id"]

    assert _offered(admin) == [{"id": connection["id"], "name": connection["name"]}], (
        f"New terminal offers the admin {_offered(admin)}"
    )
    assert list(_menu(admin)) == [theirs], f"the admin's menu shows {list(_menu(admin))}"
    with admin.client() as client:
        renamed = client.patch(f"{TEMPLATES}/{theirs}", json={"name": "Renamed by the admin"})
    assert renamed.status_code == 200, f"the admin could not rename the terminal: {renamed.text}"
    assert _menu(writer)[theirs]["name"] == "Renamed by the admin", (
        f"the writer's menu shows {_menu(writer)[theirs]['name']}"
    )
    assert _machine(admin, theirs).status_code == 200, (
        f"the admin cannot use a terminal nobody shared with them: {_machine(admin, theirs).text}"
    )


@pytest.mark.slow
def test_with_tool_servers_off_every_template_call_is_refused(
    admin, connect_terminals, on_instance, terminals
):
    connection = connect_terminals()
    with admin.client() as client:
        created = create_template(client, connection)
    terminal_id = created["terminal_id"]
    off_admin, _, connect = on_instance({"ENABLE_TOOL_SERVERS": "false"})
    connect(id=connection["id"])
    disabled = (403, "Tool servers are disabled")

    with off_admin.client() as client:
        answers = {
            "limits": _answer(client.get(f"{TEMPLATES}/connections/{connection['id']}/status")),
            "machine": _answer(client.get(f"{TEMPLATES}/{terminal_id}/instance")),
            "new terminal": _answer(
                client.post(
                    TEMPLATES,
                    json={"id": str(uuid.uuid4()), "connection_id": connection["id"], "data": {}},
                )
            ),
        }
    answers.update(_management_answers(off_admin, terminal_id))
    assert answers == {call: disabled for call in answers}, (
        f"template calls were answered with tool servers off: {answers}"
    )
    assert _offered(off_admin) == [], (
        f"New terminal offers a connection with tool servers off: {_offered(off_admin)}"
    )
    assert _menu(off_admin) == {}, (
        f"the menu lists terminals with tool servers off: {_menu(off_admin)}"
    )
    assert [template["id"] for template in terminals.templates()] == [created["id"]], (
        f"a refused call changed the orchestrator: {terminals.templates()}"
    )

    working = _management_answers(admin, terminal_id)
    assert {status for status, _ in working.values()} == {200}, (
        f"the same calls fail with tool servers on: {working}"
    )
