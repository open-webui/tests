"""Journey: a terminal of a Terminals connection is shared with people and groups, for reading.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). A person who may write to a
Terminals connection shares a terminal from its Access dialog. Sharing replaces who has the
terminal: the people named see it in their terminal menu and get a machine of it, everyone
dropped loses both. A share with a group is every member's until they leave the group. A share
is for reading and names a person or a group; sharing with everyone, a write share or another
kind of audience is refused and leaves the shares as they were. A person who is no admin shares
only with the audiences their permissions allow (Access Grants: users, groups). Every change of
a terminal (created, renamed, shared, unshared, deleted) is pushed to the tabs of the person
who made it and of everyone who has or just lost the terminal, so their menus reload; nobody
else hears of it. With Redis the push reaches a tab on another worker.

The real Terminals service (open-webui/terminals) runs on Docker (`harness.terminals`); without
its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Each
mutation below, made in a backend copy, turns its test red. An empty share list not stored
fails the replacing test, as does a terminal usable without a share (which also fails the group
test). The check for everyone dropped (403 from the permission check instead of 400), or a
write share accepted, fails the audience test. Everyone sharing like an admin fails the
permissions test. Group shares dropped before they are stored fails the group test. Only the
person who made the change told fails the push test at the first share, and the people who just
lost the terminal left out fails it at the unshare. The push kept on the worker that made the
change fails the two-worker test and leaves the push test green.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Iterator

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor, admin_of, create_user
from harness.backends import RedisProcess, wait_until
from harness.instance import LaunchedInstance
from harness.socket_client import SocketSession, connected
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    write_grant,
)
from harness.terminals import create_template, execute, output_of, share_template

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TEMPLATES = "/api/v1/terminals/templates"
NOT_SHARED = (403, "Terminal access denied")
AUDIENCE_REFUSED = (403, "You do not have permission to share with the requested audience")


def _menu(actor: Actor) -> list[str]:
    """The ids of the terminals the account's terminal menu offers."""
    with actor.client() as client:
        listed = client.get("/api/v1/terminals/")
    assert listed.status_code == 200, listed.text
    return [entry["id"] for entry in listed.json()]


def _answer(answer: httpx.Response) -> tuple[int, str | None]:
    """The status and the message a refused call shows in the dialog."""
    body = answer.json() if answer.content else {}
    return answer.status_code, body.get("detail") if isinstance(body, dict) else None


def _machine(actor: Actor, terminal_id: str) -> tuple[int, str | None]:
    """How the account's machine of the terminal answers the terminal menu, before any use."""
    with actor.client() as client:
        return _answer(client.get(f"{TEMPLATES}/{terminal_id}/instance"))


def _shares(actor: Actor, terminal_id: str) -> list[tuple[str, str, str]]:
    """The terminal's shares as its Access dialog reads them."""
    with actor.client() as client:
        shares = client.get(f"{TEMPLATES}/{terminal_id}/access")
    assert shares.status_code == 200, f"the terminal's shares could not be read: {shares.text}"
    return sorted(
        (share["principal_type"], share["principal_id"], share["permission"])
        for share in shares.json()
    )


def _share(actor: Actor, terminal_id: str, *grants: dict) -> tuple[int, str | None]:
    with actor.client() as client:
        return _answer(
            client.post(f"{TEMPLATES}/{terminal_id}/access", json={"access_grants": list(grants)})
        )


def test_sharing_replaces_who_sees_and_uses_a_terminal(make_user, connect_terminals):
    writer, first, second = make_user(), make_user(), make_user()
    connection = connect_terminals(
        write_grant(writer.id), read_grant(first.id), read_grant(second.id)
    )
    with writer.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]

    assert _share(writer, terminal_id, read_grant(first.id)) == (200, None), "sharing failed"
    assert _shares(writer, terminal_id) == [("user", first.id, "read")], (
        f"the Access dialog shows {_shares(writer, terminal_id)} after sharing with one person"
    )
    seen = {"first": _menu(first), "second": _menu(second)}
    assert seen == {"first": [terminal_id], "second": []}, f"the menus show {seen}"
    assert _machine(first, terminal_id)[0] == 200, "the person it is shared with cannot use it"
    assert _machine(second, terminal_id) == NOT_SHARED, "a person it is not shared with uses it"

    assert _share(writer, terminal_id, read_grant(second.id)) == (200, None), "sharing failed"
    assert _shares(writer, terminal_id) == [("user", second.id, "read")], (
        f"the Access dialog shows {_shares(writer, terminal_id)} after sharing with someone else"
    )
    seen = {"first": _menu(first), "second": _menu(second)}
    assert seen == {"first": [], "second": [terminal_id]}, (
        f"the menus show {seen} after the share moved to the second person"
    )
    assert _machine(first, terminal_id) == NOT_SHARED, "the person dropped from it still uses it"
    assert _machine(second, terminal_id)[0] == 200, "the person it is now shared with cannot use it"

    assert _share(writer, terminal_id) == (200, None), "taking the shares away failed"
    assert _shares(writer, terminal_id) == [], (
        f"the Access dialog shows {_shares(writer, terminal_id)} after unsharing"
    )
    assert _menu(second) == [], f"an unshared terminal stays in the menu: {_menu(second)}"
    assert _machine(second, terminal_id) == NOT_SHARED, "an unshared terminal is still used"
    assert _menu(writer) == [terminal_id], "the writer lost the terminal by unsharing it"


def test_a_share_names_a_person_or_a_group_and_is_for_reading(admin, make_user, connect_terminals):
    writer, reader, other = make_user(), make_user(), make_user()
    team = make_group(admin, [other])
    connection = connect_terminals(
        write_grant(writer.id), read_grant(reader.id), read_grant(other.id)
    )
    with writer.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
        share_template(client, terminal_id, read_grant(reader.id))
    shared_before = [("user", reader.id, "read")]
    public = (400, "Public terminal sharing is not allowed")
    read_only = (400, "Template access grants must be read-only")

    refused = {
        "everyone": _share(writer, terminal_id, read_grant("*")),
        "everyone beside a person": _share(
            writer, terminal_id, read_grant(reader.id), read_grant("*")
        ),
        "nobody named": _share(writer, terminal_id, read_grant("")),
        "another kind of audience": _share(writer, terminal_id, grant("role", "user", "read")),
        "write for a person": _share(writer, terminal_id, write_grant(other.id)),
        "write for a group": _share(writer, terminal_id, grant("group", team, "write")),
        "write beside read": _share(
            writer, terminal_id, read_grant(reader.id), write_grant(reader.id)
        ),
    }
    assert refused == {
        "everyone": public,
        "everyone beside a person": public,
        "nobody named": public,
        "another kind of audience": public,
        "write for a person": read_only,
        "write for a group": read_only,
        "write beside read": read_only,
    }, f"the shares were not refused as expected: {refused}"
    assert _shares(writer, terminal_id) == shared_before, (
        f"a refused share changed the terminal's shares: {_shares(writer, terminal_id)}"
    )
    seen = {"reader": _menu(reader), "other": _menu(other)}
    assert seen == {"reader": [terminal_id], "other": []}, (
        f"after the refused shares the menus show {seen}"
    )
    assert _answer(_settings_of(other, terminal_id)) == (
        403,
        "Terminal connection write access required",
    ), "a refused write share lets its audience manage the terminal"

    accepted = _share(writer, terminal_id, read_grant(reader.id), grant("group", team, "read"))
    assert accepted == (200, None), f"a share with a person and a group was refused: {accepted}"
    assert _shares(writer, terminal_id) == sorted([*shared_before, ("group", team, "read")]), (
        f"the Access dialog shows {_shares(writer, terminal_id)}"
    )


def _settings_of(actor: Actor, terminal_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"{TEMPLATES}/{terminal_id}")


def test_a_terminal_shared_with_a_group_is_its_members_until_they_leave(
    admin, make_user, connect_terminals, terminals
):
    member, colleague = make_user(), make_user()
    team = make_group(admin, [member])
    connection = connect_terminals(read_grant(member.id), read_grant(colleague.id))
    tag = uuid.uuid4().hex[:8]
    with admin.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
        share_template(client, terminal_id, grant("group", team, "read"))

    seen = {"member": _menu(member), "colleague": _menu(colleague)}
    assert seen == {"member": [terminal_id], "colleague": []}, (
        f"with the terminal shared with the member's group the menus show {seen}"
    )
    with member.client() as client:
        printed = output_of(execute(client, terminal_id, f"echo team-{tag}"))
    assert printed.strip() == f"team-{tag}", f"the member's command printed {printed}"
    assert [machine["context_id"] for machine in terminals.instances()] == [f"user:{member.id}"], (
        f"the command ran in these machines: {terminals.instances()}"
    )
    assert _machine(colleague, terminal_id) == NOT_SHARED, "a person outside the group uses it"

    with admin.client() as client:
        left = client.post(f"/api/v1/groups/id/{team}/users/remove", json={"user_ids": [member.id]})
    assert left.status_code == 200, f"removing the member from the group failed: {left.text}"

    assert _menu(member) == [], f"the terminal stays in a former member's menu: {_menu(member)}"
    with member.client() as client:
        after = execute(client, terminal_id, "echo still here")
    assert _answer(after) == NOT_SHARED, (
        f"a former member still runs commands: {after.status_code} {after.text}"
    )


def test_a_person_shares_only_with_the_audiences_their_permissions_allow(
    admin, make_user, connect_terminals, preserve
):
    plain, trusted, reader = make_user(), make_user(), make_user()
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        closed = {"allow_users": False, "allow_groups": False}
        saved = client.post(
            "/api/v1/users/default/permissions", json={**permissions, "access_grants": closed}
        )
    assert saved.status_code == 200, f"saving the default permissions failed: {saved.text}"
    make_group(admin, [trusted], permissions={"access_grants": {"allow_users": True}})
    team = make_group(admin, [reader])
    connection = connect_terminals(
        write_grant(plain.id), write_grant(trusted.id), read_grant(reader.id)
    )
    with plain.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
    to_person, to_group = read_grant(reader.id), grant("group", team, "read")

    refused = {
        "plain, person": _share(plain, terminal_id, to_person),
        "plain, group": _share(plain, terminal_id, to_group),
        "trusted, group": _share(trusted, terminal_id, to_group),
        "trusted, person and group": _share(trusted, terminal_id, to_person, to_group),
    }
    assert refused == {case: AUDIENCE_REFUSED for case in refused}, (
        f"sharing past the person's permissions was answered {refused}"
    )
    assert _shares(plain, terminal_id) == [("user", plain.id, "read")], (
        f"a refused share changed the terminal's shares: {_shares(plain, terminal_id)}"
    )
    assert _menu(reader) == [], f"a refused share reached its audience: {_menu(reader)}"

    assert _share(trusted, terminal_id, to_person) == (200, None), (
        "a person allowed to share with users could not share the terminal with one"
    )
    assert _menu(reader) == [terminal_id], f"the reader's menu shows {_menu(reader)}"
    assert _share(admin, terminal_id, to_group) == (200, None), (
        "an admin could not share the terminal with a group"
    )
    assert _shares(plain, terminal_id) == [("group", team, "read")], (
        f"the Access dialog shows {_shares(plain, terminal_id)}"
    )
    assert _share(plain, terminal_id) == (200, None), (
        "a person without sharing permissions could not take the shares away"
    )
    assert _menu(reader) == [], f"an unshared terminal stays in the menu: {_menu(reader)}"


def _changes(tab: SocketSession) -> int:
    """How often the tab was told to reload its terminals."""
    return sum(
        1
        for event in list(tab.events)
        if (event.get("data") or {}).get("type") == "terminals:changed"
    )


def _expect_changes(step: str, **expected: tuple[SocketSession, int]) -> None:
    for who, (tab, count) in expected.items():
        assert wait_until(lambda: _changes(tab) >= count, 10), (
            f"after {step} the {who}'s tab was told {_changes(tab)} times to reload its "
            f"terminals, not {count} times"
        )


def test_every_change_of_a_terminal_is_pushed_to_the_tabs_of_those_who_have_it(
    make_user, connect_terminals
):
    writer, reader, bystander = make_user(), make_user(), make_user()
    connection = connect_terminals(
        write_grant(writer.id), read_grant(reader.id), read_grant(bystander.id)
    )
    with (
        connected(writer) as writers,
        connected(reader) as readers,
        connected(bystander) as bystanders,
        writer.client() as client,
    ):
        terminal_id = create_template(client, connection)["terminal_id"]
        _expect_changes("the creation", writer=(writers, 1))

        share_template(client, terminal_id, read_grant(reader.id))
        _expect_changes("the share", writer=(writers, 2), reader=(readers, 1))

        renamed = client.patch(f"{TEMPLATES}/{terminal_id}", json={"name": "Renamed"})
        assert renamed.status_code == 200, f"renaming failed: {renamed.text}"
        _expect_changes("the rename", writer=(writers, 3), reader=(readers, 2))

        share_template(client, terminal_id)
        _expect_changes("the unshare", writer=(writers, 4), reader=(readers, 3))

        renamed = client.patch(f"{TEMPLATES}/{terminal_id}", json={"name": "Renamed again"})
        assert renamed.status_code == 200, f"renaming failed: {renamed.text}"
        _expect_changes("the rename of the unshared terminal", writer=(writers, 5))

        share_template(client, terminal_id, read_grant(reader.id))
        _expect_changes("the second share", writer=(writers, 6), reader=(readers, 4))

        deleted = client.delete(f"{TEMPLATES}/{terminal_id}")
        assert deleted.status_code == 200, f"deleting failed: {deleted.text}"
        _expect_changes("the deletion", writer=(writers, 7), reader=(readers, 5))

        # one change for all three: a tab gets its events in order, so anything it was
        # wrongly sent before is counted by the time this one arrives
        other_id = create_template(client, connection)["terminal_id"]
        share_template(client, other_id, read_grant(reader.id), read_grant(bystander.id))
        _expect_changes(
            "a share with all three",
            writer=(writers, 9),
            reader=(readers, 6),
            bystander=(bystanders, 1),
        )
        counted = {
            "writer": _changes(writers),
            "reader": _changes(readers),
            "bystander": _changes(bystanders),
        }
        assert counted == {"writer": 9, "reader": 6, "bystander": 1}, (
            f"tabs were told to reload terminals that did not change for them: {counted}"
        )


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[RedisProcess]:
    server = RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def workers(shared_redis, instance_with) -> tuple[LaunchedInstance, LaunchedInstance]:
    """Two instances on one database and one Redis, as two workers of one deployment."""
    shared = {"REDIS_URL": shared_redis.url, "WEBSOCKET_MANAGER": "redis"}
    near = instance_with({**shared, "WEBUI_NAME": "worker-a"})
    far = instance_with({**shared, "WEBUI_NAME": "worker-b", "DATABASE_URL": near.database_url})
    return near, far


@pytest.mark.slow
def test_a_change_made_on_one_worker_reaches_a_tab_on_another(workers, terminals):
    near, far = workers
    writer, reader = create_user(near), create_user(near)
    reader_on_far = dataclasses.replace(reader, base_url=far.base_url)
    grants = {"access_grants": [write_grant(writer.id), read_grant(reader.id)]}
    connection = terminals.connection(config=grants)
    with admin_of(near).client() as admin_client:
        before = admin_client.get(TERMINAL_SERVERS_CONFIG[0])
        assert before.status_code == 200, before.text
        configure_terminals(admin_client, connection)
        try:
            with (
                connected(writer) as writers,
                connected(reader_on_far) as readers,
                writer.client() as client,
            ):
                terminal_id = create_template(client, connection)["terminal_id"]
                _expect_changes("the creation", writer=(writers, 1))
                assert _menu(reader_on_far) == [], (
                    f"the other worker lists an unshared terminal: {_menu(reader_on_far)}"
                )

                share_template(client, terminal_id, read_grant(reader.id))

                _expect_changes("the share", writer=(writers, 2), reader=(readers, 1))
                assert _menu(reader_on_far) == [terminal_id], (
                    f"told to reload, the reader's tab gets {_menu(reader_on_far)} "
                    "from its own worker"
                )
        finally:
            restored = admin_client.post(TERMINAL_SERVERS_CONFIG[1], json=before.json())
            assert restored.status_code == 200, restored.text
