"""Journey: what a group's permissions give its members, combined with the defaults and each other.

Permissions are additive, as the RBAC docs put it: a member gets the default permissions and every
switch any of their groups turns on, and no group can take away what the defaults or another
group grant. So a member of two groups that disagree ends up with the switch on, whichever group
was made first, and a group that switches a default off leaves its members with it. The server
holds them to that: a group's Knowledge Access lets its member create a knowledge base the
defaults refuse, and removing them from the group refuses them again. Changing a group's
permissions or members tells each affected member's open tab to reload its access (an
`access:updated` message, then a reconnect); a member of another group hears nothing, and an edit
that leaves the permissions as they were tells nobody.

Discriminates: in a backend copy, `combine_permissions` letting the later group's value win (in
place of the most permissive one) turns the two-group and switched-off tests red, and
`refresh_group_sessions` doing nothing turns the open-tab tests red.
"""

from __future__ import annotations

import threading
import uuid
from typing import Iterator

import pytest

from harness.access import make_group
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

DEFAULTS = "/api/v1/users/default/permissions"
NOTICE_TIMEOUT = 15


@pytest.fixture
def defaults(admin, preserve) -> Iterator[dict]:
    """The default permissions as they stand, restored afterwards."""
    preserve("permissions")
    with admin.client() as client:
        current = client.get(DEFAULTS)
    assert current.status_code == 200, current.text
    yield current.json()


def set_default(admin, defaults: dict, section: str, key: str, value: bool) -> None:
    changed = {**defaults, section: {**defaults[section], key: value}}
    with admin.client() as client:
        saved = client.post(DEFAULTS, json=changed)
    assert saved.status_code == 200, saved.text
    defaults[section] = changed[section]


@pytest.fixture
def groups(admin) -> Iterator[list[str]]:
    made: list[str] = []
    yield made
    with admin.client() as client:
        for group_id in made:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _group(admin, groups: list[str], members, permissions: dict) -> str:
    group_id = make_group(admin, members, permissions)
    groups.append(group_id)
    return group_id


def effective(actor) -> dict:
    """The permissions the web client reads, from both places it reads them."""
    with actor.client() as client:
        listed = client.get("/api/v1/users/permissions")
        session = client.get("/api/v1/auths/")
    assert listed.status_code == 200, listed.text
    assert session.status_code == 200, session.text
    assert listed.json() == session.json()["permissions"]
    return listed.json()


def creates_knowledge(actor) -> bool:
    form = {"name": f"Group notes {uuid.uuid4().hex[:6]}", "description": ""}
    with actor.client() as client:
        created = client.post("/api/v1/knowledge/create", json=form)
    assert created.status_code in (200, 401), created.text
    return created.status_code == 200


def reads_notes(actor) -> bool:
    with actor.client() as client:
        listed = client.get("/api/v1/notes/")
    assert listed.status_code in (200, 401), listed.text
    return listed.status_code == 200


GRANTING = {
    "workspace": {"knowledge": True},
    "sharing": {"notes": True},
    "chat": {"controls": True},
    "features": {"web_search": True},
}
DENYING = {
    "workspace": {"knowledge": False},
    "sharing": {"notes": False},
    "chat": {"controls": False},
    "features": {"web_search": False},
}


@pytest.mark.parametrize("first", ["granting", "denying"])
def test_a_member_of_two_disagreeing_groups_gets_every_switch_either_turns_on(
    admin, make_user, defaults, groups, first
):
    for section, switches in DENYING.items():
        for key, value in switches.items():
            set_default(admin, defaults, section, key, value)
    member, denied_only = make_user(), make_user()
    order = [(GRANTING, [member]), (DENYING, [member, denied_only])]
    for permissions, members in order if first == "granting" else reversed(order):
        _group(admin, groups, members, permissions)

    granted = effective(member)
    refused = effective(denied_only)

    for section, switches in GRANTING.items():
        for key in switches:
            assert granted[section][key] is True, f"{section}.{key} lost to the denying group"
            assert refused[section][key] is False, f"{section}.{key} reached the denying group"
    assert creates_knowledge(member)
    assert not creates_knowledge(denied_only)


def test_a_group_that_switches_a_default_off_does_not_take_it_away(
    admin, make_user, defaults, groups
):
    set_default(admin, defaults, "features", "notes", True)
    member = make_user()
    _group(admin, groups, [member], {"features": {"notes": False}, "chat": {"file_upload": False}})

    permissions = effective(member)

    assert permissions["features"]["notes"] is True
    assert permissions["chat"]["file_upload"] is True
    assert reads_notes(member)


def test_with_the_default_off_too_the_switch_stays_off(admin, make_user, defaults, groups):
    set_default(admin, defaults, "features", "notes", False)
    member = make_user()
    _group(admin, groups, [member], {"features": {"notes": False}})

    assert effective(member)["features"]["notes"] is False
    assert not reads_notes(member)


def test_the_server_lets_a_member_past_the_check_only_while_in_the_group(
    admin, make_user, defaults, groups
):
    set_default(admin, defaults, "workspace", "knowledge", False)
    member, outsider = make_user(), make_user()
    group_id = _group(admin, groups, [member], {"workspace": {"knowledge": True}})
    assert creates_knowledge(member)
    assert not creates_knowledge(outsider)

    with admin.client() as client:
        removed = client.post(
            f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [member.id]}
        )
    assert removed.status_code == 200, removed.text

    assert not creates_knowledge(member)


# what an open tab hears when its account's group changes


class Tab:
    """A member's open tab: whether it was told to reload its access, and whether it was dropped."""

    def __init__(self, session: SocketSession):
        self.session = session
        self.told = threading.Event()
        self.dropped = threading.Event()
        session.client.on("access:updated", lambda *_: self.told.set())
        session.client.on("disconnect", lambda *_: self.dropped.set())

    def was_told(self) -> bool:
        return self.told.wait(NOTICE_TIMEOUT) and self.dropped.wait(NOTICE_TIMEOUT)

    def heard_nothing(self) -> bool:
        # bounded: the member's notice arrives within this when it is sent at all
        return not self.told.wait(2) and self.session.client.connected


def _update_group(admin, group_id: str, **changes) -> None:
    with admin.client() as client:
        stored = client.get(f"/api/v1/groups/id/{group_id}").json()
        form = {"name": stored["name"], "description": stored["description"], **changes}
        updated = client.post(f"/api/v1/groups/id/{group_id}/update", json=form)
    assert updated.status_code == 200, updated.text


def test_a_permission_change_tells_the_members_open_tab_and_no_one_else(admin, make_user, groups):
    member, outsider = make_user(), make_user()
    group_id = _group(admin, groups, [member], {"workspace": {"knowledge": False}})
    _group(admin, groups, [outsider], {})

    with connected(member) as member_socket, connected(outsider) as outsider_socket:
        member_tab, outsider_tab = Tab(member_socket), Tab(outsider_socket)
        _update_group(admin, group_id, permissions={"workspace": {"knowledge": True}})

        assert member_tab.was_told(), "the member's open tab was not told to reload its access"
        assert outsider_tab.heard_nothing()


def test_an_edit_that_keeps_the_permissions_tells_nobody(admin, make_user, groups):
    member = make_user()
    permissions = {"workspace": {"knowledge": True}}
    group_id = _group(admin, groups, [member], permissions)

    with connected(member) as member_socket:
        member_tab = Tab(member_socket)
        _update_group(admin, group_id, description="renamed", permissions=permissions)

        assert member_tab.heard_nothing()


def test_removing_a_member_tells_their_open_tab(admin, make_user, groups):
    member, staying = make_user(), make_user()
    group_id = _group(admin, groups, [member, staying], {"workspace": {"knowledge": True}})

    with connected(member) as member_socket, connected(staying) as staying_socket:
        member_tab, staying_tab = Tab(member_socket), Tab(staying_socket)
        with admin.client() as client:
            removed = client.post(
                f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [member.id]}
            )
        assert removed.status_code == 200, removed.text

        assert member_tab.was_told(), "the removed member's open tab kept its old access"
        assert staying_tab.heard_nothing()
