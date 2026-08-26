"""User listing and channel membership regressions fixed in Open WebUI 0.11.1.

Four upstream fixes, all exercised against a real (scratch) database:

* DM lookup, commit a41faa3c22 (issue #28257). ``ChannelTable.get_dm_channel_by_user_ids``
  counted rows in ``channel_member`` only, so the leftover membership row of a deleted
  account still counted towards the exact-set match and the existing DM was never found:
  messaging that person opened a second conversation next to the first one. The fix joins
  ``User`` and counts surviving accounts.
* SCIM listing, commit fb4f476316. SCIM read paths went through the generic user queries, so
  a directory sync listed (and could modify) accounts created with a password inside Open
  WebUI. The fix routes them through ``get_scim_users``/``get_scim_user_by_id``, which keep
  only rows with an ``oauth`` or ``scim`` payload.
* User-list ordering, same commit. ``routers/users.py`` stuffed ``direction`` into ``filter``
  unconditionally, which made ``filter`` truthy on an empty search and pushed the default
  ``created_at desc`` ordering into a branch that could never run: the admin user list came
  back in raw storage order. The fix moves ordering into its own ``sort`` argument.
* Channel member list, PR #28289 / commit e3e82b1471 (issue #28288). Read grants for users
  and for groups were passed to ``Users.get_users`` as two filters, which AND together, so a
  channel shared with one person plus one group they are not in listed nobody while the count
  next to it said two, and the channel owner never showed up at all. The fix adds
  ``get_channel_member_user_ids``, which expands groups and unions in the owner.

Discriminates: passes on v0.11.1, fails on v0.11.0 (pre-fix the deleted-account DM is not
found, SCIM lists local password accounts, the unsorted user list is not newest-first, and the
channel member list drops the owner and intersects user with group grants).
"""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

# Far in the future so these rows sort ahead of anything else in the scratch database.
# The wall clock is added so each run sits above the rows its previous runs left behind:
# the scratch database persists, and a fixed constant accumulates ties on the same second.
FUTURE_EPOCH = 4_000_000_000


def _newest_epochs(count: int) -> list[int]:
    base = FUTURE_EPOCH + int(time.time())
    return [base + offset for offset in range(1, count + 1)]


@pytest.fixture(scope="module")
def owui(owui_module):
    """Backend modules plus a migrated scratch database."""
    owui_module("open_webui.config")  # runs alembic upgrade head against DATA_DIR
    users = owui_module("open_webui.models.users")
    channels = owui_module("open_webui.models.channels")
    groups = owui_module("open_webui.models.groups")
    grants = owui_module("open_webui.models.access_grants")
    return SimpleNamespace(
        db=owui_module("open_webui.internal.db"),
        users=users,
        channels=channels,
        groups=groups,
        grants=grants,
        Users=users.Users,
        Channels=channels.Channels,
        channels_router=owui_module("open_webui.routers.channels"),
        users_router=owui_module("open_webui.routers.users"),
        scim_router=owui_module("open_webui.routers.scim"),
    )


async def _add(owui, *rows):
    async with owui.db.get_async_db() as session:
        session.add_all(rows)
        await session.commit()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _user(owui, name: str, *, role: str = "user", oauth=None, scim=None, created_at=None):
    user_id = _new_id("user")
    return owui.users.User(
        id=user_id,
        email=f"{user_id}@example.test",
        name=name,
        role=role,
        profile_image_url="/user.png",
        oauth=oauth,
        scim=scim,
        created_at=created_at if created_at is not None else int(time.time()),
        updated_at=int(time.time()),
        last_active_at=int(time.time()),
    )


def _channel(owui, owner_id: str, *, type_: str | None):
    now = int(time.time_ns())
    return owui.channels.Channel(
        id=_new_id("channel"),
        user_id=owner_id,
        type=type_,
        name=_new_id("name"),
        created_at=now,
        updated_at=now,
    )


def _membership(owui, channel_id: str, user_id: str):
    return owui.channels.ChannelMember(
        id=_new_id("member"),
        channel_id=channel_id,
        user_id=user_id,
        role="member",
        is_active=True,
        joined_at=int(time.time_ns()),
    )


def _read_grant(owui, channel_id: str, principal_type: str, principal_id: str):
    return owui.grants.AccessGrant(
        id=_new_id("grant"),
        resource_type="channel",
        resource_id=channel_id,
        principal_type=principal_type,
        principal_id=principal_id,
        permission="read",
        created_at=int(time.time()),
    )


def _group(owui, owner_id: str):
    now = int(time.time())
    return owui.groups.Group(
        id=_new_id("group"),
        user_id=owner_id,
        name=_new_id("groupname"),
        description="",
        created_at=now,
        updated_at=now,
    )


def _group_membership(owui, group_id: str, user_id: str):
    return owui.groups.GroupMember(
        id=_new_id("groupmember"),
        group_id=group_id,
        user_id=user_id,
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )


async def _dm_channel_with_members(owui, member_ids, owner_id):
    channel = _channel(owui, owner_id, type_="dm")
    await _add(owui, channel, *[_membership(owui, channel.id, uid) for uid in member_ids])
    return channel


async def _list_channel_members(owui, channel_id, monkeypatch, query=None):
    """Drive the real members endpoint with the channels feature flag stubbed on.

    `query` keeps the assertions independent of unrelated accounts in the scratch database.
    """
    monkeypatch.setattr(
        owui.channels_router,
        "Config",
        SimpleNamespace(get=_always_enabled),
        raising=True,
    )
    caller = SimpleNamespace(id=_new_id("caller"), role="admin")
    return await owui.channels_router.get_channel_members_by_id(
        request=None, id=channel_id, query=query, user=caller, db=None
    )


async def _always_enabled(key, *args, **kwargs):
    return True if key == "channels.enable" else {}


############################
# 46 - DM lookup after an account is deleted
############################


@pytest.mark.asyncio(loop_scope="module")
async def test_dm_lookup_ignores_membership_row_of_deleted_account(owui):
    """Narrow: the stale membership row of a deleted user must not hide the existing DM."""
    alice, bob = _user(owui, "Alice"), _user(owui, "Bob")
    await _add(owui, alice, bob)
    deleted_user_id = _new_id("deleted")
    channel = await _dm_channel_with_members(owui, [alice.id, bob.id, deleted_user_id], alice.id)

    found = await owui.Channels.get_dm_channel_by_user_ids([alice.id, bob.id])

    assert found is not None, "existing DM was not found, a duplicate conversation would be opened"
    assert found.id == channel.id


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("surviving_count", [1, 2, 3])
async def test_dm_lookup_counts_only_surviving_accounts(owui, surviving_count):
    """Broad: membership rows without a user row never contribute to the exact-set match."""
    survivors = [_user(owui, f"Survivor {i}") for i in range(surviving_count)]
    await _add(owui, *survivors)
    survivor_ids = [u.id for u in survivors]
    ghost_ids = [_new_id("ghost") for _ in range(2)]
    channel = await _dm_channel_with_members(owui, survivor_ids + ghost_ids, survivor_ids[0])

    found = await owui.Channels.get_dm_channel_by_user_ids(survivor_ids)

    assert found is not None
    assert found.id == channel.id


@pytest.mark.asyncio(loop_scope="module")
async def test_dm_lookup_matches_intact_conversation(owui):
    """Nearby: the ordinary two-live-members case still matches."""
    alice, bob = _user(owui, "Alice"), _user(owui, "Bob")
    await _add(owui, alice, bob)
    channel = await _dm_channel_with_members(owui, [alice.id, bob.id], alice.id)

    found = await owui.Channels.get_dm_channel_by_user_ids([bob.id, alice.id])

    assert found is not None
    assert found.id == channel.id


@pytest.mark.asyncio(loop_scope="module")
async def test_dm_lookup_rejects_channel_with_extra_live_member(owui):
    """Nearby: a three-way channel is not the two-way DM."""
    alice, bob, carol = _user(owui, "Alice"), _user(owui, "Bob"), _user(owui, "Carol")
    await _add(owui, alice, bob, carol)
    await _dm_channel_with_members(owui, [alice.id, bob.id, carol.id], alice.id)

    assert await owui.Channels.get_dm_channel_by_user_ids([alice.id, bob.id]) is None


############################
# 156 - SCIM listing local accounts
############################


@pytest.mark.asyncio(loop_scope="module")
async def test_scim_user_lookup_skips_local_password_account(owui):
    """Narrow: a password account created inside Open WebUI is invisible to SCIM."""
    local = _user(owui, "Local Only")
    await _add(owui, local)

    response = await owui.scim_router.get_users(
        request=SimpleNamespace(base_url="http://test/"),
        startIndex=1,
        count=20,
        filter=f'userName eq "{local.email}"',
        _=True,
        db=None,
    )

    assert response.totalResults == 0
    assert response.Resources == []


@pytest.mark.asyncio(loop_scope="module")
async def test_scim_get_user_by_id_skips_local_password_account(owui):
    """Narrow: SCIM cannot read (and so cannot update or delete) a local password account."""
    local = _user(owui, "Local Only")
    await _add(owui, local)

    response = await owui.scim_router.get_user(
        user_id=local.id,
        request=SimpleNamespace(base_url="http://test/"),
        _=True,
        db=None,
    )

    assert getattr(response, "status_code", None) == 404


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "oauth,scim",
    [
        ({"oidc": {"sub": "abc"}}, None),
        (None, {"oidc": {"external_id": "abc"}}),
        ({"oidc": {"sub": "abc"}}, {"oidc": {"external_id": "abc"}}),
    ],
)
async def test_scim_user_lookup_returns_provisioned_account(owui, oauth, scim):
    """Nearby: accounts carrying an oauth or scim payload stay visible to the directory."""
    provisioned = _user(owui, "Directory User", oauth=oauth, scim=scim)
    await _add(owui, provisioned)

    response = await owui.scim_router.get_users(
        request=SimpleNamespace(base_url="http://test/"),
        startIndex=1,
        count=20,
        filter=f'userName eq "{provisioned.email}"',
        _=True,
        db=None,
    )

    assert response.totalResults == 1
    assert [resource.id for resource in response.Resources] == [provisioned.id]


############################
# 157 - Sorting the user list
############################


@pytest.mark.asyncio(loop_scope="module")
async def test_user_list_without_search_is_newest_first(owui):
    """Narrow: an unsearched, unsorted admin listing falls back to created_at desc."""
    low, mid, high = _newest_epochs(3)
    oldest = _user(owui, "Oldest", created_at=low)
    newest = _user(owui, "Newest", created_at=high)
    middle = _user(owui, "Middle", created_at=mid)
    await _add(owui, oldest, newest, middle)  # insertion order differs from created_at order

    # Unsearched and unsorted, which is the fallback under test. These three are the
    # newest rows in the database, so they lead the first page.
    result = await owui.users_router.get_users(user=None, db=None)
    listed_ids = [user.id for user in result["users"]][:3]

    assert listed_ids == [newest.id, middle.id, oldest.id]


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("direction,expected", [("asc", "asc"), ("desc", "desc")])
async def test_user_list_sorts_by_name_in_both_directions(owui, direction, expected):
    """Nearby: an explicit name sort still applies with no search term."""
    marker = uuid.uuid4().hex[:8]
    first = _user(owui, f"aaa-{marker}")
    last = _user(owui, f"zzz-{marker}")
    await _add(owui, last, first)

    result = await owui.users_router.get_users(
        query=marker, order_by="name", direction=direction, user=None, db=None
    )
    listed_ids = [user.id for user in result["users"]]

    assert listed_ids == ([first.id, last.id] if expected == "asc" else [last.id, first.id])


############################
# 184 - Who a channel says its members are
############################


@pytest.mark.asyncio(loop_scope="module")
async def test_channel_members_union_user_and_group_grants(owui, monkeypatch):
    """Narrow: a channel shared with a person and with a group lists both, plus the owner."""
    marker = uuid.uuid4().hex[:8]
    owner = _user(owui, f"Owner {marker}")
    invited = _user(owui, f"Invited {marker}")
    group_member = _user(owui, f"Group Member {marker}")
    await _add(owui, owner, invited, group_member)
    group = _group(owui, owner.id)
    await _add(owui, group)
    await _add(owui, _group_membership(owui, group.id, group_member.id))
    channel = _channel(owui, owner.id, type_=None)
    await _add(
        owui,
        channel,
        _read_grant(owui, channel.id, "user", invited.id),
        _read_grant(owui, channel.id, "group", group.id),
    )

    result = await _list_channel_members(owui, channel.id, monkeypatch, query=marker)

    assert {user.id for user in result["users"]} == {owner.id, invited.id, group_member.id}
    assert result["total"] == 3


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("grant_kinds", [("user",), ("group",), ("user", "group")])
async def test_channel_member_list_always_contains_owner(owui, monkeypatch, grant_kinds):
    """Broad: whatever the grant shape, the owner is a member of their own channel."""
    marker = uuid.uuid4().hex[:8]
    owner = _user(owui, f"Owner {marker}")
    invited = _user(owui, f"Invited {marker}")
    group_member = _user(owui, f"Group Member {marker}")
    await _add(owui, owner, invited, group_member)
    group = _group(owui, owner.id)
    await _add(owui, group)
    await _add(owui, _group_membership(owui, group.id, group_member.id))
    channel = _channel(owui, owner.id, type_=None)
    grants = [
        _read_grant(owui, channel.id, "user", invited.id)
        if kind == "user"
        else _read_grant(owui, channel.id, "group", group.id)
        for kind in grant_kinds
    ]
    await _add(owui, channel, *grants)

    result = await _list_channel_members(owui, channel.id, monkeypatch, query=marker)
    listed_ids = {user.id for user in result["users"]}

    assert owner.id in listed_ids
    assert result["total"] == len(listed_ids)
    if "user" in grant_kinds:
        assert invited.id in listed_ids
    if "group" in grant_kinds:
        assert group_member.id in listed_ids


@pytest.mark.asyncio(loop_scope="module")
async def test_publicly_readable_channel_lists_everyone(owui, monkeypatch):
    """Nearby: a public read grant means no user filter at all."""
    marker = uuid.uuid4().hex[:8]
    owner = _user(owui, f"Owner {marker}")
    bystander = _user(owui, f"Bystander {marker}")
    await _add(owui, owner, bystander)
    channel = _channel(owui, owner.id, type_=None)
    await _add(owui, channel, _read_grant(owui, channel.id, "user", "*"))

    result = await _list_channel_members(owui, channel.id, monkeypatch, query=marker)
    listed_ids = {user.id for user in result["users"]}

    assert listed_ids == {owner.id, bystander.id}


@pytest.mark.asyncio(loop_scope="module")
async def test_pending_accounts_are_not_channel_members(owui, monkeypatch):
    """Nearby: a granted but still pending account is left out of the member list."""
    marker = uuid.uuid4().hex[:8]
    owner = _user(owui, f"Owner {marker}")
    pending = _user(owui, f"Pending {marker}", role="pending")
    await _add(owui, owner, pending)
    channel = _channel(owui, owner.id, type_=None)
    await _add(
        owui,
        channel,
        _read_grant(owui, channel.id, "user", pending.id),
    )

    result = await _list_channel_members(owui, channel.id, monkeypatch, query=marker)

    assert pending.id not in {user.id for user in result["users"]}
