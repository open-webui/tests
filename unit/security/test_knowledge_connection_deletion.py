"""Regression: deleting one external knowledge base must not tear down the
external connection that other knowledge bases are still using.

open-webui 0.11.1 fix `dc03e7e59` (#28113): `DELETE /knowledge/{id}/delete`
read the `connection_id` out of an external knowledge base's meta and
unconditionally rewrote the admin-owned connection list with that connection
removed. External connections are shared: several knowledge bases point at the
same one. So any user with write access to a single external knowledge base
deleted the connection out from under everybody else, and every other knowledge
base on it stopped resolving. The fix only clears the connection when an admin
deletes the LAST knowledge base referencing it, matching the guard the explicit
connection-delete route already had.

The tests assert on the teardown call itself (`_set_external_connections`, the
one write that persists the connection list) rather than on the endpoint's
return value, because the bug never failed the request: the delete succeeded,
it just took the shared connection with it.

Discriminates: passes on v0.11.1, fails on v0.11.0 (deleting one of two shared
knowledge bases removes the connection, and a non-admin deleting the last one
removes it too).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

SHARED_CONNECTION = "conn-shared"
OTHER_CONNECTION = "conn-other"

ALICE = SimpleNamespace(id="alice", role="user")
BOB = SimpleNamespace(id="bob", role="user")
ADMIN = SimpleNamespace(id="root", role="admin")
STRANGER = SimpleNamespace(id="mallory", role="user")


def _external_kb(id: str, user_id: str, connection_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        user_id=user_id,
        name=id,
        description="",
        meta={"source": "external", "external": {"connection_id": connection_id}},
    )


def _local_kb(id: str, user_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=id, user_id=user_id, name=id, description="", meta={})


class World:
    """Two connections and the knowledge bases hanging off them, wired into the
    router's model and config layer so the real endpoint code runs unmodified.

    `kb-alice` and `kb-bob` share `conn-shared`; `kb-solo` is the only user of
    `conn-other`.
    """

    def __init__(self, router_module):
        self.router = router_module
        self.knowledge_bases = {
            "kb-alice": _external_kb("kb-alice", "alice", SHARED_CONNECTION),
            "kb-bob": _external_kb("kb-bob", "bob", SHARED_CONNECTION),
            "kb-solo": _external_kb("kb-solo", "alice", OTHER_CONNECTION),
            "kb-local": _local_kb("kb-local", "alice"),
        }
        self.connections = [
            {"id": SHARED_CONNECTION, "name": "shared", "provider": "external"},
            {"id": OTHER_CONNECTION, "name": "other", "provider": "external"},
        ]
        # every persisted connection-list write, in order
        self.connection_writes = []
        self.deleted_knowledge_bases = []
        self.dropped_collections = []
        self.vector_deletes = []

    @property
    def connection_ids(self) -> list[str]:
        return [connection.get("id") for connection in self.connections]

    async def get_knowledge_by_id(self, id=None, db=None):
        return self.knowledge_bases.get(id)

    async def get_knowledge_bases(self, skip=0, limit=30, db=None):
        return list(self.knowledge_bases.values())

    async def delete_knowledge_by_id(self, id=None, db=None):
        self.deleted_knowledge_bases.append(id)
        self.knowledge_bases.pop(id, None)
        return True

    async def get_external_connections(self):
        return [dict(connection) for connection in self.connections]

    async def set_external_connections(self, connections):
        self.connection_writes.append([connection.get("id") for connection in connections])
        self.connections = list(connections)

    async def vector_delete(self, collection_name=None, **kwargs):
        self.vector_deletes.append(collection_name)

    async def vector_delete_collection(self, collection_name=None, **kwargs):
        self.dropped_collections.append(collection_name)


@pytest.fixture
def world(owui_module):
    router_module = owui_module("open_webui.routers.knowledge")
    fixture = World(router_module)
    vector_client = SimpleNamespace(
        delete=fixture.vector_delete,
        delete_collection=fixture.vector_delete_collection,
    )
    patches = [
        patch.object(router_module.Knowledges, "get_knowledge_by_id", fixture.get_knowledge_by_id),
        patch.object(router_module.Knowledges, "get_knowledge_bases", fixture.get_knowledge_bases),
        patch.object(
            router_module.Knowledges, "delete_knowledge_by_id", fixture.delete_knowledge_by_id
        ),
        patch.object(router_module.Models, "get_all_models", AsyncMock(return_value=[])),
        patch.object(router_module.AccessGrants, "has_access", AsyncMock(return_value=False)),
        patch.object(router_module, "_get_external_connections", fixture.get_external_connections),
        patch.object(router_module, "_set_external_connections", fixture.set_external_connections),
        patch.object(router_module, "ASYNC_VECTOR_DB_CLIENT", vector_client),
        patch.object(router_module, "publish_event", AsyncMock()),
    ]
    for p in patches:
        p.start()
    try:
        yield fixture
    finally:
        for p in reversed(patches):
            p.stop()


async def _delete_kb(world, id, user):
    return await world.router.delete_knowledge_by_id(request=None, id=id, user=user, db=None)


# ── Narrow: the bug itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deleting_one_of_two_leaves_the_shared_connection(world):
    """Alice deletes her knowledge base; Bob's still uses the same connection."""
    assert await _delete_kb(world, "kb-alice", ALICE) is True
    assert world.connection_writes == [], (
        "deleting one of two knowledge bases on a shared external connection "
        f"rewrote the connection list ({world.connection_writes}): the connection is "
        "torn down while Bob's knowledge base still points at it (#28113)"
    )
    assert SHARED_CONNECTION in world.connection_ids, (
        f"shared connection is gone after deleting one of its two knowledge bases, "
        f"connections left: {world.connection_ids} (#28113)"
    )


@pytest.mark.asyncio
async def test_non_admin_deleting_the_last_one_keeps_the_connection(world):
    """Connections are admin-owned: a plain user never removes one."""
    await world.delete_knowledge_by_id(id="kb-bob")
    world.deleted_knowledge_bases.clear()

    assert await _delete_kb(world, "kb-alice", ALICE) is True
    assert world.connection_writes == [], (
        "a non-admin deleting the last knowledge base on a connection rewrote the "
        f"admin-owned connection list ({world.connection_writes}) (#28113)"
    )
    assert SHARED_CONNECTION in world.connection_ids, (
        f"a non-admin removed an admin-owned external connection, connections left: "
        f"{world.connection_ids} (#28113)"
    )


@pytest.mark.asyncio
async def test_write_access_grant_does_not_let_a_stranger_drop_the_connection(world):
    """The path the report describes: a delegate with write access, not the owner."""
    with patch.object(world.router.AccessGrants, "has_access", AsyncMock(return_value=True)):
        assert await _delete_kb(world, "kb-alice", STRANGER) is True
    assert world.connection_writes == [], (
        "a user holding only a write grant on one knowledge base tore down the "
        f"shared connection ({world.connection_writes}) (#28113)"
    )
    assert SHARED_CONNECTION in world.connection_ids


# ── Broad: the surrounding external-connection lifecycle ────────────────────


@pytest.mark.asyncio
async def test_admin_deleting_the_last_one_clears_the_connection(world):
    """Once no knowledge base is left on it, an admin's delete does clean up.

    Passes on both refs: the fix only narrows when this write happens, so this is a
    no-regression check that the cleanup was not removed outright.
    """
    await world.delete_knowledge_by_id(id="kb-bob")
    world.deleted_knowledge_bases.clear()

    assert await _delete_kb(world, "kb-alice", ADMIN) is True
    assert world.connection_writes == [[OTHER_CONNECTION]], (
        f"admin deleting the last knowledge base on a connection should persist the "
        f"connection list without it, got writes {world.connection_writes} (#28113)"
    )
    assert world.connection_ids == [OTHER_CONNECTION], (
        f"the unrelated connection must survive, connections left: {world.connection_ids} (#28113)"
    )


@pytest.mark.asyncio
async def test_deleting_the_solo_knowledge_base_only_touches_its_own_connection(world):
    """Cleanup is scoped: the shared connection is untouched by an unrelated delete."""
    await _delete_kb(world, "kb-solo", ADMIN)
    assert world.connection_writes == [[SHARED_CONNECTION]]
    assert world.connection_ids == [SHARED_CONNECTION]


@pytest.mark.asyncio
async def test_connection_delete_route_refuses_while_knowledge_bases_reference_it(world):
    with pytest.raises(HTTPException) as raised:
        await world.router.delete_external_knowledge_connection(
            request=None, id=SHARED_CONNECTION, user=ADMIN, db=None
        )
    assert raised.value.status_code == 400
    assert world.connection_writes == [], (
        "the connection-delete route removed a connection two knowledge bases still use"
    )


@pytest.mark.asyncio
async def test_connection_delete_route_clears_an_unused_connection(world):
    await world.delete_knowledge_by_id(id="kb-solo")
    assert (
        await world.router.delete_external_knowledge_connection(
            request=None, id=OTHER_CONNECTION, user=ADMIN, db=None
        )
        is True
    )
    assert world.connection_ids == [SHARED_CONNECTION]


@pytest.mark.asyncio
async def test_unknown_connection_delete_is_a_404_without_a_write(world):
    with pytest.raises(HTTPException) as raised:
        await world.router.delete_external_knowledge_connection(
            request=None, id="conn-that-never-existed", user=ADMIN, db=None
        )
    assert raised.value.status_code == 404
    assert world.connection_writes == []


# ── Nearby: the delete endpoint still does its job ──────────────────────────


@pytest.mark.asyncio
async def test_local_knowledge_base_delete_drops_its_vector_collection(world):
    assert await _delete_kb(world, "kb-local", ALICE) is True
    assert world.dropped_collections == ["kb-local"]
    assert world.connection_writes == [], (
        "deleting a plain, non-external knowledge base touched the external connection list"
    )
    assert world.deleted_knowledge_bases == ["kb-local"]


@pytest.mark.asyncio
async def test_external_knowledge_base_delete_does_not_drop_a_vector_collection(world):
    await _delete_kb(world, "kb-alice", ADMIN)
    assert world.dropped_collections == [], (
        "external knowledge bases have no local vector collection to drop"
    )
    assert world.deleted_knowledge_bases == ["kb-alice"]


@pytest.mark.asyncio
async def test_external_knowledge_base_without_a_connection_id_is_deleted_cleanly(world):
    world.knowledge_bases["kb-orphan"] = SimpleNamespace(
        id="kb-orphan",
        user_id="alice",
        name="kb-orphan",
        description="",
        meta={"source": "external"},
    )
    assert await _delete_kb(world, "kb-orphan", ADMIN) is True
    assert world.connection_writes == []
    assert world.connection_ids == [SHARED_CONNECTION, OTHER_CONNECTION]


@pytest.mark.asyncio
async def test_unknown_knowledge_base_is_refused_before_any_cleanup(world):
    with pytest.raises(HTTPException) as raised:
        await _delete_kb(world, "kb-that-never-existed", ADMIN)
    assert raised.value.status_code == 400
    assert world.connection_writes == []
    assert world.deleted_knowledge_bases == []


@pytest.mark.asyncio
async def test_user_without_access_cannot_delete_or_touch_the_connection(world):
    with pytest.raises(HTTPException) as raised:
        await _delete_kb(world, "kb-alice", STRANGER)
    assert raised.value.status_code == 400
    assert world.connection_writes == [], (
        "a user with no access to the knowledge base still rewrote the connection list"
    )
    assert world.deleted_knowledge_bases == []


@pytest.mark.asyncio
async def test_knowledge_base_row_is_removed_even_when_the_connection_stays(world):
    """The delete must not become a no-op in the name of protecting the connection."""
    await _delete_kb(world, "kb-alice", ALICE)
    assert world.deleted_knowledge_bases == ["kb-alice"]
    assert "kb-alice" not in world.knowledge_bases
    assert "kb-bob" in world.knowledge_bases
