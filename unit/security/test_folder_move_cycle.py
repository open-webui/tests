"""Regression: no folder-tree walk may follow a parent loop forever.

open-webui 0.11.1, fix `23b3a69bc` (#28748): a folder could be moved under its own descendant,
which wrote a parent loop to the database. Every folder-tree walk (`has_folder_access`,
`get_children_folders_by_id_and_user_id`, the subtree id collector and the delete cascade) then
followed the loop with no record of where it had been, hammering the database until the worker
was gone. The fix bounds every walk with a set of visited ids.

The move refusal and the listing repair are pinned over HTTP in the twin,
integration/security/test_folder_move_cycle.py. The walks stay here because reaching one over
HTTP on a checkout without the fix would wedge the shared instance. They run against a private
SQLite database holding the real tables, whose engine refuses to run more statements than any
walk over these few folders needs, so a walk that follows the loop fails fast.

Discriminates: passes on bbfa876af, fails with the visited-id sets of 23b3a69bc removed (every
walk over the loop runs into the statement budget: the access check and the subtree collector
raise, the children walk and the delete cascade swallow it and return nothing).
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest
import pytest_asyncio

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

pytestmark = pytest.mark.regression

OWNER = "alice"
STRANGER = "mallory"

# A walk over three folders needs a few dozen statements at most.
STATEMENT_BUDGET = 200

PARENT_LOOP = {"a": "b", "b": "a"}
CHAIN = {"a": None, "b": "a", "c": "b"}


class StatementBudgetExceeded(RuntimeError):
    pass


@pytest.fixture(scope="module")
def folder_models(owui_module):
    return owui_module("open_webui.models.folders")


@pytest.fixture(scope="module")
def folder_access(owui_module):
    return owui_module("open_webui.utils.access_control.folders")


@pytest_asyncio.fixture
async def seed_folders(folder_models, owui_module, tmp_path):
    """`seed_folders({id: parent_id})` stores those folders for OWNER in a private database."""
    internal_db = owui_module("open_webui.internal.db")
    access_grants = owui_module("open_webui.models.access_grants")
    groups = owui_module("open_webui.models.groups")
    database_url = f"sqlite:///{tmp_path / 'folders.db'}"

    sync_engine = create_engine(database_url)
    tables = [
        folder_models.Folder.__table__,
        access_grants.AccessGrant.__table__,
        groups.Group.__table__,
        groups.GroupMember.__table__,
    ]
    internal_db.Base.metadata.create_all(sync_engine, tables=tables)

    async_engine = create_async_engine(database_url.replace("sqlite", "sqlite+aiosqlite", 1))
    statement_count = itertools.count(1)

    @event.listens_for(async_engine.sync_engine, "before_cursor_execute")
    def enforce_budget(*_):
        if next(statement_count) > STATEMENT_BUDGET:
            raise StatementBudgetExceeded(
                f"more than {STATEMENT_BUDGET} statements over three folders: the walk is "
                "following a parent loop and would never stop (#28748)"
            )

    def seed(parent_by_id: dict[str, str | None]) -> None:
        with Session(sync_engine) as session:
            session.add_all(
                folder_models.Folder(
                    id=folder_id,
                    parent_id=parent_id,
                    user_id=OWNER,
                    name=folder_id,
                    created_at=0,
                    updated_at=0,
                )
                for folder_id, parent_id in parent_by_id.items()
            )
            session.commit()

    session_maker = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
    with patch.object(internal_db, "AsyncSessionLocal", session_maker):
        yield seed
    await async_engine.dispose()
    sync_engine.dispose()


async def _children_walk(folders, root_id: str) -> set[str] | None:
    children = await folders.get_children_folders_by_id_and_user_id(id=root_id, user_id=OWNER)
    return None if children is None else {root_id, *(child.id for child in children)}


async def _subtree_walk(folders, root_id: str) -> set[str]:
    return set(await folders.get_folder_ids_by_id_and_user_id_in_subtree(id=root_id, user_id=OWNER))


async def _delete_walk(folders, root_id: str) -> set[str]:
    return set(await folders.delete_folder_by_id_and_user_id(id=root_id, user_id=OWNER))


WALKS = {"children": _children_walk, "subtree": _subtree_walk, "delete": _delete_walk}


# narrow: the walks stop at an id they have already seen


@pytest.mark.asyncio
@pytest.mark.parametrize("walk", WALKS)
async def test_every_folder_tree_walk_terminates_on_a_parent_loop(
    folder_models, seed_folders, walk
):
    seed_folders(PARENT_LOOP)

    reached = await WALKS[walk](folder_models.Folders, "a")

    assert reached == {"a", "b"}, (
        f"the {walk} walk over a two-folder parent loop returned {reached!r} instead of both "
        "folders, so it followed the loop until it failed (#28748)"
    )


@pytest.mark.asyncio
async def test_access_check_terminates_on_a_parent_loop(folder_models, folder_access, seed_folders):
    seed_folders(PARENT_LOOP)
    folder = await folder_models.Folders.get_folder_by_id(id="a")

    allowed = await folder_access.has_folder_access(
        user_id=STRANGER, folder=folder, permission="read", db=None
    )

    assert allowed is False, "a stranger was granted access to a folder in a parent loop (#28748)"


# nearby: an acyclic tree is still walked in full


@pytest.mark.asyncio
@pytest.mark.parametrize("walk", WALKS)
async def test_every_folder_tree_walk_still_reaches_the_whole_subtree(
    folder_models, seed_folders, walk
):
    seed_folders(CHAIN)

    reached = await WALKS[walk](folder_models.Folders, "a")

    assert reached == {"a", "b", "c"}, f"the {walk} walk lost part of an acyclic tree: {reached!r}"
