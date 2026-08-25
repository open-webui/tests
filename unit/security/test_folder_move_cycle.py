"""Regression: a folder can never become its own ancestor.

open-webui 0.11.1 fix `23b3a69bc` (#28748): `POST /folders/{id}/update/parent`
accepted a move of a folder into one of its own descendants. A folder whose
parent chain loops has no root, so it and its whole subtree vanished from the
sidebar with no way to get them back from the UI, and every folder-tree walk
(`get_children_folders_by_id_and_user_id`, the subtree id collector, the delete
cascade, `has_folder_access`) then followed the loop forever, hammering the
database until the worker was gone.

The fix rejects the move with a 400, puts folders whose parent chain loops back
at the top level on the next folder listing, and bounds every tree walk with a
visited-id set.

Discriminates: passes on v0.11.1, fails on v0.11.0 (the move into a descendant
is accepted and the loop is written to the database, the listing leaves the
orphaned folders buried, and the access walk never terminates).

Every walk here is bounded by a fake that raises once it has been called far
more often than an acyclic tree could justify, so the pre-fix code fails fast
instead of wedging the run.
"""

import re
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

OWNER = "alice"
OTHER = "mallory"

# Any walk over the fixtures below visits a handful of folders. A walk that
# asks for more than this is following a loop.
WALK_BUDGET = 64


class _Folder(SimpleNamespace):
    """A folder row. `model_dump` is what the listing response is built from."""

    def model_dump(self) -> dict:
        return dict(vars(self))


def _folder(id: str, user_id: str = OWNER, parent_id: str | None = None) -> _Folder:
    return _Folder(
        id=id,
        user_id=user_id,
        parent_id=parent_id,
        name=id,
        items=None,
        meta=None,
        data=None,
        is_expanded=False,
        created_at=0,
        updated_at=0,
    )


def _user(id: str = OWNER, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=id, role=role)


class _FolderStore:
    """In-memory stand-in for the `Folders` model layer.

    The subtree walk mirrors the shipped one but is bounded, so a checkout
    without the fix raises instead of spinning on a parent loop.
    """

    def __init__(self, *folders: SimpleNamespace) -> None:
        self.by_id = {folder.id: folder for folder in folders}
        self.parent_writes: list[tuple[str, str | None]] = []
        self.lookups = 0

    async def get_folder_by_id(self, id, db=None):
        self._charge()
        return self.by_id.get(id)

    async def get_folder_by_id_and_user_id(self, id, user_id, db=None):
        folder = self.by_id.get(id)
        return folder if folder and folder.user_id == user_id else None

    async def get_folder_by_parent_id_and_user_id_and_name(self, parent_id, user_id, name, db=None):
        return None

    async def get_folders_by_user_id(self, user_id, db=None):
        return [entry for entry in self.by_id.values() if entry.user_id == user_id]

    async def get_folders_by_parent_id_and_user_id(self, parent_id, user_id, db=None):
        self._charge()
        return [
            entry
            for entry in self.by_id.values()
            if entry.parent_id == parent_id and entry.user_id == user_id
        ]

    async def get_folder_ids_by_id_and_user_id_in_subtree(self, id, user_id, db=None):
        root = self.by_id.get(id)
        if not root or root.user_id != user_id:
            return []
        found = {id}
        pending = [id]
        while pending:
            self._charge()
            parent_id = pending.pop()
            for entry in self.by_id.values():
                if entry.parent_id == parent_id and entry.user_id == user_id:
                    if entry.id not in found:
                        found.add(entry.id)
                        pending.append(entry.id)
        return list(found)

    async def update_folder_parent_id_by_id_and_user_id(self, id, user_id, parent_id, db=None):
        self.parent_writes.append((id, parent_id))
        folder = self.by_id[id]
        folder.parent_id = parent_id
        return folder

    def _charge(self) -> None:
        self.lookups += 1
        if self.lookups > WALK_BUDGET:
            raise AssertionError(
                f"the folder walk made more than {WALK_BUDGET} lookups over a "
                "handful of folders, so it is following a parent loop that never "
                "terminates (#28748)"
            )


@contextmanager
def _router_boundary_patched(mod, store: _FolderStore):
    """Replace only the I/O boundary the folder handlers touch."""
    patches = [
        patch.object(mod, "Folders", store),
        patch.object(mod, "check_folders_permission", AsyncMock()),
        patch.object(mod, "get_folder_unread_counts", AsyncMock(return_value={})),
        patch.object(mod, "publish_event", AsyncMock()),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        yield


async def _move(mod, store: _FolderStore, folder_id: str, parent_id: str | None):
    with _router_boundary_patched(mod, store):
        return await mod.update_folder_parent_id_by_id(
            SimpleNamespace(app=SimpleNamespace()),
            folder_id,
            mod.FolderParentIdForm(parent_id=parent_id),
            user=_user(),
            db=None,
        )


async def _list_folders(mod, store: _FolderStore):
    with _router_boundary_patched(mod, store):
        return await mod.get_folders(SimpleNamespace(app=SimpleNamespace()), user=_user(), db=None)


def _nested_tree() -> _FolderStore:
    """A three-deep chain plus an unrelated sibling at the top level."""
    return _FolderStore(
        _folder("projects"),
        _folder("2026", parent_id="projects"),
        _folder("q1", parent_id="2026"),
        _folder("archive"),
    )


# ---------------------------------------------------------------------------
# narrow: the reported move is refused, at every depth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["2026", "q1"])
async def test_move_into_own_descendant_is_refused(owui_module, target):
    """The bug: 'projects' accepted a move under its own child or grandchild."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")
    store = _nested_tree()

    with pytest.raises(HTTPException) as excinfo:
        await _move(mod, store, "projects", target)

    assert excinfo.value.status_code == 400, (
        f"moving 'projects' under its own descendant '{target}' was accepted; the "
        "parent chain then loops and the folder plus everything in it disappears "
        "from the sidebar unrecoverably (#28748)"
    )
    assert store.parent_writes == [], (
        f"the loop was written to the database before the check ran: {store.parent_writes} (#28748)"
    )
    assert store.by_id["projects"].parent_id is None, (
        "'projects' was reparented under its own descendant (#28748)"
    )


@pytest.mark.asyncio
async def test_move_into_itself_is_refused(owui_module):
    """The degenerate loop: a folder set as its own parent."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")
    store = _nested_tree()

    with pytest.raises(HTTPException) as excinfo:
        await _move(mod, store, "2026", "2026")

    assert excinfo.value.status_code == 400, (
        "a folder was allowed to become its own parent (#28748)"
    )
    assert store.parent_writes == [], f"self-parent was persisted: {store.parent_writes} (#28748)"


@pytest.mark.asyncio
async def test_refusal_explains_the_folder_cannot_move_into_itself(owui_module):
    """The 400 has to be distinguishable from the name-collision 400 next to it."""
    from fastapi import HTTPException

    mod = owui_module("open_webui.routers.folders")

    with pytest.raises(HTTPException) as excinfo:
        await _move(mod, _nested_tree(), "projects", "q1")

    detail = str(excinfo.value.detail).lower()
    assert "itself" in detail or "subfolder" in detail, (
        f"the refusal reads {excinfo.value.detail!r}, which does not tell the user "
        "the move was into the folder's own subtree (#28748)"
    )


# ---------------------------------------------------------------------------
# narrow: the listing recovers folders already stuck in a loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listing_returns_a_looping_folder_to_the_top_level(owui_module):
    """Data written before the fix must become reachable again."""
    mod = owui_module("open_webui.routers.folders")
    store = _FolderStore(
        _folder("projects", parent_id="2026"),
        _folder("2026", parent_id="projects"),
        _folder("archive"),
    )

    listed = await _list_folders(mod, store)

    by_id = {entry.id: entry for entry in listed}
    assert by_id["projects"].parent_id is None or by_id["2026"].parent_id is None, (
        "both folders of a two-folder parent loop were listed still pointing at "
        "each other, so neither is a root and the whole loop stays invisible in "
        "the sidebar (#28748)"
    )
    assert store.parent_writes, (
        "the loop was left in the database; the user has no way to reach those "
        "folders again from the UI (#28748)"
    )
    assert all(parent_id is None for _, parent_id in store.parent_writes), (
        f"recovery reparented a looping folder somewhere other than the top "
        f"level: {store.parent_writes} (#28748)"
    )


@pytest.mark.asyncio
async def test_listing_recovers_a_longer_loop(owui_module):
    """A three-folder loop is the same defect, one hop further out."""
    mod = owui_module("open_webui.routers.folders")
    store = _FolderStore(
        _folder("a", parent_id="c"),
        _folder("b", parent_id="a"),
        _folder("c", parent_id="b"),
    )

    listed = await _list_folders(mod, store)

    assert any(entry.parent_id is None for entry in listed), (
        "a three-folder parent loop was listed intact, so none of the three "
        "folders is a root and all of them stay hidden (#28748)"
    )


@pytest.mark.asyncio
async def test_listing_recovery_does_not_orphan_the_subtree_below_a_loop(owui_module):
    """Only the looping folders move; their children keep their parent."""
    mod = owui_module("open_webui.routers.folders")
    store = _FolderStore(
        _folder("projects", parent_id="2026"),
        _folder("2026", parent_id="projects"),
        _folder("q1", parent_id="2026"),
    )

    listed = await _list_folders(mod, store)

    by_id = {entry.id: entry for entry in listed}
    assert by_id["q1"].parent_id == "2026", (
        "a healthy child below the loop was flattened to the top level as well; "
        "recovery should only break the loop itself (#28748)"
    )


# ---------------------------------------------------------------------------
# nearby: the legitimate paths still work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_to_an_unrelated_folder_still_works(owui_module):
    """The guard must not block ordinary reorganisation."""
    mod = owui_module("open_webui.routers.folders")
    store = _nested_tree()

    await _move(mod, store, "q1", "archive")

    assert store.parent_writes == [("q1", "archive")], (
        f"a legitimate move into an unrelated folder was blocked or mangled: "
        f"{store.parent_writes} (#28748)"
    )


@pytest.mark.asyncio
async def test_move_up_to_the_top_level_still_works(owui_module):
    """parent_id=None is never a loop and must stay allowed."""
    mod = owui_module("open_webui.routers.folders")
    store = _nested_tree()

    await _move(mod, store, "q1", None)

    assert store.parent_writes == [("q1", None)], (
        f"moving a folder back to the top level was blocked: {store.parent_writes} (#28748)"
    )


@pytest.mark.asyncio
async def test_move_into_an_ancestors_sibling_still_works(owui_module):
    """A folder above the mover in the tree is not in its subtree."""
    mod = owui_module("open_webui.routers.folders")
    store = _nested_tree()

    await _move(mod, store, "q1", "projects")

    assert store.parent_writes == [("q1", "projects")], (
        f"moving a folder under its own grandparent was refused, but that is not a "
        f"loop: {store.parent_writes} (#28748)"
    )


@pytest.mark.asyncio
async def test_listing_a_healthy_tree_rewrites_nothing(owui_module):
    """The cycle check must not disturb folders that are already fine."""
    mod = owui_module("open_webui.routers.folders")
    store = _nested_tree()

    listed = await _list_folders(mod, store)

    assert store.parent_writes == [], (
        f"listing a healthy folder tree rewrote parent ids: {store.parent_writes} (#28748)"
    )
    assert {entry.id: entry.parent_id for entry in listed} == {
        "projects": None,
        "2026": "projects",
        "q1": "2026",
        "archive": None,
    }


@pytest.mark.asyncio
async def test_listing_still_recovers_a_folder_whose_parent_is_gone(owui_module):
    """The pre-existing repair for a dangling parent has to survive the fix."""
    mod = owui_module("open_webui.routers.folders")
    store = _FolderStore(_folder("q1", parent_id="deleted-folder"))

    listed = await _list_folders(mod, store)

    assert listed[0].parent_id is None, (
        "a folder pointing at a parent that no longer exists was left unreachable (#28748)"
    )


# ---------------------------------------------------------------------------
# broad: no folder-tree walk may follow a loop forever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_access_check_terminates_on_a_parent_loop(owui_module):
    """`has_folder_access` walked the ancestor chain with no visited set."""
    mod = owui_module("open_webui.utils.access_control.folders")
    store = _FolderStore(
        _folder("a", user_id=OWNER, parent_id="b"),
        _folder("b", user_id=OWNER, parent_id="a"),
    )
    grants = SimpleNamespace(has_access=AsyncMock(return_value=False))

    with patch.object(mod, "Folders", store), patch.object(mod, "AccessGrants", grants):
        allowed = await mod.has_folder_access(OTHER, store.by_id["a"], "read", None)

    assert allowed is False, "a stranger was granted access to a folder in a parent loop (#28748)"


@pytest.mark.asyncio
async def test_access_check_still_inherits_from_an_ancestor(owui_module):
    """Bounding the walk must not drop inherited access."""
    mod = owui_module("open_webui.utils.access_control.folders")
    store = _FolderStore(
        _folder("projects", user_id=OWNER),
        _folder("2026", user_id=OWNER, parent_id="projects"),
        _folder("q1", user_id=OWNER, parent_id="2026"),
    )
    grants = SimpleNamespace(has_access=AsyncMock(return_value=False))

    with patch.object(mod, "Folders", store), patch.object(mod, "AccessGrants", grants):
        allowed = await mod.has_folder_access(OWNER, store.by_id["q1"], "read", None)

    assert allowed is True, "the owner lost access to their own nested folder (#28748)"


@pytest.mark.asyncio
async def test_access_check_still_honours_a_grant_on_an_ancestor(owui_module):
    """An access grant two levels up still reaches the leaf."""
    mod = owui_module("open_webui.utils.access_control.folders")
    store = _FolderStore(
        _folder("projects", user_id=OWNER),
        _folder("2026", user_id=OWNER, parent_id="projects"),
        _folder("q1", user_id=OWNER, parent_id="2026"),
    )

    async def has_access(user_id, resource_type, resource_id, permission, db=None):
        return resource_id == "projects"

    grants = SimpleNamespace(has_access=has_access)

    with patch.object(mod, "Folders", store), patch.object(mod, "AccessGrants", grants):
        allowed = await mod.has_folder_access(OTHER, store.by_id["q1"], "read", None)

    assert allowed is True, "inherited access from a granted ancestor was lost (#28748)"


TRAVERSALS = [
    "get_children_folders_by_id_and_user_id",
    "get_folder_ids_by_id_and_user_id_in_subtree",
    "delete_folder_by_id_and_user_id",
]


def _method_source(source: str, name: str) -> str:
    """The body of one `async def` on `FolderTable`, up to the next method."""
    start = re.search(rf"^    async def {re.escape(name)}\(", source, re.MULTILINE)
    assert start, f"{name} is not in models/folders.py; the test's premise is stale"
    rest = source[start.end() :]
    end = re.search(r"^    async def ", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


@pytest.mark.parametrize("name", TRAVERSALS)
def test_every_folder_tree_walk_skips_ids_it_has_already_seen(open_webui_backend, name):
    """The class of bug: any walk without a visited set loops on corrupt data.

    A source audit rather than a call, because these three go through
    `get_async_db_context` and a real `select()`; running them on a loop is
    exactly the non-termination this guards against.
    """
    source = (open_webui_backend / "open_webui" / "models" / "folders.py").read_text(
        encoding="utf-8"
    )
    body = _method_source(source, name)

    assert re.search(r"\bif\s+[\w.]+\s+(?:not\s+)?in\s+\w*(?:seen|ids)\w*\b", body), (
        f"FolderTable.{name} walks the folder tree without checking whether it has "
        "already visited an id, so a parent loop makes it recurse or spin forever "
        "and hammer the database until the worker is exhausted (#28748)"
    )
