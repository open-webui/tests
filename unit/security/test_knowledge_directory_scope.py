"""Regression: knowledge directory ids stay inside the knowledge base addressed.

open-webui 0.11.4 fix `a9541c18c` (#29887): the knowledge file routes took a
caller-supplied `directory_id` or `parent_id` and used it without checking it
belongs to the knowledge base in the URL. Anyone with write access to one
knowledge base could reach into another base's directory tree: add a file
into a foreign directory, create a subdirectory under a foreign parent, move
files into a foreign directory. The fix adds `_verify_directory_in_knowledge`
to every route that takes a directory id, scopes the breadcrumb walk with
`knowledge_id`, and makes `process_uploaded_file` drop a foreign
`metadata.directory_id` instead of honouring it.

The tests stub the model layer so a foreign directory fully exists, with a
different `knowledge_id`, and capture the mutating calls at the boundary.

Discriminates: passes on v0.11.4, fails on v0.11.3 (the foreign directory id
is accepted by every route: the file is filed into it, the directory is
created under it, the breadcrumb walk reaches it, and the auto-link files
the upload into it).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy
import sqlalchemy.ext.asyncio

pytest.importorskip("fastapi")

from fastapi import HTTPException

pytestmark = pytest.mark.regression

TARGET_KB = "kb-alice"
FOREIGN_KB = "kb-bob"
ALICE = SimpleNamespace(id="alice", role="user")


def _directory(id: str, knowledge_id: str, parent_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, knowledge_id=knowledge_id, parent_id=parent_id, name=id, user_id="alice"
    )


class Mutations:
    def __init__(self):
        self.added_files = []  # (knowledge_id, file_id, directory_id)
        self.created_directories = []  # (knowledge_id, parent_id)
        self.moved_files = []  # (knowledge_id, file_id, directory_id)
        self.updated_directories = []
        self.deleted_directories = []


class World:
    """Two knowledge bases and one directory each, wired into the router."""

    def __init__(self, router_module, knowledge_model_module):
        self.router = router_module
        self.log = Mutations()
        knowledge_model = knowledge_model_module.KnowledgeModel
        self.knowledge_bases = {
            TARGET_KB: knowledge_model(
                id=TARGET_KB, user_id="alice", name="Alice KB", description="", meta={},
                created_at=0, updated_at=0,
            ),
            FOREIGN_KB: knowledge_model(
                id=FOREIGN_KB, user_id="bob", name="Bob KB", description="", meta={},
                created_at=0, updated_at=0,
            ),
        }
        self.directories = {
            "dir-alice": _directory("dir-alice", TARGET_KB),
            "dir-bob": _directory("dir-bob", FOREIGN_KB),
        }

    async def get_knowledge_by_id(self, id=None, db=None):
        return self.knowledge_bases.get(id)

    async def get_directory_by_id(self, directory_id, db=None):
        return self.directories.get(directory_id)

    async def add_file_to_knowledge_by_id(
        self, knowledge_id, file_id, user_id, directory_id=None, db=None
    ):
        self.log.added_files.append((knowledge_id, file_id, directory_id))
        return SimpleNamespace(
            id="kf-1",
            knowledge_id=knowledge_id,
            file_id=file_id,
            directory_id=directory_id,
            user_id=user_id,
            created_at=0,
            updated_at=0,
        )

    async def create_directory(self, knowledge_id, name, user_id, parent_id=None, db=None):
        self.log.created_directories.append((knowledge_id, parent_id))
        return _directory(f"new-{name}", knowledge_id, parent_id)

    async def update_directory(self, directory_id, name=None, parent_id="__unset__", db=None):
        self.log.updated_directories.append(directory_id)
        return self.directories.get(directory_id)

    async def delete_directory(self, directory_id, move_files_to_parent=True, db=None):
        self.log.deleted_directories.append(directory_id)
        return True

    async def move_file_to_directory(self, knowledge_id, file_id, directory_id=None, db=None):
        self.log.moved_files.append((knowledge_id, file_id, directory_id))
        return True

    async def get_file_by_id(self, id=None, db=None):
        return SimpleNamespace(
            id=id, user_id="alice", data={"content": "x"}, meta={}, hash="h", path=None
        )

    async def get_files_by_ids(self, ids, db=None):
        return [await self.get_file_by_id(id) for id in ids]

    async def has_file(self, knowledge_id, file_id, db=None):
        return True


@pytest.fixture
def world(owui_module):
    router_module = owui_module("open_webui.routers.knowledge")
    knowledge_model_module = owui_module("open_webui.models.knowledge")
    fixture = World(router_module, knowledge_model_module)
    patches = [
        patch.object(router_module.Knowledges, "get_knowledge_by_id", fixture.get_knowledge_by_id),
        patch.object(
            router_module.Knowledges, "get_directory_by_id", fixture.get_directory_by_id
        ),
        patch.object(
            router_module.Knowledges,
            "add_file_to_knowledge_by_id",
            fixture.add_file_to_knowledge_by_id,
        ),
        patch.object(router_module.Knowledges, "create_directory", fixture.create_directory),
        patch.object(router_module.Knowledges, "update_directory", fixture.update_directory),
        patch.object(router_module.Knowledges, "delete_directory", fixture.delete_directory),
        patch.object(
            router_module.Knowledges, "move_file_to_directory", fixture.move_file_to_directory
        ),
        patch.object(router_module.Knowledges, "has_file", fixture.has_file),
        patch.object(
            router_module.Files, "get_file_by_id", fixture.get_file_by_id
        ),
        patch.object(router_module.Files, "get_files_by_ids", fixture.get_files_by_ids),
        patch.object(router_module, "publish_event", AsyncMock()),
        patch.object(
            router_module,
            "process_file",
            AsyncMock(),
        ),
        patch.object(
            router_module,
            "process_files_batch",
            AsyncMock(),
        ),
    ]
    for p in patches:
        p.start()
    try:
        yield fixture
    finally:
        for p in reversed(patches):
            p.stop()


# ── Narrow: the bug itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_file_endpoint_refuses_a_foreign_directory(world):
    form = world.router.KnowledgeFileIdForm(file_id="file-1", directory_id="dir-bob")
    with pytest.raises(HTTPException) as raised:
        await world.router.add_file_to_knowledge_by_id(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.added_files == [], (
        "the add-file endpoint filed a file into another knowledge base's "
        "directory (#29887)"
    )


@pytest.mark.asyncio
async def test_batch_add_endpoint_refuses_a_foreign_directory(world):
    forms = [world.router.KnowledgeFileIdForm(file_id="file-1", directory_id="dir-bob")]
    with pytest.raises(HTTPException) as raised:
        await world.router.add_files_to_knowledge_batch(
            request=None, id=TARGET_KB, form_data=forms, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.added_files == [], (
        "the batch-add endpoint filed files into another knowledge base's "
        "directory (#29887)"
    )


@pytest.mark.asyncio
async def test_create_directory_endpoint_refuses_a_foreign_parent(world):
    form = world.router.KnowledgeDirectoryCreateForm(name="newdir", parent_id="dir-bob")
    with pytest.raises(HTTPException) as raised:
        await world.router.create_knowledge_directory(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.created_directories == [], (
        "a subdirectory was created under another knowledge base's directory (#29887)"
    )


@pytest.mark.asyncio
async def test_move_file_endpoint_refuses_a_foreign_target_directory(world):
    form = world.router.KnowledgeFileMoveForm(file_id="file-1", directory_id="dir-bob")
    with pytest.raises(HTTPException) as raised:
        await world.router.move_file_in_knowledge(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.moved_files == [], (
        "the move endpoint moved a file into another knowledge base's directory (#29887)"
    )


@pytest.mark.asyncio
async def test_update_directory_endpoint_refuses_a_foreign_new_parent(world):
    form = world.router.KnowledgeDirectoryUpdateForm(name="renamed", parent_id="dir-bob")
    with pytest.raises(HTTPException) as raised:
        await world.router.update_knowledge_directory(
            request=None,
            id=TARGET_KB,
            dir_id="dir-alice",
            form_data=form,
            user=ALICE,
            db=None,
        )
    assert raised.value.status_code == 404
    assert world.log.updated_directories == [], (
        "a directory was re-parented under another knowledge base's directory (#29887)"
    )


@pytest.mark.asyncio
async def test_auto_link_on_upload_ignores_a_foreign_directory(owui_module, world):
    """`process_uploaded_file` is in routers.files but was fixed by the same commit:
    a client-supplied metadata.directory_id of another base must be dropped,
    not honoured."""
    files_router = owui_module("open_webui.routers.files")

    class FakeAsyncSession:
        pass

    metadata = {"knowledge_id": TARGET_KB, "directory_id": "dir-bob"}

    async def fake_transcribe(*args, **kwargs):
        return {"text": "x"}

    async def fake_process_file(request, form, user, db=None):
        pass

    file_item = SimpleNamespace(id="file-9", data={}, meta={})
    file = SimpleNamespace(content_type="text/plain", file=None, filename="p.txt")

    patches = [
        patch.object(files_router.Config, "get", AsyncMock(return_value=[])),
        patch.object(files_router, "transcribe", fake_transcribe),
        patch.object(files_router, "process_file", fake_process_file),
        patch.object(
            files_router.Files,
            "update_file_data_by_id",
            AsyncMock(),
        ),
        patch.object(files_router.Storage, "get_file", lambda path: path),
    ]
    for p in patches:
        p.start()
    try:
        await files_router.process_uploaded_file(
            request=None,
            file=file,
            file_path=None,
            file_item=file_item,
            file_metadata=metadata,
            user=ALICE,
            db=FakeAsyncSession(),
        )
    finally:
        for p in reversed(patches):
            p.stop()

    assert world.log.added_files == [(TARGET_KB, "file-9", None)], (
        "an upload's client-supplied directory_id of another knowledge base was "
        "honoured instead of dropped (#29887)"
    )


@pytest.mark.asyncio
async def test_breadcrumb_walk_stays_inside_the_knowledge_base(owui_module, tmp_path):
    """The model-layer half of the fix: the walk up the parent chain is scoped
    by knowledge_id, so a foreign directory stops the walk instead of leaking
    its chain into the caller's breadcrumbs."""
    knowledge_model = owui_module("open_webui.models.knowledge")
    internal_db = owui_module("open_webui.internal.db")

    db_path = tmp_path / "knowledge_dirs.db"
    sync_engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    knowledge_model.KnowledgeDirectory.__table__.create(sync_engine)
    sync_engine.dispose()

    async_engine = sqlalchemy.ext.asyncio.create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_maker = sqlalchemy.ext.asyncio.async_sessionmaker(
        bind=async_engine,
        class_=sqlalchemy.ext.asyncio.AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    now = 0
    sync_engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    rows = [
        knowledge_model.KnowledgeDirectory(
            id="dir-foreign", knowledge_id=FOREIGN_KB, parent_id=None, name="f",
            user_id="bob", created_at=now, updated_at=now,
        ),
        knowledge_model.KnowledgeDirectory(
            id="dir-own", knowledge_id=TARGET_KB, parent_id=None, name="o",
            user_id="alice", created_at=now, updated_at=now,
        ),
    ]
    with sqlalchemy.orm.Session(sync_engine) as session:
        session.add_all(rows)
        session.commit()

    try:
        with patch.object(internal_db, "AsyncSessionLocal", session_maker):
            foreign = await knowledge_model.Knowledges.get_directory_breadcrumbs(
                TARGET_KB, "dir-foreign"
            )
            own = await knowledge_model.Knowledges.get_directory_breadcrumbs(TARGET_KB, "dir-own")
    finally:
        await async_engine.dispose()

    assert [directory.id for directory in foreign] == [], (
        "the breadcrumb walk followed a directory id into another knowledge "
        "base's tree (#29887)"
    )
    assert [directory.id for directory in own] == ["dir-own"], (
        "the walk still builds breadcrumbs for the base's own directories"
    )


# ── Nearby: own directories still work, unknown ids still 404 ───────────────


@pytest.mark.asyncio
async def test_own_directory_is_accepted_for_add(world):
    form = world.router.KnowledgeFileIdForm(file_id="file-1", directory_id="dir-alice")
    await world.router.add_file_to_knowledge_by_id(
        request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
    )
    assert world.log.added_files == [(TARGET_KB, "file-1", "dir-alice")]


@pytest.mark.asyncio
async def test_root_level_add_with_no_directory_is_accepted(world):
    form = world.router.KnowledgeFileIdForm(file_id="file-1", directory_id=None)
    await world.router.add_file_to_knowledge_by_id(
        request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
    )
    assert world.log.added_files == [(TARGET_KB, "file-1", None)]


@pytest.mark.asyncio
async def test_unknown_directory_id_is_refused(world):
    form = world.router.KnowledgeFileIdForm(file_id="file-1", directory_id="dir-ghost")
    with pytest.raises(HTTPException) as raised:
        await world.router.add_file_to_knowledge_by_id(
            request=None, id=TARGET_KB, form_data=form, user=ALICE, db=None
        )
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_own_directory_delete_still_goes_through(world):
    result = await world.router.delete_knowledge_directory(
        request=None, id=TARGET_KB, dir_id="dir-alice", move_files=True, user=ALICE, db=None
    )
    assert result == {"status": True}
    assert world.log.deleted_directories == ["dir-alice"]


@pytest.mark.asyncio
async def test_delete_endpoint_refuses_a_foreign_directory(world):
    with pytest.raises(HTTPException) as raised:
        await world.router.delete_knowledge_directory(
            request=None, id=TARGET_KB, dir_id="dir-bob", move_files=True, user=ALICE, db=None
        )
    assert raised.value.status_code == 404
    assert world.log.deleted_directories == []
