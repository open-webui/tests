"""Regression: a folder's contents must stay bounded by its OWNER's access,
never by the access of whoever happens to be acting on the folder.

open-webui 0.11.0 fix `56183fcb1` (PR #27464). Two halves:

1. `routers/folders.py` validated attached files against the *caller*, so a
   collaborator with write access on someone else's shared folder could park
   their own private files there. Everyone who can read the folder then reads
   those files through it. The fix resolves the folder owner and gates the
   attach on `can_read_all_folder_files(entries, owner)`; `create_folder` gained
   the same gate, which it had entirely lacked.
2. `retrieval/utils.get_sources_from_items` expanded a folder item straight into
   `folder.data['files']`, and `folder_items` then waives the per-file access
   check for every entry it expanded. Files the owner had since lost access to
   kept being served as chat knowledge. The fix routes the expansion through
   `get_owner_accessible_folder_files(folder)`.

Only the DB boundary is mocked: in this world a file is readable exactly by the
user who owns it, so the real `has_access_to_file` decides every case.

Discriminates: passes on v0.11.1, fails whenever the attach is checked against the
caller and folder items are never re-checked against the owner (v0.10.2 shape).
Stubs track the 0.11.1 model/knowledge accessors.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression


def _user(id: str, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=id, role=role)


def _file(id: str, user_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        user_id=user_id,
        filename=f"{id}.txt",
        meta={},
        data={"content": f"contents of {id}"},
    )


def _folder(id: str, user_id: str, entries: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        parent_id=None,
        user_id=user_id,
        name="Project",
        items=None,
        meta=None,
        data={"files": list(entries)},
        is_expanded=False,
        created_at=0,
        updated_at=0,
    )


ALICE = _user("alice")
BOB = _user("bob")
ADMIN = _user("root", role="admin")
USERS = {user.id: user for user in (ALICE, BOB, ADMIN)}


@contextmanager
def _backend_world(files: dict[str, SimpleNamespace], readable_collections=()):
    """DB boundary: a file is readable only by its owner, a collection only by
    the (collection_id, user_id) pairs listed."""
    import open_webui.models.access_grants as access_grants_module
    import open_webui.models.channels as channels_module
    import open_webui.models.chats as chats_module
    import open_webui.models.files as files_module
    import open_webui.models.groups as groups_module
    import open_webui.models.knowledge as knowledge_module
    import open_webui.models.models as models_module
    import open_webui.models.users as users_module

    grants = set(readable_collections)
    stubs = [
        (
            files_module.Files,
            "get_file_by_id",
            AsyncMock(side_effect=lambda id, db=None: files.get(id)),
        ),
        (
            users_module.Users,
            "get_user_by_id",
            AsyncMock(side_effect=lambda id, db=None: USERS.get(id)),
        ),
        (groups_module.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        (knowledge_module.Knowledges, "get_knowledges_by_file_id", AsyncMock(return_value=[])),
        (knowledge_module.Knowledges, "get_knowledge_by_id", AsyncMock(return_value=None)),
        (
            knowledge_module.Knowledges,
            "check_access_by_user_id",
            AsyncMock(
                side_effect=lambda id, user_id, permission, db=None, user_group_ids=None: (
                    (id, user_id) in grants
                )
            ),
        ),
        (
            channels_module.Channels,
            "get_channels_by_file_id_and_user_id",
            AsyncMock(return_value=[]),
        ),
        (chats_module.Chats, "get_shared_chat_ids_by_file_id", AsyncMock(return_value=[])),
        (models_module.Models, "get_model_owners_attaching_file", AsyncMock(return_value={})),
        (access_grants_module.AccessGrants, "has_access", AsyncMock(return_value=False)),
        (
            access_grants_module.AccessGrants,
            "get_accessible_resource_ids",
            AsyncMock(return_value=[]),
        ),
    ]
    with ExitStack() as stack:
        for target, attribute, stub in stubs:
            stack.enter_context(patch.object(target, attribute, stub))
        yield


@contextmanager
def _folders_router(folders_module, folder: SimpleNamespace):
    """Patch the folders router's own collaborators and hand back the two write
    mocks, so a test can assert nothing was persisted."""
    import open_webui.models.folders as folders_model_module

    saved = AsyncMock(return_value=folder)
    created = AsyncMock(return_value=folder)
    with ExitStack() as stack:
        stack.enter_context(patch.object(folders_module, "check_folders_permission", AsyncMock()))
        stack.enter_context(patch.object(folders_module, "publish_event", AsyncMock()))
        stack.enter_context(
            patch.object(folders_module, "_has_folder_access", AsyncMock(return_value=True))
        )
        stack.enter_context(
            patch.object(
                folders_model_module.Folders,
                "get_folder_by_id_and_user_id",
                AsyncMock(
                    side_effect=lambda id, user_id, db=None: (
                        folder if folder.user_id == user_id else None
                    )
                ),
            )
        )
        stack.enter_context(
            patch.object(
                folders_model_module.Folders, "get_folder_by_id", AsyncMock(return_value=folder)
            )
        )
        stack.enter_context(
            patch.object(
                folders_model_module.Folders,
                "get_folder_by_parent_id_and_user_id_and_name",
                AsyncMock(return_value=None),
            )
        )
        stack.enter_context(
            patch.object(folders_model_module.Folders, "update_folder_by_id_and_user_id", saved)
        )
        stack.enter_context(
            patch.object(folders_model_module.Folders, "insert_new_folder", created)
        )
        yield saved, created


async def _retrieved_ids(retrieval_module, items, user) -> set[str]:
    """Run the real retrieval item walk and report which file ids produced a
    source. `bypass_embedding_and_retrieval` keeps it off the vector DB."""
    sources = await retrieval_module.get_sources_from_items(
        request=SimpleNamespace(),
        items=items,
        queries=["question"],
        embedding_function=None,
        k=3,
        reranking_function=None,
        k_reranker=3,
        r=0.0,
        hybrid_bm25_weight=0.5,
        hybrid_search=False,
        full_context=False,
        user=user,
    )
    return {source["source"].get("id") for source in sources}


@contextmanager
def _retrieval_world(retrieval_module, folder: SimpleNamespace, files: dict[str, SimpleNamespace]):
    import open_webui.models.config as config_module
    import open_webui.models.folders as folders_model_module

    with ExitStack() as stack:
        stack.enter_context(_backend_world(files))
        stack.enter_context(
            patch.object(
                config_module.Config,
                "get",
                AsyncMock(
                    side_effect=lambda key, default=None: (
                        key == "rag.bypass_embedding_and_retrieval"
                    )
                ),
            )
        )
        stack.enter_context(
            patch.object(
                folders_model_module.Folders, "get_folder_by_id", AsyncMock(return_value=folder)
            )
        )
        stack.enter_context(
            patch.object(retrieval_module, "has_folder_access", AsyncMock(return_value=True))
        )
        yield


# ---------------------------------------------------------------- narrow


@pytest.mark.asyncio
async def test_collaborator_cannot_attach_a_file_the_folder_owner_cannot_read(owui_module):
    """#27464 half one: bob has write access to alice's folder and attaches his
    own private file. Alice cannot read it, so the attach must be refused."""
    folders_module = owui_module("open_webui.routers.folders")
    folder = _folder("shared", ALICE.id, [])
    form = folders_module.FolderUpdateForm(
        data={"files": [{"type": "file", "id": "bobs-private-file"}]}
    )

    with _backend_world({"bobs-private-file": _file("bobs-private-file", BOB.id)}):
        with _folders_router(folders_module, folder) as (saved, _):
            with pytest.raises(HTTPException) as excinfo:
                await folders_module.update_folder_name_by_id(
                    request=SimpleNamespace(), id=folder.id, form_data=form, user=BOB, db=None
                )

    assert excinfo.value.status_code == 403, (
        "a collaborator smuggled their own file into someone else's folder, so everyone "
        "who can read that folder now reads the file through it (#27464)"
    )
    assert not saved.called, (
        "the folder was written before the owner's access was checked (#27464)"
    )


@pytest.mark.asyncio
async def test_folder_knowledge_drops_files_the_owner_lost_access_to(owui_module):
    """#27464 half two: the folder still lists a file its owner can no longer
    read. Expanding the folder for chat must drop it, even though the caller
    could read it on their own."""
    retrieval_module = owui_module("open_webui.retrieval.utils")
    entries = [{"type": "file", "id": "alices-file"}, {"type": "file", "id": "bobs-file"}]
    folder = _folder("shared", ALICE.id, entries)
    files = {
        "alices-file": _file("alices-file", ALICE.id),
        "bobs-file": _file("bobs-file", BOB.id),
    }

    with _retrieval_world(retrieval_module, folder, files):
        retrieved = await _retrieved_ids(
            retrieval_module, [{"type": "folder", "id": folder.id}], BOB
        )

    assert retrieved == {"alices-file"}, (
        "a folder served a file its owner can no longer read as chat knowledge, so stale "
        "attachments keep leaking after the owner's access is revoked (#27464)"
    )


# ---------------------------------------------------------------- broad
# The invariant: every point where files enter a folder gates on the OWNER.


@pytest.mark.asyncio
async def test_admin_collaborator_cannot_widen_a_folder_beyond_its_owner(owui_module):
    """An admin acting on someone else's folder is still bound by that owner:
    an admin bypass here would hand the owner's readers a file they may not see."""
    folders_module = owui_module("open_webui.routers.folders")
    folder = _folder("shared", ALICE.id, [])
    form = folders_module.FolderUpdateForm(
        data={"files": [{"type": "file", "id": "bobs-private-file"}]}
    )

    with _backend_world({"bobs-private-file": _file("bobs-private-file", BOB.id)}):
        with _folders_router(folders_module, folder) as (saved, _):
            with pytest.raises(HTTPException) as excinfo:
                await folders_module.update_folder_name_by_id(
                    request=SimpleNamespace(), id=folder.id, form_data=form, user=ADMIN, db=None
                )

    assert excinfo.value.status_code == 403, (
        "an admin attached a file into another user's folder that the owner cannot read (#27464)"
    )
    assert not saved.called


@pytest.mark.asyncio
async def test_subfolder_in_a_shared_folder_is_bounded_by_the_parent_owner(owui_module):
    """Creating a subfolder inside someone else's shared folder is the same
    attach in a different door, and the new folder inherits the parent's owner."""
    folders_module = owui_module("open_webui.routers.folders")
    parent = _folder("shared", ALICE.id, [])
    form = folders_module.FolderForm(
        name="Sub",
        parent_id=parent.id,
        data={"files": [{"type": "file", "id": "bobs-private-file"}]},
    )

    with _backend_world({"bobs-private-file": _file("bobs-private-file", BOB.id)}):
        with _folders_router(folders_module, parent) as (_, created):
            with pytest.raises(HTTPException) as excinfo:
                await folders_module.create_folder(
                    request=SimpleNamespace(), form_data=form, user=BOB, db=None
                )

    assert excinfo.value.status_code == 403, (
        "a collaborator created a subfolder owned by someone else and pre-loaded it with a "
        "file that owner cannot read (#27464)"
    )
    assert not created.called


@pytest.mark.asyncio
async def test_entries_that_cannot_be_access_checked_are_dropped(owui_module):
    """Entries of an unknown type or without an id have no access check to run,
    so keeping them lets any shape through the filter untouched."""
    access_module = owui_module("open_webui.utils.access_control.files")
    entries = [
        {"type": "file", "id": "alices-file"},
        {"type": "folder", "id": "nested"},
        {"type": "file"},
        "not-a-dict",
    ]

    with _backend_world({"alices-file": _file("alices-file", ALICE.id)}):
        accessible = await access_module.get_accessible_folder_files(entries, ALICE)

    assert accessible == [{"type": "file", "id": "alices-file"}], (
        "an entry that cannot be access-checked survived the folder filter, so an "
        "arbitrary payload reaches chat knowledge unchecked (#27464)"
    )


# ---------------------------------------------------------------- nearby
# Correct behaviour the fix must not have broken. These hold on both refs.


@pytest.mark.asyncio
async def test_owner_can_attach_a_file_they_own(owui_module):
    folders_module = owui_module("open_webui.routers.folders")
    folder = _folder("private", ALICE.id, [])
    form = folders_module.FolderUpdateForm(data={"files": [{"type": "file", "id": "alices-file"}]})

    with _backend_world({"alices-file": _file("alices-file", ALICE.id)}):
        with _folders_router(folders_module, folder) as (saved, _):
            await folders_module.update_folder_name_by_id(
                request=SimpleNamespace(), id=folder.id, form_data=form, user=ALICE, db=None
            )

    assert saved.called


@pytest.mark.asyncio
async def test_collaborator_can_attach_a_collection_the_owner_can_read(owui_module):
    """Bob may still contribute to alice's folder as long as alice can read what
    he attaches, so shared folders keep working."""
    folders_module = owui_module("open_webui.routers.folders")
    folder = _folder("shared", ALICE.id, [])
    form = folders_module.FolderUpdateForm(
        data={"files": [{"type": "collection", "id": "team-kb"}]}
    )

    with _backend_world({}, readable_collections={("team-kb", ALICE.id), ("team-kb", BOB.id)}):
        with _folders_router(folders_module, folder) as (saved, _):
            await folders_module.update_folder_name_by_id(
                request=SimpleNamespace(), id=folder.id, form_data=form, user=BOB, db=None
            )

    assert saved.called, (
        "a collaborator was blocked from attaching a file the owner can read (#27464)"
    )


@pytest.mark.asyncio
async def test_empty_folder_contributes_no_sources(owui_module):
    retrieval_module = owui_module("open_webui.retrieval.utils")
    folder = _folder("shared", ALICE.id, [])

    with _retrieval_world(retrieval_module, folder, {}):
        retrieved = await _retrieved_ids(
            retrieval_module, [{"type": "folder", "id": folder.id}], BOB
        )

    assert retrieved == set()


@pytest.mark.asyncio
async def test_owner_access_change_applies_without_reattaching(owui_module):
    """The stored entries never change; only the owner's grant does. Filtering
    happens at use time, so the entry has to disappear on its own."""
    access_module = owui_module("open_webui.utils.access_control.files")
    entries = [{"type": "collection", "id": "team-kb"}]

    with _backend_world({}, readable_collections={("team-kb", ALICE.id)}):
        while_granted = await access_module.get_accessible_folder_files(entries, ALICE)
    with _backend_world({}):
        after_revoke = await access_module.get_accessible_folder_files(entries, ALICE)

    assert while_granted == entries
    assert after_revoke == [], (
        "a revoked collection stayed in the folder listing until it was re-attached (#27464)"
    )
