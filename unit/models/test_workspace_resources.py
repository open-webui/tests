"""Workspace-resource model and router regressions fixed in Open WebUI 0.11.1.

Five independent fixes, all in the workspace-resource layer:

- Notes with a non-string ``data.content.md`` (commit 8d1c205d8e, issue #28222). A note whose
  markdown body had been stored as a dict or list broke the whole notes page. ``NoteModel`` /
  ``NoteForm`` / ``NoteUpdateForm`` gained a ``field_validator`` calling the new
  ``sanitize_note_data``, which renders dict/list bodies as a fenced ```json block and str()s
  anything else.
- Folder duplicate-name check (PR #28695, commit 1d1c14bd7, issue #28694). The check used
  ``Folder.name.ilike(name)``, which treats the candidate name as a LIKE pattern, so ``%`` and ``_``
  in it matched unrelated existing folders and a legitimate name was refused. Now
  ``func.lower(Folder.name) == func.lower(name)``.
- Skill ids outside the URL-path slug charset (PR #27660, commit 3df485582, issue #27655). A skill
  created with such an id became permanently unreachable. ``create_new_skill`` now rejects anything
  not matching ``[a-z0-9_-]+`` with a 400.
- Publicly shared read-only items (PR #27637, commit c0d09a5de, issue #27487). The read-only filter
  deliberately excluded public grants, so a note shared publicly for reading was reachable by link
  but listed nowhere. The public ``user:*`` grant now counts as a read grant.
- Shared folder ordering (PR #28804, commit 6db64c485). Two unordered selects gained
  ``.order_by(Folder.updated_at.desc())``, so renaming one shared folder no longer shuffles its
  neighbours on PostgreSQL.

Discriminates: passes on v0.11.1, fails on v0.11.0 (0.11.0 leaves note bodies unsanitized, matches
folder names as LIKE patterns, stores unreachable skill ids, hides publicly shared read-only items
and returns folders in physical row order).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

pytestmark = pytest.mark.regression


# --------------------------------------------------------------------------------------
# module fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_module(owui_module):
    return owui_module("open_webui.internal.db")


@pytest.fixture(scope="module")
def notes_module(owui_module):
    return owui_module("open_webui.models.notes")


@pytest.fixture(scope="module")
def folders_module(owui_module):
    return owui_module("open_webui.models.folders")


@pytest.fixture(scope="module")
def access_grants_module(owui_module):
    return owui_module("open_webui.models.access_grants")


@pytest.fixture(scope="module")
def skills_router_module(owui_module):
    return owui_module("open_webui.routers.skills")


@asynccontextmanager
async def scratch_session(*tables):
    """Throwaway in-memory SQLite session with only the given ORM tables created."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        async with engine.begin() as conn:
            for table in tables:
                await conn.run_sync(table.create)
        maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            yield session
    finally:
        await engine.dispose()


def share_sessions(monkeypatch, db_module):
    """get_async_db_context ignores a caller-supplied session unless sharing is on."""
    monkeypatch.setattr(db_module, "DATABASE_ENABLE_SESSION_SHARING", True)


async def add_folder(session, folders_module, *, name, user_id, parent_id=None, updated_at=0):
    folder = folders_module.Folder(
        id=str(uuid.uuid4()),
        parent_id=parent_id,
        user_id=user_id,
        name=name,
        items=None,
        meta=None,
        data=None,
        is_expanded=False,
        created_at=updated_at,
        updated_at=updated_at,
    )
    session.add(folder)
    await session.commit()
    return folder


async def add_note(session, notes_module, *, note_id, user_id, title="note"):
    session.add(
        notes_module.Note(
            id=note_id,
            user_id=user_id,
            title=title,
            data={"content": {"md": ""}},
            meta=None,
            created_at=0,
            updated_at=0,
        )
    )
    await session.commit()


async def add_grant(
    session,
    access_grants_module,
    *,
    resource_id,
    principal_id,
    permission,
    principal_type="user",
    resource_type="note",
):
    session.add(
        access_grants_module.AccessGrant(
            id=str(uuid.uuid4()),
            resource_type=resource_type,
            resource_id=resource_id,
            principal_type=principal_type,
            principal_id=principal_id,
            permission=permission,
            created_at=int(time.time()),
        )
    )
    await session.commit()


# ======================================================================================
# 45 — notes saved in an unexpected shape (models/notes.py, issue #28222)
# ======================================================================================


@pytest.mark.parametrize("form_name", ["NoteForm", "NoteUpdateForm", "NoteModel"])
def test_note_dict_markdown_body_is_rendered_as_text(notes_module, form_name):
    """NARROW: a dict `content.md` must arrive as markdown text, not a dict."""
    payload = {
        "id": "n1",
        "user_id": "u1",
        "title": "t",
        "data": {"content": {"md": {"type": "doc", "text": "hi"}}},
        "created_at": 0,
        "updated_at": 0,
    }
    model = getattr(notes_module, form_name).model_validate(payload)

    md = model.data["content"]["md"]
    assert isinstance(md, str)
    assert md.startswith("```json")
    assert '"text": "hi"' in md


def test_note_list_markdown_body_is_rendered_as_text(notes_module):
    """NARROW: a list body is fenced as JSON too."""
    form = notes_module.NoteForm.model_validate({"title": "t", "data": {"content": {"md": [1, 2]}}})

    assert form.data["content"]["md"] == "```json\n[\n  1,\n  2\n]\n```"


def test_note_scalar_markdown_body_is_stringified(notes_module):
    """NARROW: a non-dict, non-list body is str()-ed rather than left as-is."""
    form = notes_module.NoteForm.model_validate({"title": "t", "data": {"content": {"md": 42}}})

    assert form.data["content"]["md"] == "42"


def test_note_data_that_is_not_a_dict_is_wrapped(notes_module):
    """NARROW: `data` itself being a scalar used to be a validation error."""
    form = notes_module.NoteForm.model_validate({"title": "t", "data": "just some text"})

    assert form.data == {"content": {"md": "just some text"}}


def test_note_dict_body_keeps_the_other_content_and_data_keys(notes_module):
    """BROAD: sanitizing the body must not drop the rest of the note."""
    form = notes_module.NoteForm.model_validate(
        {
            "title": "t",
            "data": {"versions": [1], "content": {"html": "<p>x</p>", "md": {"a": 1}}},
        }
    )

    assert form.data["versions"] == [1]
    assert form.data["content"]["html"] == "<p>x</p>"
    assert isinstance(form.data["content"]["md"], str)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {"content": {"md": "# already markdown"}},
        {"content": {"html": "<p>x</p>"}},
        {"content": "not a dict"},
        {},
    ],
)
def test_note_well_formed_data_passes_through_untouched(notes_module, data):
    """NEARBY: nothing else about `data` may be rewritten."""
    assert notes_module.NoteForm.model_validate({"title": "t", "data": data}).data == data


def test_note_unicode_body_is_not_escaped(notes_module):
    """NEARBY: the JSON rendering keeps non-ASCII readable."""
    form = notes_module.NoteForm.model_validate(
        {"title": "t", "data": {"content": {"md": {"a": "über"}}}}
    )

    assert "über" in form.data["content"]["md"]


# ======================================================================================
# 89 — folder names with unusual characters (models/folders.py, issue #28694)
# ======================================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_name", "candidate_name"),
    [
        ("Notes1", "Notes_"),
        ("Archive 2024", "Archive%"),
        ("Report", "%"),
    ],
)
async def test_folder_name_lookup_is_not_a_like_pattern(
    monkeypatch, db_module, folders_module, existing_name, candidate_name
):
    """NARROW: LIKE wildcards in a candidate name must not match a different folder."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        await add_folder(session, folders_module, name=existing_name, user_id="u1")

        match = await folders_module.Folders.get_folder_by_parent_id_and_user_id_and_name(
            None, "u1", candidate_name, db=session
        )

    assert match is None


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_name", ["Notes_", "Archive%"])
async def test_folder_with_wildcard_name_matches_only_itself(
    monkeypatch, db_module, folders_module, candidate_name
):
    """BROAD: a folder literally named with a wildcard is still found by its own name."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        await add_folder(session, folders_module, name=candidate_name, user_id="u1")

        match = await folders_module.Folders.get_folder_by_parent_id_and_user_id_and_name(
            None, "u1", candidate_name, db=session
        )

    assert match is not None
    assert match.name == candidate_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate_name", "expected"),
    [
        ("Projects", True),
        ("projects", True),
        ("PROJECTS", True),
        ("Project", False),
        ("Projectsx", False),
    ],
)
async def test_folder_duplicate_check_stays_case_insensitive(
    monkeypatch, db_module, folders_module, candidate_name, expected
):
    """NEARBY: the check is still a case-insensitive exact match."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        await add_folder(session, folders_module, name="Projects", user_id="u1")

        match = await folders_module.Folders.get_folder_by_parent_id_and_user_id_and_name(
            None, "u1", candidate_name, db=session
        )

    assert (match is not None) is expected


@pytest.mark.asyncio
async def test_folder_duplicate_check_is_scoped_to_owner_and_parent(
    monkeypatch, db_module, folders_module
):
    """NEARBY: another user's or another parent's folder is not a duplicate."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        await add_folder(session, folders_module, name="Shared", user_id="other")
        await add_folder(session, folders_module, name="Shared", user_id="u1", parent_id="p1")

        at_root = await folders_module.Folders.get_folder_by_parent_id_and_user_id_and_name(
            None, "u1", "Shared", db=session
        )
        under_parent = await folders_module.Folders.get_folder_by_parent_id_and_user_id_and_name(
            "p1", "u1", "Shared", db=session
        )

    assert at_root is None
    assert under_parent is not None


# ======================================================================================
# 100 — skill identifiers that cannot be reached (routers/skills.py, issue #27655)
# ======================================================================================


@pytest.fixture
def skill_create_call(monkeypatch, skills_router_module):
    """create_new_skill with only its I/O boundary stubbed; the id decision stays real."""
    module = skills_router_module
    inserted = []

    async def fake_insert(user_id, form_data, db=None):
        inserted.append(form_data.id)
        return SimpleNamespace(id=form_data.id, name=form_data.name)

    monkeypatch.setattr(module.Config, "get", AsyncMock(return_value={}))
    monkeypatch.setattr(module.Skills, "get_skill_by_id", AsyncMock(return_value=None))
    monkeypatch.setattr(module.Skills, "insert_new_skill", fake_insert)
    monkeypatch.setattr(module, "publish_event", AsyncMock(return_value=None))

    async def call(skill_id):
        form = module.SkillForm(id=skill_id, name="Skill", content="body")
        return await module.create_new_skill(
            request=MagicMock(),
            form_data=form,
            user=SimpleNamespace(id="u1", role="admin", name="Admin", email="a@example.com"),
            db=None,
        )

    return SimpleNamespace(call=call, inserted=inserted)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "skill_id", ["my/skill", "../etc", "skill?x", "skill#1", "skill.v1", "скилл"]
)
async def test_unreachable_skill_id_is_rejected(skill_create_call, skill_id):
    """NARROW: an id outside the URL-path slug charset must never be stored."""
    with pytest.raises(HTTPException) as excinfo:
        await skill_create_call.call(skill_id)

    assert excinfo.value.status_code == 400
    assert "Invalid skill ID" in str(excinfo.value.detail)
    assert skill_create_call.inserted == []


@pytest.mark.asyncio
async def test_skill_id_whose_only_problem_is_spaces_is_still_accepted(skill_create_call):
    """BROAD: the pre-existing space-to-dash normalisation still wins over the new check."""
    skill = await skill_create_call.call("My Skill")

    assert skill.id == "my-skill"
    assert skill_create_call.inserted == ["my-skill"]


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_id", ["my-skill", "my_skill", "skill123", "MySkill"])
async def test_valid_skill_ids_are_accepted(skill_create_call, skill_id):
    """NEARBY: the slug charset itself is untouched."""
    skill = await skill_create_call.call(skill_id)

    assert skill.id == skill_id.lower()


@pytest.mark.asyncio
async def test_duplicate_skill_id_still_reports_id_taken(
    monkeypatch, skills_router_module, skill_create_call
):
    """NEARBY: the duplicate check is not shadowed by the new charset check."""
    monkeypatch.setattr(
        skills_router_module.Skills,
        "get_skill_by_id",
        AsyncMock(return_value=SimpleNamespace(id="my-skill")),
    )

    with pytest.raises(HTTPException) as excinfo:
        await skill_create_call.call("my-skill")

    assert excinfo.value.status_code == 400
    assert "Invalid skill ID" not in str(excinfo.value.detail)
    assert skill_create_call.inserted == []


# ======================================================================================
# 121 — finding notes shared read only (models/access_grants.py, issue #27487)
# ======================================================================================


async def read_only_note_ids(session, access_grants_module, notes_module, user_id, group_ids=()):
    query = access_grants_module.AccessGrants.has_permission_filter(
        session,
        select(notes_module.Note),
        notes_module.Note,
        {"user_id": user_id, "group_ids": list(group_ids)},
        "note",
        permission="read_only",
    )
    result = await session.execute(query)
    return {note.id for note in result.scalars().all()}


@pytest.mark.asyncio
async def test_publicly_shared_read_only_note_is_listed(access_grants_module, notes_module):
    """NARROW: a public read grant must make the note listable, not link-only."""
    async with scratch_session(
        notes_module.Note.__table__, access_grants_module.AccessGrant.__table__
    ) as session:
        await add_note(session, notes_module, note_id="public-note", user_id="owner")
        await add_grant(
            session,
            access_grants_module,
            resource_id="public-note",
            principal_id="*",
            permission="read",
        )

        listed = await read_only_note_ids(session, access_grants_module, notes_module, "viewer")

    assert listed == {"public-note"}


@pytest.mark.asyncio
async def test_publicly_shared_read_only_note_is_listed_for_a_group_member(
    access_grants_module, notes_module
):
    """BROAD: the public grant counts regardless of which groups the viewer is in."""
    async with scratch_session(
        notes_module.Note.__table__, access_grants_module.AccessGrant.__table__
    ) as session:
        await add_note(session, notes_module, note_id="public-note", user_id="owner")
        await add_grant(
            session,
            access_grants_module,
            resource_id="public-note",
            principal_id="*",
            permission="read",
        )

        listed = await read_only_note_ids(
            session, access_grants_module, notes_module, "viewer", group_ids=["g1"]
        )

    assert listed == {"public-note"}


@pytest.mark.asyncio
async def test_directly_shared_read_only_note_is_listed(access_grants_module, notes_module):
    """NEARBY: an explicit per-user read grant was already listed."""
    async with scratch_session(
        notes_module.Note.__table__, access_grants_module.AccessGrant.__table__
    ) as session:
        await add_note(session, notes_module, note_id="direct-note", user_id="owner")
        await add_grant(
            session,
            access_grants_module,
            resource_id="direct-note",
            principal_id="viewer",
            permission="read",
        )

        listed = await read_only_note_ids(session, access_grants_module, notes_module, "viewer")

    assert listed == {"direct-note"}


@pytest.mark.asyncio
async def test_writable_and_owned_notes_stay_out_of_the_read_only_list(
    access_grants_module, notes_module
):
    """NEARBY: read_only still means read but not write, and never the viewer's own notes."""
    async with scratch_session(
        notes_module.Note.__table__, access_grants_module.AccessGrant.__table__
    ) as session:
        await add_note(session, notes_module, note_id="writable", user_id="owner")
        await add_grant(
            session,
            access_grants_module,
            resource_id="writable",
            principal_id="viewer",
            permission="read",
        )
        await add_grant(
            session,
            access_grants_module,
            resource_id="writable",
            principal_id="viewer",
            permission="write",
        )

        await add_note(session, notes_module, note_id="public-writable", user_id="owner")
        await add_grant(
            session,
            access_grants_module,
            resource_id="public-writable",
            principal_id="*",
            permission="read",
        )
        await add_grant(
            session,
            access_grants_module,
            resource_id="public-writable",
            principal_id="*",
            permission="write",
        )

        await add_note(session, notes_module, note_id="own", user_id="viewer")
        await add_grant(
            session, access_grants_module, resource_id="own", principal_id="*", permission="read"
        )

        await add_note(session, notes_module, note_id="unshared", user_id="owner")

        listed = await read_only_note_ids(session, access_grants_module, notes_module, "viewer")

    assert listed == set()


# ======================================================================================
# 132 — shared folders reordering themselves (models/folders.py, PR #28804)
# ======================================================================================


@pytest.mark.asyncio
async def test_child_folders_come_back_newest_first(monkeypatch, db_module, folders_module):
    """NARROW: without ORDER BY the rows came back in physical insertion order."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        for name, updated_at in (("oldest", 100), ("middle", 200), ("newest", 300)):
            await add_folder(
                session,
                folders_module,
                name=name,
                user_id="u1",
                parent_id="p1",
                updated_at=updated_at,
            )

        children = await folders_module.Folders.get_folders_by_parent_id_and_user_id(
            "p1", "u1", db=session
        )

    assert [folder.name for folder in children] == ["newest", "middle", "oldest"]


@pytest.mark.asyncio
async def test_get_folders_by_ids_returns_newest_first(monkeypatch, db_module, folders_module):
    """NARROW: the shared-folder listing now fetches by id set in one ordered query."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        folders = [
            await add_folder(
                session, folders_module, name=name, user_id="owner", updated_at=updated_at
            )
            for name, updated_at in (("oldest", 100), ("middle", 200), ("newest", 300))
        ]
        ids = [folder.id for folder in folders]

        fetched = await folders_module.Folders.get_folders_by_ids(ids, db=session)

    assert [folder.name for folder in fetched] == ["newest", "middle", "oldest"]


@pytest.mark.asyncio
async def test_get_folders_by_ids_ignores_unknown_ids(monkeypatch, db_module, folders_module):
    """BROAD: a stale grant pointing at a deleted folder must not break the listing."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        folder = await add_folder(
            session, folders_module, name="live", user_id="owner", updated_at=1
        )

        fetched = await folders_module.Folders.get_folders_by_ids([folder.id, "gone"], db=session)
        empty = await folders_module.Folders.get_folders_by_ids([], db=session)

    assert [f.name for f in fetched] == ["live"]
    assert empty == []


@pytest.mark.asyncio
async def test_child_folder_listing_stays_scoped_to_owner_and_parent(
    monkeypatch, db_module, folders_module
):
    """NEARBY: ordering must not widen what the query selects."""
    share_sessions(monkeypatch, db_module)
    async with scratch_session(folders_module.Folder.__table__) as session:
        await add_folder(
            session, folders_module, name="mine", user_id="u1", parent_id="p1", updated_at=1
        )
        await add_folder(
            session, folders_module, name="theirs", user_id="other", parent_id="p1", updated_at=2
        )
        await add_folder(
            session, folders_module, name="elsewhere", user_id="u1", parent_id="p2", updated_at=3
        )
        await add_folder(session, folders_module, name="root", user_id="u1", updated_at=4)

        children = await folders_module.Folders.get_folders_by_parent_id_and_user_id(
            "p1", "u1", db=session
        )
        roots = await folders_module.Folders.get_folders_by_parent_id_and_user_id(
            None, "u1", db=session
        )

    assert [folder.name for folder in children] == ["mine"]
    assert [folder.name for folder in roots] == ["root"]
