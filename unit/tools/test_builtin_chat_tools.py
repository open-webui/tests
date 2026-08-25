"""Regression tests for the builtin chat tools a model calls (tools/builtin.py).

Five fixes shipped in 0.11.1:

* f8ac75d188 (models/automations.py, search_automations): the folder filter was
  guarded by `if folder_id is not None`, and the list_automations tool passes
  `folder_id=''` whenever the model names no folder. An empty string is not
  None, so the query became `folder_id IS NULL` and every automation living in
  a folder disappeared from the listing.
* 90bb94abf9 (update_automation): a blank `folder_id` or `model_id` filled in by
  the model instead of omitted was written straight through, moving the
  automation out of its folder and clearing its model.
* 1deeaf71d (PR open-webui/open-webui#27777, update_calendar_event): the update
  form was built from all locals, so every field the model did not mention went
  in as None and `update_event_by_id`'s `exclude_unset` dump wiped the stored
  value.
* 9550731cc1 (issue open-webui/open-webui#27717, search_calendar_events): the
  open-ended upper bound was `now_ns + 365*86400*1e12`, roughly 3.2e19, which
  overflows PostgreSQL's bigint, so a calendar search with no end date failed.
* 54cefd2b9 (PR open-webui/open-webui#27642, issue #27641): `_has_read_access_to_file`
  and the agentic retrieval tools rebuilt a UserModel from only id and role.
  UserModel requires email, name and the timestamps, so the group-shared file
  path raised a ValidationError, and the embedding calls got a stub user with no
  email for instances forwarding user details to their embedding service.

The tests drive the real tool functions and stub only the persistence and
embedding boundaries.

Discriminates: passes on v0.11.1, fails on v0.11.0 (foldered automations vanish
from the listing, blank fields overwrite stored values, unmentioned calendar
fields are set to None, the open-ended calendar bound exceeds bigint, and the
shared-file and embedding paths never see a usable user).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]

BIGINT_MAX = 2**63 - 1

USER = {
    "id": "user-1",
    "email": "alice@example.com",
    "name": "Alice",
    "role": "user",
    "timezone": "UTC",
    "last_active_at": 0,
    "updated_at": 0,
    "created_at": 0,
}
OWNER = SimpleNamespace(id=USER["id"], timezone="UTC")
REQUEST = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def _payload(raw: str) -> dict:
    return json.loads(raw)


# --- automations listing ---------------------------------------------------


def _automation_row(
    automations_module, automation_id: str, name: str, folder_id: str | None, active=True
):
    return automations_module.Automation(
        id=automation_id,
        user_id=USER["id"],
        folder_id=folder_id,
        name=name,
        data={"prompt": "do the thing", "model_id": "gpt-x", "rrule": "FREQ=DAILY"},
        meta=None,
        is_active=active,
        last_run_at=None,
        next_run_at=None,
        created_at=0,
        updated_at=0,
    )


@pytest_asyncio.fixture()
async def automation_store(builtin_tools_module, owui_module):
    """Serve the automation table from a throwaway in-memory database."""
    automations_module = owui_module("open_webui.models.automations")
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    table = automations_module.Automation.__table__
    async with engine.begin() as conn:
        await conn.run_sync(table.metadata.create_all, tables=[table])

    sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sessionmaker() as session:
        session.add(_automation_row(automations_module, "auto-in", "In a folder", "folder-1"))
        session.add(_automation_row(automations_module, "auto-out", "Loose", None))
        session.add(
            _automation_row(automations_module, "auto-paused", "Paused", "folder-1", active=False)
        )
        await session.commit()

        @asynccontextmanager
        async def _db_context(db=None):
            yield session

        with (
            patch.object(automations_module, "get_async_db_context", _db_context),
            patch.object(
                owui_module("open_webui.models.users").Users,
                "get_user_by_id",
                AsyncMock(return_value=OWNER),
            ),
        ):
            yield builtin_tools_module

    await engine.dispose()


async def test_listing_without_a_folder_includes_foldered_automations(automation_store) -> None:
    """Regression for f8ac75d188: folder_id='' must not mean 'no folder'."""
    result = _payload(
        await automation_store.list_automations(folder_id="", __request__=REQUEST, __user__=USER)
    )

    assert {item["id"] for item in result["automations"]} == {"auto-in", "auto-out", "auto-paused"}
    assert result["total"] == 3


async def test_listing_with_an_unset_folder_includes_foldered_automations(automation_store) -> None:
    result = _payload(await automation_store.list_automations(__request__=REQUEST, __user__=USER))

    assert {item["id"] for item in result["automations"]} == {"auto-in", "auto-out", "auto-paused"}


async def test_listing_with_a_named_folder_filters_to_that_folder(
    automation_store, owui_module
) -> None:
    folders_module = owui_module("open_webui.models.folders")
    with patch.object(
        folders_module.Folders,
        "get_folder_by_id_and_user_id",
        AsyncMock(return_value=SimpleNamespace(id="folder-1")),
    ):
        result = _payload(
            await automation_store.list_automations(
                folder_id="folder-1", __request__=REQUEST, __user__=USER
            )
        )

    assert {item["id"] for item in result["automations"]} == {"auto-in", "auto-paused"}


async def test_listing_paused_automations_still_spans_folders(automation_store) -> None:
    result = _payload(
        await automation_store.list_automations(
            status="paused", folder_id="", __request__=REQUEST, __user__=USER
        )
    )

    assert [item["id"] for item in result["automations"]] == ["auto-paused"]


# --- automation updates ----------------------------------------------------


@asynccontextmanager
async def _automation_update(owui_module):
    """Drive update_automation against a stored automation, capturing the write."""
    automations_module = owui_module("open_webui.models.automations")
    routers_module = owui_module("open_webui.routers.automations")
    users_module = owui_module("open_webui.models.users")

    stored = SimpleNamespace(
        id="auto-1",
        user_id=USER["id"],
        folder_id="folder-1",
        name="Daily digest",
        data={"prompt": "summarise", "model_id": "gpt-x", "rrule": "FREQ=DAILY"},
        is_active=True,
    )
    captured: dict = {}

    async def _update_by_id(automation_id, form, next_run_at, **kwargs):
        captured["form"] = form
        return SimpleNamespace(
            id=automation_id,
            name=form.name,
            folder_id=form.folder_id,
            data=form.data.model_dump(),
            is_active=form.is_active,
        )

    with (
        patch.object(users_module.Users, "get_user_by_id", AsyncMock(return_value=OWNER)),
        patch.object(automations_module.Automations, "get_by_id", AsyncMock(return_value=stored)),
        patch.object(automations_module.Automations, "update_by_id", _update_by_id),
        patch.object(routers_module, "check_automation_limits", AsyncMock(return_value=None)),
    ):
        yield captured


async def test_blank_folder_id_keeps_the_current_folder(builtin_tools_module, owui_module) -> None:
    """Regression for 90bb94abf9: a blank folder must not evict the automation."""
    async with _automation_update(owui_module) as captured:
        result = _payload(
            await builtin_tools_module.update_automation(
                "auto-1", name="Nightly digest", folder_id="", __request__=REQUEST, __user__=USER
            )
        )

    assert captured["form"].folder_id == "folder-1"
    assert result["folder_id"] == "folder-1"


async def test_blank_model_id_keeps_the_current_model(builtin_tools_module, owui_module) -> None:
    """Regression for 90bb94abf9: a blank model must not clear the stored model."""
    async with _automation_update(owui_module) as captured:
        result = _payload(
            await builtin_tools_module.update_automation(
                "auto-1", name="Nightly digest", model_id="   ", __request__=REQUEST, __user__=USER
            )
        )

    assert captured["form"].data.model_id == "gpt-x"
    assert result["model_id"] == "gpt-x"


async def test_named_folder_and_model_are_applied(builtin_tools_module, owui_module) -> None:
    folders_module = owui_module("open_webui.models.folders")
    async with _automation_update(owui_module) as captured:
        with patch.object(
            folders_module.Folders,
            "get_folder_by_id_and_user_id",
            AsyncMock(return_value=SimpleNamespace(id="folder-2")),
        ):
            await builtin_tools_module.update_automation(
                "auto-1", model_id="gpt-y", folder_id="folder-2", __request__=REQUEST, __user__=USER
            )

    assert captured["form"].folder_id == "folder-2"
    assert captured["form"].data.model_id == "gpt-y"


async def test_untouched_fields_survive_an_automation_rename(
    builtin_tools_module, owui_module
) -> None:
    async with _automation_update(owui_module) as captured:
        await builtin_tools_module.update_automation(
            "auto-1", name="Renamed", __request__=REQUEST, __user__=USER
        )

    form = captured["form"]
    assert form.name == "Renamed"
    assert (form.data.prompt, form.data.rrule) == ("summarise", "FREQ=DAILY")


# --- calendar event updates ------------------------------------------------


@asynccontextmanager
async def _calendar_update(owui_module):
    """Drive update_calendar_event against a stored event, capturing the form."""
    calendar_module = owui_module("open_webui.models.calendar")

    stored = SimpleNamespace(
        id="event-1",
        user_id=USER["id"],
        calendar_id="cal-1",
        title="Standup",
        description="daily sync",
        start_at=1_700_000_000_000_000_000,
        end_at=1_700_003_600_000_000_000,
        all_day=False,
        location="Room 3",
        color="#fff",
        is_cancelled=False,
        meta={"alert_minutes": 10},
    )
    captured: dict = {}

    async def _update_event_by_id(event_id, form, **kwargs):
        captured["form"] = form
        return stored

    with (
        patch.object(
            calendar_module.CalendarEvents, "get_event_by_id", AsyncMock(return_value=stored)
        ),
        patch.object(calendar_module.CalendarEvents, "update_event_by_id", _update_event_by_id),
    ):
        yield captured


async def test_renaming_an_event_leaves_every_other_field_unset(
    builtin_tools_module, owui_module
) -> None:
    """Regression for PR open-webui/open-webui#27777."""
    async with _calendar_update(owui_module) as captured:
        await builtin_tools_module.update_calendar_event(
            "event-1", title="Standup (moved)", __request__=REQUEST, __user__=USER
        )

    assert captured["form"].model_dump(exclude_unset=True) == {"title": "Standup (moved)"}


async def test_cancelling_an_event_leaves_every_other_field_unset(
    builtin_tools_module, owui_module
) -> None:
    async with _calendar_update(owui_module) as captured:
        await builtin_tools_module.update_calendar_event(
            "event-1", is_cancelled=True, __request__=REQUEST, __user__=USER
        )

    assert captured["form"].model_dump(exclude_unset=True) == {"is_cancelled": True}


async def test_moving_an_event_leaves_every_other_field_unset(
    builtin_tools_module, owui_module
) -> None:
    async with _calendar_update(owui_module) as captured:
        await builtin_tools_module.update_calendar_event(
            "event-1", start="2026-04-20 09:00", __request__=REQUEST, __user__=USER
        )

    written = captured["form"].model_dump(exclude_unset=True)
    assert set(written) == {"start_at"}
    assert written["start_at"] > 0


async def test_every_mentioned_calendar_field_is_written(builtin_tools_module, owui_module) -> None:
    async with _calendar_update(owui_module) as captured:
        await builtin_tools_module.update_calendar_event(
            "event-1",
            title="Retro",
            description="quarterly",
            location="Room 4",
            all_day=True,
            reminder_minutes=30,
            __request__=REQUEST,
            __user__=USER,
        )

    written = captured["form"].model_dump(exclude_unset=True)
    assert written == {
        "title": "Retro",
        "description": "quarterly",
        "location": "Room 4",
        "all_day": True,
        "meta": {"alert_minutes": 30},
    }


async def test_updating_an_unknown_event_reports_not_found(
    builtin_tools_module, owui_module
) -> None:
    calendar_module = owui_module("open_webui.models.calendar")
    with patch.object(
        calendar_module.CalendarEvents, "get_event_by_id", AsyncMock(return_value=None)
    ):
        result = _payload(
            await builtin_tools_module.update_calendar_event(
                "nope", title="x", __request__=REQUEST, __user__=USER
            )
        )

    assert result == {"error": "Event not found"}


# --- calendar search bounds ------------------------------------------------


@asynccontextmanager
async def _calendar_range(owui_module):
    calendar_module = owui_module("open_webui.models.calendar")
    captured: dict = {}

    async def _get_events_by_range(user_id, start, end, **kwargs):
        captured.update(start=start, end=end)
        return []

    with patch.object(calendar_module.CalendarEvents, "get_events_by_range", _get_events_by_range):
        yield captured


async def test_open_ended_search_upper_bound_fits_a_bigint(
    builtin_tools_module, owui_module
) -> None:
    """Regression for issue open-webui/open-webui#27717."""
    async with _calendar_range(owui_module) as captured:
        await builtin_tools_module.search_calendar_events(
            start="2026-04-20 00:00", __request__=REQUEST, __user__=USER
        )

    assert captured["end"] <= BIGINT_MAX


async def test_open_ended_search_with_a_query_fits_a_bigint(
    builtin_tools_module, owui_module
) -> None:
    async with _calendar_range(owui_module) as captured:
        await builtin_tools_module.search_calendar_events(
            query="retro", start="2026-04-20 00:00", __request__=REQUEST, __user__=USER
        )

    assert captured["end"] <= BIGINT_MAX


async def test_open_ended_lower_bound_fits_a_bigint(builtin_tools_module, owui_module) -> None:
    async with _calendar_range(owui_module) as captured:
        await builtin_tools_module.search_calendar_events(
            end="2026-04-27 00:00", __request__=REQUEST, __user__=USER
        )

    assert captured["start"] == 0
    assert 0 < captured["end"] <= BIGINT_MAX


async def test_a_closed_range_passes_both_bounds_through(builtin_tools_module, owui_module) -> None:
    async with _calendar_range(owui_module) as captured:
        await builtin_tools_module.search_calendar_events(
            start="2026-04-20 00:00", end="2026-04-27 00:00", __request__=REQUEST, __user__=USER
        )

    assert 0 < captured["start"] < captured["end"] <= BIGINT_MAX


async def test_an_unparseable_start_is_reported(builtin_tools_module) -> None:
    result = _payload(
        await builtin_tools_module.search_calendar_events(
            start="next tuesday-ish", __request__=REQUEST, __user__=USER
        )
    )

    assert "Invalid start datetime" in result["error"]


# --- user context in file access and retrieval -----------------------------


SHARED_FILE = SimpleNamespace(
    id="file-1",
    user_id="someone-else",
    filename="shared.md",
    data={"content": "the shared body"},
    meta={},
    created_at=0,
    updated_at=0,
)


async def test_shared_file_access_check_receives_the_whole_user(
    builtin_tools_module, owui_module
) -> None:
    """Regression for PR open-webui/open-webui#27642: a stub UserModel never validated."""
    files_module = owui_module("open_webui.models.files")
    access_module = owui_module("open_webui.utils.access_control.files")
    captured: dict = {}

    async def _has_access_to_file(file_id, access_type, user, **kwargs):
        captured["user"] = user
        return True

    with (
        patch.object(files_module.Files, "get_file_by_id", AsyncMock(return_value=SHARED_FILE)),
        patch.object(access_module, "has_access_to_file", _has_access_to_file),
    ):
        result = _payload(
            await builtin_tools_module.view_file("file-1", __request__=REQUEST, __user__=USER)
        )

    assert result["content"] == "the shared body"
    assert (captured["user"].email, captured["user"].name) == ("alice@example.com", "Alice")


async def test_grep_over_a_shared_file_receives_the_whole_user(
    builtin_tools_module, owui_module
) -> None:
    files_module = owui_module("open_webui.models.files")
    access_module = owui_module("open_webui.utils.access_control.files")
    captured: dict = {}

    async def _has_access_to_file(file_id, access_type, user, **kwargs):
        captured["user"] = user
        return False

    with (
        patch.object(files_module.Files, "get_file_by_id", AsyncMock(return_value=SHARED_FILE)),
        patch.object(access_module, "has_access_to_file", _has_access_to_file),
    ):
        await builtin_tools_module.grep_knowledge_files(
            "shared", file_id="file-1", __request__=REQUEST, __user__=USER
        )

    assert (captured["user"].email, captured["user"].name) == ("alice@example.com", "Alice")


async def test_knowledge_base_search_sends_the_whole_user_to_the_embedder(
    builtin_tools_module, owui_module
) -> None:
    """Regression for issue open-webui/open-webui#27641."""
    groups_module = owui_module("open_webui.models.groups")
    knowledge_module = owui_module("open_webui.models.knowledge")
    captured: dict = {}

    async def _embed(text, prefix=None, user=None):
        captured["user"] = user
        return [0.0, 0.1]

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(EMBEDDING_FUNCTION=_embed)))
    empty = SimpleNamespace(items=[], total=0)

    with (
        patch.object(groups_module.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        patch.object(
            knowledge_module.Knowledges, "search_knowledge_bases", AsyncMock(return_value=empty)
        ),
    ):
        result = _payload(
            await builtin_tools_module.query_knowledge_bases(
                "budget", __request__=request, __user__=USER
            )
        )

    assert "error" not in result
    assert (captured["user"].email, captured["user"].name) == ("alice@example.com", "Alice")


async def test_the_owner_reads_their_own_file_without_an_access_grant(
    builtin_tools_module, owui_module
) -> None:
    files_module = owui_module("open_webui.models.files")
    access_module = owui_module("open_webui.utils.access_control.files")
    own_file = SimpleNamespace(
        id="file-2",
        user_id=USER["id"],
        filename="mine.md",
        data={"content": "my own body"},
        meta={},
        created_at=0,
        updated_at=0,
    )
    grant_check = AsyncMock(return_value=False)

    with (
        patch.object(files_module.Files, "get_file_by_id", AsyncMock(return_value=own_file)),
        patch.object(access_module, "has_access_to_file", grant_check),
    ):
        result = _payload(
            await builtin_tools_module.view_file("file-2", __request__=REQUEST, __user__=USER)
        )

    assert result["content"] == "my own body"
    grant_check.assert_not_awaited()


async def test_a_denied_shared_file_reads_as_missing(builtin_tools_module, owui_module) -> None:
    files_module = owui_module("open_webui.models.files")
    access_module = owui_module("open_webui.utils.access_control.files")

    with (
        patch.object(files_module.Files, "get_file_by_id", AsyncMock(return_value=SHARED_FILE)),
        patch.object(access_module, "has_access_to_file", AsyncMock(return_value=False)),
    ):
        result = _payload(
            await builtin_tools_module.view_file("file-1", __request__=REQUEST, __user__=USER)
        )

    assert result == {"error": "File not found"}
