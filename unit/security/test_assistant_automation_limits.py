"""Regression: automations the assistant creates on the user's behalf must obey
the same limits as automations the user creates through the HTTP API.

open-webui 0.11.0 fix `076a84e3f` (#27523, issue #27121): the builtin chat tools
`create_automation` and `update_automation` wrote straight to `Automations.insert`
and `Automations.update_by_id`, skipping `check_automation_limits` that
`/api/v1/automations/create` and `/api/v1/automations/{id}/update` run. So a
non-admin could ask the model for unlimited automations, ignoring
`automations.max_count`, and could reschedule below `automations.min_interval`.
Both tools now call the same helper the routers use and return its rejection to
the model as a plain error message.

Discriminates: passes on v0.11.0, fails on v0.10.2 (tools return
`status: success` and insert/update the automation regardless of the limits).
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

# Clock-anchored rules, so the computed interval is exact and DTSTART-free.
RRULE_HOURLY = "RRULE:FREQ=HOURLY;INTERVAL=1"  # 3600s
RRULE_EVERY_30_MIN = "RRULE:FREQ=MINUTELY;INTERVAL=30"  # 1800s
HOUR_SECONDS = 3600

REQUEST = SimpleNamespace()


def _user(role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id="alice", role=role, timezone="UTC")


def _automation(id: str, rrule: str, user_id: str = "alice") -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        user_id=user_id,
        folder_id=None,
        name=f"automation {id}",
        data={"prompt": "check the news", "model_id": "gpt-test", "rrule": rrule},
        is_active=True,
    )


class FakeAutomationStore:
    """In-memory stand-in for the `Automations` table singleton."""

    def __init__(self, *entries: SimpleNamespace):
        self.entries = {entry.id: entry for entry in entries}
        self.inserted: list[SimpleNamespace] = []
        self.updated: list[SimpleNamespace] = []

    async def count_by_user(self, user_id: str, db=None) -> int:
        return sum(1 for entry in self.entries.values() if entry.user_id == user_id)

    async def get_by_id(self, id: str, db=None):
        return self.entries.get(id)

    async def insert(self, user_id: str, form, next_run_at, db=None):
        entry = _automation(f"new-{len(self.inserted)}", form.data.rrule, user_id=user_id)
        entry.name = form.name
        self.entries[entry.id] = entry
        self.inserted.append(entry)
        return entry

    async def update_by_id(self, id: str, form, next_run_at, db=None):
        entry = self.entries[id]
        entry.name = form.name
        entry.data = form.data.model_dump()
        self.updated.append(entry)
        return entry

    async def delete(self, id: str, db=None) -> bool:
        return self.entries.pop(id, None) is not None


class FakeRunStore:
    async def delete_by_automation(self, automation_id: str, db=None) -> int:
        return 0


@contextmanager
def _backend(owui_module, store: FakeAutomationStore, limits: dict, user: SimpleNamespace):
    """Patch the I/O boundary the tools and the routers share.

    `check_automation_limits` reads `Automations` from the router module while the
    tools read it from the model module, so the same store is bound to both.
    """
    users_models = owui_module("open_webui.models.users")
    automation_models = owui_module("open_webui.models.automations")
    config_models = owui_module("open_webui.models.config")
    automations_router = owui_module("open_webui.routers.automations")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(users_models.Users, "get_user_by_id", AsyncMock(return_value=user))
        )
        stack.enter_context(patch.object(automation_models, "Automations", store))
        stack.enter_context(patch.object(automations_router, "Automations", store))
        stack.enter_context(patch.object(automation_models, "AutomationRuns", FakeRunStore()))
        stack.enter_context(
            patch.object(
                config_models.Config,
                "get",
                AsyncMock(side_effect=lambda key, default=None: limits.get(key, default)),
            )
        )
        yield


async def _create_from_chat(builtin, **overrides) -> dict:
    kwargs = {
        "name": "morning briefing",
        "prompt": "summarise my inbox",
        "rrule": RRULE_HOURLY,
        "__request__": REQUEST,
        "__user__": {"id": "alice", "role": "user"},
        "__metadata__": {"model_id": "gpt-test"},
    }
    kwargs.update(overrides)
    return json.loads(await builtin.create_automation(**kwargs))


async def _update_from_chat(builtin, automation_id: str, **overrides) -> dict:
    kwargs = {
        "automation_id": automation_id,
        "__request__": REQUEST,
        "__user__": {"id": "alice", "role": "user"},
    }
    kwargs.update(overrides)
    return json.loads(await builtin.update_automation(**kwargs))


@pytest.mark.asyncio
async def test_chat_create_refused_at_max_count(builtin_tools_module, owui_module):
    """The bug: the model creates automation number max_count + 1."""
    max_count = 3
    store = FakeAutomationStore(*(_automation(f"a{i}", RRULE_HOURLY) for i in range(max_count)))
    limits = {"automations.max_count": max_count}

    with _backend(owui_module, store, limits, _user()):
        result = await _create_from_chat(builtin_tools_module)

    assert "error" in result, (
        f"the assistant created automation number {max_count + 1} for a user already at the "
        "automations.max_count limit, so the cap is bypassable from chat (#27121)"
    )
    assert str(max_count) in result["error"]
    assert store.inserted == [], "a rejected automation must not reach the database (#27121)"


@pytest.mark.asyncio
async def test_chat_create_refused_below_min_interval(builtin_tools_module, owui_module):
    """The bug: the model schedules a run more often than the floor allows."""
    min_interval = HOUR_SECONDS
    store = FakeAutomationStore()
    limits = {"automations.min_interval": min_interval}

    with _backend(owui_module, store, limits, _user()):
        result = await _create_from_chat(builtin_tools_module, rrule=RRULE_EVERY_30_MIN)

    assert "error" in result, (
        "the assistant scheduled an automation every 30 minutes while "
        f"automations.min_interval is {min_interval} seconds, so the frequency floor is "
        "bypassable from chat (#27121)"
    )
    assert str(min_interval) in result["error"]
    assert store.inserted == [], "a too-frequent automation must not reach the database (#27121)"


@pytest.mark.asyncio
async def test_chat_reschedule_refused_below_min_interval(builtin_tools_module, owui_module):
    """The reschedule half of the same hole: an existing automation is sped up."""
    min_interval = HOUR_SECONDS
    existing = _automation("a0", RRULE_HOURLY)
    store = FakeAutomationStore(existing)
    limits = {"automations.min_interval": min_interval}

    with _backend(owui_module, store, limits, _user()):
        result = await _update_from_chat(builtin_tools_module, "a0", rrule=RRULE_EVERY_30_MIN)

    assert "error" in result, (
        "the assistant rescheduled an existing automation below "
        f"automations.min_interval ({min_interval}s), so the floor is bypassable by update "
        "even when it holds on create (#27121)"
    )
    assert str(min_interval) in result["error"]
    assert store.updated == [], "a rejected reschedule must not be written (#27121)"
    assert existing.data["rrule"] == RRULE_HOURLY


@pytest.mark.asyncio
async def test_both_paths_share_one_limit_check(builtin_tools_module, owui_module):
    """The invariant: the HTTP path and the chat path must reach the *same*
    `check_automation_limits`, so the limits and the admin bypass cannot drift."""
    from fastapi import HTTPException

    automation_models = owui_module("open_webui.models.automations")
    automations_router = owui_module("open_webui.routers.automations")
    store = FakeAutomationStore()
    admin = _user(role="admin")

    limit_check = AsyncMock(side_effect=HTTPException(status_code=403, detail="limit reached"))
    form = automation_models.AutomationForm(
        name="morning briefing",
        data=automation_models.AutomationData(
            prompt="summarise my inbox", model_id="gpt-test", rrule=RRULE_HOURLY
        ),
    )

    with _backend(owui_module, store, {}, admin):
        with (
            patch.object(automations_router, "check_automation_limits", limit_check),
            patch.object(
                automations_router.Config,
                "get_many",
                AsyncMock(return_value={"automations.enable": True, "user.permissions": {}}),
            ),
        ):
            with pytest.raises(HTTPException):
                await automations_router.create_new_automation(REQUEST, form, admin, None)
            calls_after_http_path = limit_check.await_count

            result = await _create_from_chat(builtin_tools_module)

    assert calls_after_http_path == 1, "the HTTP create route no longer runs the limit check"
    assert limit_check.await_count == 2, (
        "the chat tool did not go through the router's check_automation_limits, so the chat "
        "path carries its own (missing) copy of the limits (#27121)"
    )
    assert "error" in result
    assert store.inserted == []


# The checks below hold on both sides of the fix: they prove the limits did not
# turn into a blanket refusal of assistant-created automations.


@pytest.mark.asyncio
async def test_chat_create_allowed_below_max_count(builtin_tools_module, owui_module):
    max_count = 3
    store = FakeAutomationStore(_automation("a0", RRULE_HOURLY))
    limits = {"automations.max_count": max_count, "automations.min_interval": HOUR_SECONDS}

    with _backend(owui_module, store, limits, _user()):
        result = await _create_from_chat(builtin_tools_module)

    assert result.get("status") == "success", result
    assert len(store.inserted) == 1


@pytest.mark.asyncio
async def test_chat_create_allowed_at_exactly_min_interval(builtin_tools_module, owui_module):
    """Boundary: the floor is inclusive, so an hourly rule passes a 3600s floor."""
    store = FakeAutomationStore()
    limits = {"automations.min_interval": HOUR_SECONDS}

    with _backend(owui_module, store, limits, _user()):
        result = await _create_from_chat(builtin_tools_module, rrule=RRULE_HOURLY)

    assert result.get("status") == "success", (
        "a schedule exactly at automations.min_interval must be accepted, otherwise the "
        "limit is off by one interval"
    )
    assert len(store.inserted) == 1


@pytest.mark.asyncio
async def test_deleting_an_automation_frees_a_slot(builtin_tools_module, owui_module):
    max_count = 2
    store = FakeAutomationStore(_automation("a0", RRULE_HOURLY), _automation("a1", RRULE_HOURLY))
    limits = {"automations.max_count": max_count}

    with _backend(owui_module, store, limits, _user()):
        deleted = json.loads(
            await builtin_tools_module.delete_automation(
                automation_id="a0",
                __request__=REQUEST,
                __user__={"id": "alice", "role": "user"},
            )
        )
        result = await _create_from_chat(builtin_tools_module)

    assert deleted.get("status") == "success", deleted
    assert result.get("status") == "success", (
        "deleting an automation must free a slot, otherwise a user who hits the cap once "
        "can never create another automation from chat"
    )


@pytest.mark.asyncio
async def test_admin_is_not_capped_in_chat(builtin_tools_module, owui_module):
    """Admins bypass the limits on the HTTP path, so chat must not cap them either."""
    max_count = 1
    store = FakeAutomationStore(_automation("a0", RRULE_HOURLY))
    limits = {"automations.max_count": max_count, "automations.min_interval": HOUR_SECONDS}

    with _backend(owui_module, store, limits, _user(role="admin")):
        result = await _create_from_chat(builtin_tools_module, rrule=RRULE_EVERY_30_MIN)

    assert result.get("status") == "success", result
    assert len(store.inserted) == 1
