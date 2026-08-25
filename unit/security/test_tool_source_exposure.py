"""Regression: read access to a tool must not hand out the tool's Python source.

open-webui 0.11.0 fix `c05de13b4` (PR #27005): `GET /tools/id/{id}` built its
reply as `ToolAccessResponse(**tools.model_dump(), write_access=...)`.
`ToolResponse` deliberately omits `content`, but `ToolUserResponse` sets
`ConfigDict(extra='allow')`, so the `content` key from `model_dump()` was
re-admitted and shipped. Any caller with a mere read grant, including every
authenticated user when a tool is read-shared with `*`, got the full source.
Tool source routinely carries hard-coded API keys and internal URLs. The fix
computes `write_access` first and pops `content` when the caller lacks it.

The same leak has a second mouth in the listing path: `Tools.get_tools`
accepted `defer_content` and ignored it (`stmt = stmt`), so `GET /tools/list`
loaded source for every tool it returned. 0.11.0 turned that flag into a real
column select.

Discriminates: passes on v0.11.0, fails on v0.10.2 (source returned to a
read-only caller, and the listing query still selects `tool.content`).
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

SECRET_MARKER = "sk-live-do-not-leak"
SECRET_SOURCE = f'API_KEY = "{SECRET_MARKER}"\n\nclass Tools:\n    pass\n'

OWNER = "owner-user"
READER = "reader-user"
WRITER = "writer-user"


def _user(id: str, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=id, role=role)


def _tool(tools_models):
    return tools_models.ToolModel(
        id="weather",
        user_id=OWNER,
        name="Weather",
        content=SECRET_SOURCE,
        specs=[{"name": "get_weather", "parameters": {}}],
        meta={"description": "Looks up the weather"},
        access_grants=[],
        updated_at=1,
        created_at=1,
    )


def _grants(read: bool = False, write: bool = False) -> AsyncMock:
    """Stand in for `AccessGrants.has_access`, answering per permission."""
    granted = {"read": read or write, "write": write}
    return AsyncMock(side_effect=lambda **kwargs: granted.get(kwargs["permission"], False))


async def _get_tool_by_id(router, tools_models, caller, grants, bypass_admin=False):
    with (
        patch.object(
            tools_models.Tools, "get_tool_by_id", AsyncMock(return_value=_tool(tools_models))
        ),
        patch.object(tools_models.AccessGrants, "has_access", grants),
        patch.object(router, "BYPASS_ADMIN_ACCESS_CONTROL", bypass_admin),
    ):
        return await router.get_tools_by_id("weather", user=caller, db=None)


def _source_is_visible(response) -> bool:
    return SECRET_MARKER in str(response.model_dump())


@pytest.fixture(scope="session")
def tools_router(owui_module):
    return owui_module("open_webui.routers.tools")


@pytest.fixture(scope="session")
def tools_models(owui_module):
    return owui_module("open_webui.models.tools")


# ── narrow: the leak itself ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,grants,bypass_admin,has_write_access",
    [
        (_user(OWNER), _grants(), False, True),
        (_user(WRITER), _grants(write=True), False, True),
        (_user(READER), _grants(read=True), False, False),
        (_user("root", role="admin"), _grants(), True, True),
        (_user("root", role="admin"), _grants(), False, False),
    ],
    ids=["owner", "write-grant", "read-grant", "admin-bypass", "admin-no-bypass"],
)
async def test_only_write_access_unlocks_the_tool_source(
    tools_router, tools_models, caller, grants, bypass_admin, has_write_access
):
    """The bug: a read grant is not permission to read the source. It is write
    access, not the admin role, that unlocks it."""
    response = await _get_tool_by_id(tools_router, tools_models, caller, grants, bypass_admin)

    assert response.write_access is has_write_access
    assert _source_is_visible(response) is has_write_access, (
        "tool source must be visible exactly to callers who can edit the tool; a "
        "read-only caller receiving it gets the API keys and internal URLs the "
        "source commonly embeds (#27005)"
    )


# ── broad: no tool-returning path leaks source to a read-only caller ─────


@pytest.mark.asyncio
async def test_deferred_tool_listing_query_omits_the_source_column(tools_models):
    """`Tools.get_tools(defer_content=True)` must really skip `Tool.content`,
    otherwise every listing endpoint loads source it must not hand out."""
    statements = []

    class _Result:
        def all(self):
            return []

        def scalars(self):
            return self

    async def _execute(stmt):
        statements.append(stmt)
        return _Result()

    @asynccontextmanager
    async def _db_context(db=None):
        yield SimpleNamespace(execute=_execute)

    with (
        patch.object(tools_models, "get_async_db_context", _db_context),
        patch.object(
            tools_models.AccessGrants, "get_grants_by_resources", AsyncMock(return_value={})
        ),
    ):
        assert await tools_models.Tools.get_tools(defer_content=True) == []

    selected_sql = str(statements[0])
    assert "content" not in selected_sql, (
        f"the deferred tool listing still selects the source column, so listings "
        f"carry every tool's Python source (#27005): {selected_sql}"
    )


@pytest.mark.asyncio
async def test_tool_list_endpoint_asks_for_deferred_content(tools_router, tools_models):
    """`GET /tools/list` must never ask the model layer for source in the
    first place; it dumps whole models into an `extra='allow'` response."""
    get_tools = AsyncMock(return_value=[])
    with (
        patch.object(tools_models.Tools, "get_tools", get_tools),
        patch.object(tools_router.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        patch.object(tools_router, "BYPASS_ADMIN_ACCESS_CONTROL", False),
        patch.object(tools_router, "ENABLE_PLUGINS", True),
    ):
        assert await tools_router.get_tool_list(user=_user(READER), db=None) == []

    # 0.11.1 fetches through Tools.get_tools with the caller's own access filter.
    assert get_tools.call_args.kwargs.get("user_id") == READER
    assert get_tools.call_args.kwargs.get("defer_content") is True, (
        "the tool list endpoint fetched full tool records, so source rides along "
        "into an extra='allow' response model (#27005)"
    )


@pytest.mark.asyncio
async def test_tool_export_requires_the_export_permission(tools_router):
    """`GET /tools/export` is the one path that returns source by design, so
    it stays behind an explicit permission rather than read access."""
    with (
        patch.object(tools_router, "has_permission", AsyncMock(return_value=False)),
        patch.object(tools_router.Config, "get", AsyncMock(return_value={})),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await tools_router.export_tools(request=None, user=_user(READER), db=None)
    assert excinfo.value.status_code == 401


# ── nearby: what read access is still supposed to get ───────────────────


@pytest.mark.asyncio
async def test_read_only_caller_still_gets_the_descriptive_fields(tools_router, tools_models):
    """Stripping source must not blank the record, or the workspace UI and the
    chat tool picker lose the tool."""
    response = await _get_tool_by_id(tools_router, tools_models, _user(READER), _grants(read=True))
    assert response.id == "weather"
    assert response.name == "Weather"
    assert response.meta.description == "Looks up the weather"
    assert response.specs == [{"name": "get_weather", "parameters": {}}]


@pytest.mark.asyncio
async def test_caller_without_any_grant_is_rejected(tools_router, tools_models):
    with pytest.raises(HTTPException) as excinfo:
        await _get_tool_by_id(tools_router, tools_models, _user("stranger"), _grants())
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_unknown_tool_is_a_404(tools_router, tools_models):
    with patch.object(tools_models.Tools, "get_tool_by_id", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as excinfo:
            await tools_router.get_tools_by_id("nope", user=_user(READER), db=None)
    assert excinfo.value.status_code == 404
