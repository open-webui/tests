"""No HTTP session is opened for pipeline filters when there are none to call.

PR #29146, commit `88bbe4e1d`: `process_pipeline_inlet_filter` and
`process_pipeline_outlet_filter` opened an `aiohttp.ClientSession` before checking whether any
pipeline filter matched, so the default deployment (no pipelines at all) built and tore down a
session on every chat message and background task. Both now return the payload untouched when
the sorted filter list is empty.

Stays a unit test: the saving is a session construction with no effect on any response.
`aiohttp.ClientSession` is replaced by a `create_autospec` stand-in, the only I/O boundary.

Discriminates: passes on bbfa876af; fails with the empty-filter early return removed (a session
is constructed with no pipeline filter to call).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]


def _filter(filter_id: str, targets: list[str], kind: str = "filter", priority: int = 0) -> dict:
    """A pipeline model with no `urlIdx`, so the request loop skips it before any HTTP."""
    pipeline = {"type": kind, "priority": priority, "pipelines": targets}
    return {"id": filter_id, "pipeline": pipeline}


@pytest.fixture(scope="session")
def pipelines_router(owui_module):
    return owui_module("open_webui.routers.pipelines")


@pytest.fixture(scope="session")
def user(owui_module):
    return owui_module("open_webui.models.users").UserModel(
        id="u1",
        name="U1",
        email="u1@example.com",
        role="user",
        profile_image_url="",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


@pytest.fixture
def client_session(pipelines_router):
    with patch.object(pipelines_router.aiohttp, "ClientSession", autospec=True) as session_class:
        yield session_class


async def _run(pipelines_router, stage: str, models: dict, user) -> dict:
    handler = getattr(pipelines_router, f"process_pipeline_{stage}_filter")
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    result = await handler(request=None, payload=payload, user=user, models=models)
    return {"payload": payload, "result": result}


@pytest.mark.parametrize("stage", ["inlet", "outlet"])
@pytest.mark.parametrize(
    "models",
    [
        {"m": {"id": "m"}},
        {"m": {"id": "m"}, "pf": _filter("pf", ["other-model"])},
        {"m": {"id": "m"}, "pf": _filter("pf", ["*"], kind="manifold")},
    ],
    ids=["no-pipelines", "filter-for-another-model", "not-a-filter"],
)
async def test_no_matching_filter_opens_no_session(
    pipelines_router, client_session, user, stage, models
):
    ran = await _run(pipelines_router, stage, models, user)

    client_session.assert_not_called()
    assert ran["result"] is ran["payload"]


@pytest.mark.parametrize("stage", ["inlet", "outlet"])
@pytest.mark.parametrize(
    "models",
    [{"m": {"id": "m"}, "pf": _filter("pf", ["*"])}, {"m": _filter("m", ["*"])}],
    ids=["global-filter", "pipeline-model-itself"],
)
async def test_a_matching_filter_still_opens_one_session(
    pipelines_router, client_session, user, stage, models
):
    await _run(pipelines_router, stage, models, user)

    client_session.assert_called_once()


async def test_filters_run_in_priority_order(pipelines_router):
    models = {"m": {"id": "m"}, "late": _filter("late", ["*"], priority=9)}
    models["early"] = _filter("early", ["*"], priority=1)

    ordered = pipelines_router.get_sorted_filters(model_id="m", models=models)

    assert [entry["id"] for entry in ordered] == ["early", "late"]
