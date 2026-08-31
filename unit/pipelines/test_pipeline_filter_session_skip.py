"""Regression test for the pipeline filter session setup skip (PR #29146, commit 88bbe4e1d).

`process_pipeline_inlet_filter` and `process_pipeline_outlet_filter` opened an
`aiohttp.ClientSession` before checking whether any pipeline filter matched, so the default
deployment (no pipelines configured at all) paid a session construct and teardown on every
chat message and background task. Both now return the payload untouched when the sorted
filter list is empty.

Discriminates: passes on v0.11.3, fails on v0.11.1 (the session is constructed even with no
pipeline filters to call).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def pipelines_router(owui_module):
    return owui_module("open_webui.routers.pipelines")


class RecordingClientSession:
    """Stand-in for aiohttp.ClientSession, the only I/O boundary these two functions have."""

    constructed = 0

    def __init__(self, *args, **kwargs):
        type(self).constructed += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def session_recorder(pipelines_router, monkeypatch):
    RecordingClientSession.constructed = 0
    monkeypatch.setattr(
        pipelines_router.aiohttp, "ClientSession", RecordingClientSession, raising=True
    )
    return RecordingClientSession


def _user():
    return SimpleNamespace(id="u1", email="u1@example.com", name="U1", role="user")


def _pipeline_filter(filter_id: str = "pf1", priority: int = 0) -> dict:
    """A pipeline filter model with no urlIdx, so the request loop skips it before any HTTP."""
    return {
        "id": filter_id,
        "pipeline": {"type": "filter", "priority": priority, "pipelines": ["*"]},
    }


PLAIN_MODELS = {"m": {"id": "m"}}
PIPELINE_MODELS = {"m": {"id": "m"}, "pf1": _pipeline_filter()}


# =============================================================================
# Narrow -- no pipelines configured means no session
# =============================================================================


@pytest.mark.asyncio
async def test_inlet_with_no_pipelines_opens_no_session(pipelines_router, session_recorder):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    result = await pipelines_router.process_pipeline_inlet_filter(
        SimpleNamespace(), payload, _user(), PLAIN_MODELS
    )

    assert session_recorder.constructed == 0, (
        "an aiohttp session was opened for a deployment with no pipeline filters (#29146)"
    )
    assert result is payload


@pytest.mark.asyncio
async def test_outlet_with_no_pipelines_opens_no_session(pipelines_router, session_recorder):
    payload = {"model": "m", "messages": [{"role": "assistant", "content": "hi"}]}

    result = await pipelines_router.process_pipeline_outlet_filter(
        SimpleNamespace(), payload, _user(), PLAIN_MODELS
    )

    assert session_recorder.constructed == 0, (
        "an aiohttp session was opened for a deployment with no pipeline filters (#29146)"
    )
    assert result is payload


# =============================================================================
# Broad -- the skip holds for every shape that resolves to no filters
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models",
    [
        PLAIN_MODELS,
        # A pipeline filter that targets some other model.
        {
            "m": {"id": "m"},
            "pf1": {
                "id": "pf1",
                "pipeline": {"type": "filter", "priority": 0, "pipelines": ["other"]},
            },
        },
        # A pipeline that is not of type 'filter'.
        {
            "m": {"id": "m"},
            "pf1": {
                "id": "pf1",
                "pipeline": {"type": "manifold", "priority": 0, "pipelines": ["*"]},
            },
        },
    ],
)
@pytest.mark.parametrize("stage", ["inlet", "outlet"])
async def test_no_matching_filter_opens_no_session(
    pipelines_router, session_recorder, models, stage
):
    handler = getattr(pipelines_router, f"process_pipeline_{stage}_filter")
    payload = {"model": "m", "messages": []}

    await handler(SimpleNamespace(), payload, _user(), models)

    assert session_recorder.constructed == 0


# =============================================================================
# Nearby -- unchanged behaviour when pipelines really are configured
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["inlet", "outlet"])
async def test_matching_pipeline_filter_still_opens_a_session(
    pipelines_router, session_recorder, stage
):
    handler = getattr(pipelines_router, f"process_pipeline_{stage}_filter")
    payload = {"model": "m", "messages": []}

    result = await handler(SimpleNamespace(), payload, _user(), PIPELINE_MODELS)

    assert session_recorder.constructed == 1
    assert result == payload


@pytest.mark.asyncio
async def test_pipeline_model_itself_still_opens_a_session(pipelines_router, session_recorder):
    """The requested model carrying its own 'pipeline' key counts as a filter on both stages."""
    models = {"m": _pipeline_filter("m")}
    payload = {"model": "m", "messages": []}

    await pipelines_router.process_pipeline_inlet_filter(
        SimpleNamespace(), payload, _user(), models
    )

    assert session_recorder.constructed == 1


@pytest.mark.asyncio
async def test_sorted_filters_ordering_is_unchanged(pipelines_router):
    models = {
        "m": {"id": "m"},
        "late": _pipeline_filter("late", priority=9),
        "early": _pipeline_filter("early", priority=1),
    }

    assert [f["id"] for f in pipelines_router.get_sorted_filters("m", models)] == ["early", "late"]
