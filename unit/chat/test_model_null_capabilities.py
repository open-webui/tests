"""Regression: resolving a model's default features must not crash when the
model's `meta.capabilities` is explicitly null.

open-webui 0.10.2 fix `650b81792` (#26412): the feature resolver did
`meta.get('capabilities', {})`, which returns `None` when the key is PRESENT with
a null value (the default only applies to a missing key). The next line then did
`capabilities.get(feature_id)` → `AttributeError: 'NoneType' object has no
attribute 'get'`, breaking chat for models with unset capabilities whenever the
memory feature or automations were involved. Fix: `meta.get('capabilities') or {}`.

0.11.1 (`2649e3305`) merged `_resolve_model_features` and its three siblings into
`_resolve_model_defaults`, which returns (tool_ids, features, filter_ids, terminal_id).
The `or {}` guard moved with it, so the tests now unpack the features slot.

`Config.get` (async, DB-backed) is patched so this is offline/deterministic.

Discriminates: passes with `or {}`, errors with `, {}` (None.get).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


def _app_with_model_meta(meta: dict) -> SimpleNamespace:
    """Minimal app exposing app.state.MODELS[<id>].info.meta = meta."""
    return SimpleNamespace(state=SimpleNamespace(MODELS={"m": {"info": {"meta": meta}}}))


async def _resolve_features(mod, app) -> dict:
    """0.11.1 folded the resolver into `_resolve_model_defaults`, which returns a
    four-tuple; earlier refs expose `_resolve_model_features`, returning the dict."""
    resolver = getattr(mod, "_resolve_model_defaults", None)
    with patch.object(mod.Config, "get", AsyncMock(return_value=True)):
        if resolver is not None:
            _, features, _, _ = await resolver(app, "m")
            return features
        return await mod._resolve_model_features(app, "m")


@pytest.mark.asyncio
async def test_null_capabilities_does_not_crash(automations_module):
    """capabilities=None + a default feature id must NOT raise (None.get)."""
    app = _app_with_model_meta({"defaultFeatureIds": ["web_search"], "capabilities": None})
    features = await _resolve_features(automations_module, app)
    assert features == {}, f"null capabilities should yield no features, got {features!r}"


@pytest.mark.asyncio
async def test_missing_capabilities_key_does_not_crash(automations_module):
    """The key absent entirely (the case `, {}` already handled) must also be fine."""
    app = _app_with_model_meta({"defaultFeatureIds": ["web_search"]})
    features = await _resolve_features(automations_module, app)
    assert features == {}


@pytest.mark.asyncio
async def test_enabled_capability_resolves_feature(automations_module):
    """Sanity (the positive path): a real capability + admin-enabled feature +
    default id resolves the feature to True."""
    app = _app_with_model_meta(
        {"defaultFeatureIds": ["web_search"], "capabilities": {"web_search": True}}
    )
    features = await _resolve_features(automations_module, app)
    assert features.get("web_search") is True, features
