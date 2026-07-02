"""Regression: resolving a model's default features must not crash when the
model's `meta.capabilities` is explicitly null.

open-webui 0.10.2 fix `650b81792` (#26412): `_resolve_model_features` did
`meta.get('capabilities', {})`, which returns `None` when the key is PRESENT with
a null value (the default only applies to a missing key). The next line then did
`capabilities.get(feature_id)` → `AttributeError: 'NoneType' object has no
attribute 'get'`, breaking chat for models with unset capabilities whenever the
memory feature or automations were involved. Fix: `meta.get('capabilities') or {}`.

`Config.get` (async, DB-backed) is patched so this is offline/deterministic.

Discriminates: passes on v0.10.2 (`or {}`), errors on v0.10.1 (`, {}` → None.get).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


def _app_with_model_meta(meta: dict) -> SimpleNamespace:
    """Minimal app exposing app.state.MODELS[<id>].info.meta = meta."""
    return SimpleNamespace(state=SimpleNamespace(MODELS={"m": {"info": {"meta": meta}}}))


@pytest.mark.asyncio
async def test_null_capabilities_does_not_crash(automations_module):
    """capabilities=None + a default feature id must NOT raise (None.get)."""
    mod = automations_module
    app = _app_with_model_meta({"defaultFeatureIds": ["web_search"], "capabilities": None})
    with patch.object(mod.Config, "get", AsyncMock(return_value=True)):
        features = await mod._resolve_model_features(app, "m")
    assert features == {}, f"null capabilities should yield no features, got {features!r}"


@pytest.mark.asyncio
async def test_missing_capabilities_key_does_not_crash(automations_module):
    """The key absent entirely (the case `, {}` already handled) must also be fine."""
    mod = automations_module
    app = _app_with_model_meta({"defaultFeatureIds": ["web_search"]})
    with patch.object(mod.Config, "get", AsyncMock(return_value=True)):
        features = await mod._resolve_model_features(app, "m")
    assert features == {}


@pytest.mark.asyncio
async def test_enabled_capability_resolves_feature(automations_module):
    """Sanity (the positive path): a real capability + admin-enabled feature +
    default id resolves the feature to True."""
    mod = automations_module
    app = _app_with_model_meta(
        {"defaultFeatureIds": ["web_search"], "capabilities": {"web_search": True}}
    )
    with patch.object(mod.Config, "get", AsyncMock(return_value=True)):
        features = await mod._resolve_model_features(app, "m")
    assert features.get("web_search") is True, features
