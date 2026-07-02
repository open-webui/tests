"""Regression: admin config values of any data type persist without crashing.

open-webui 0.10.2 fix `ab22fe64b` (#26431): `Config.upsert` wrote values straight
into the JSON column, so a value that isn't directly JSON-serializable — e.g.
`WEBUI_BANNERS` as a list of Pydantic banner models — raised at commit and could
fail startup ("Setting WEBUI_BANNERS causes a startup failure"). The fix passes
every value through `fastapi.encoders.jsonable_encoder` before storing.

Uses the real `Config.upsert` / `Config.get` against the test sqlite DB.

Discriminates: passes on v0.10.2 (encoded), raises on v0.10.1 (raw value → the
JSON column's json.dumps rejects the Pydantic models).
"""

import pytest

pytestmark = pytest.mark.regression


@pytest.mark.asyncio
async def test_upsert_encodes_pydantic_typed_values(config_model_module):
    from pydantic import BaseModel

    Config = config_model_module.Config

    class Banner(BaseModel):
        id: str
        type: str
        content: str

    banners = [Banner(id="b1", type="info", content="hello")]
    # A list of Pydantic models is not directly JSON-serializable; storing it
    # must not raise (jsonable_encoder coerces it) and it round-trips as dicts.
    await Config.upsert({"_regression.banners": banners})
    stored = await Config.get("_regression.banners")
    assert stored == [{"id": "b1", "type": "info", "content": "hello"}], stored


@pytest.mark.asyncio
async def test_upsert_encodes_datetime_value(config_model_module):
    """A non-JSON scalar (datetime) must also store rather than crash — the same
    jsonable_encoder path. jsonable_encoder renders it as an ISO-8601 string."""
    from datetime import datetime, timezone

    Config = config_model_module.Config
    dt = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    await Config.upsert({"_regression.ts": dt})
    stored = await Config.get("_regression.ts")
    assert isinstance(stored, str) and stored.startswith("2026-07-01"), stored
