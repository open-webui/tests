"""Regression: a config value the JSON column cannot serialize made `Config.upsert` raise.

open-webui 0.10.2 fix `ab22fe64b` (#26431): `Config.upsert` and `Config.seed_defaults` wrote
values straight into the JSON column, so a list of Pydantic models (`WEBUI_BANNERS`) or any other
non-JSON value raised at commit. The fix passes every value through `jsonable_encoder` first.

The integration twin boots an instance with banners in the environment, which is the seeding half.
Every route dumps its models before it calls `upsert`, so no request reaches that half with a raw
value; it is driven here against the scratch database, under a key of its own that is deleted after.

Discriminates: passes on bbfa876af, fails with the encoding dropped from `Config.upsert` (the JSON
column's serializer rejects the model and the datetime).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.regression


class Banner(BaseModel):
    id: str
    content: str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, stored",
    [
        ([Banner(id="b1", content="hello")], [{"id": "b1", "content": "hello"}]),
        (datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc), "2026-07-01T12:00:00+00:00"),
    ],
    ids=["pydantic-models", "datetime"],
)
async def test_a_value_the_json_column_cannot_take_is_stored_encoded(
    config_model_module, value, stored
):
    config = config_model_module.Config
    key = f"_regression.{uuid.uuid4().hex[:8]}"
    try:
        await config.upsert({key: value})
        assert await config.get(key) == stored
    finally:
        await config.delete(key)
