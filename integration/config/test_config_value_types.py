"""Regression: banners set in the environment stopped the instance from starting.

open-webui 0.10.2, fix `ab22fe64b` (#26431, "Setting WEBUI_BANNERS causes a startup failure"):
`WEBUI_BANNERS` is parsed into a list of `BannerModel`s and seeded into the config store at
startup, and `Config.seed_defaults` wrote each value straight into the JSON column, whose
serializer rejects Pydantic models, so the boot died. The fix passes every value through
`jsonable_encoder` before storing it.

Twin of unit/config/test_config_value_types.py; the same encoding on `Config.upsert`, which no
route reaches with a raw model, stays there.

Discriminates: passes on bbfa876af, fails with the encoding dropped from `Config.seed_defaults`
(the instance exits during boot).
"""

from __future__ import annotations

import json

import pytest

from harness.actors import create_user

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

BANNER = {
    "id": "maintenance",
    "type": "info",
    "title": "Maintenance",
    "content": "Down for maintenance tonight",
    "dismissible": True,
    "timestamp": 1767225600,
}


def test_banners_from_the_environment_boot_and_reach_every_user(instance_with):
    launched = instance_with({"WEBUI_BANNERS": json.dumps([BANNER])})

    with create_user(launched).client() as client:
        listed = client.get("/api/v1/configs/banners")

    assert listed.status_code == 200, listed.text
    assert [{key: banner[key] for key in BANNER} for banner in listed.json()] == [BANNER]
