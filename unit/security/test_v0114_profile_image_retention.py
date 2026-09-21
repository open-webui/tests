"""Regression: a rejected profile image URL must not be retained anywhere.

open-webui 0.11.4 fix `38a8dc9f3` (PR #29971): ModelMeta's profile_image_url
validator kept a module-level set of every distinct rejected value so it could
log each one once. Any signed-in user could post model entries with invalid
profile image URLs, and FastAPI validates the body before the permission check,
so the strings piled up with no size cap: 8.9 MB retained after 2,000 distinct
4KB values, growing without bound. The set and the warning it served are gone;
the validator still clears the value exactly as before.

Discriminates: passes on dev 344ea5306, fails on `38a8dc9f3^` (the module-level
`_warned_profile_urls` set exists, grows by one entry per distinct rejected
value, and each first rejection logs a warning).
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.regression

INVALID_URL = "javascript:alert(1)"
INVALID_DATA_URI = "data:image/svg+xml;base64,AAAA"
VALID_URL = "https://example.com/avatar.png"


@pytest.fixture(scope="session")
def models_module(owui_module):
    """`open_webui.models.models` (ModelMeta)."""
    return owui_module("open_webui.models.models")


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


def test_the_module_keeps_no_set_of_rejected_urls(models_module):
    assert not hasattr(models_module, "_warned_profile_urls"), (
        "a module-level container of rejected profile image URLs is back; each "
        "distinct value a signed-in user posts is retained for the process lifetime"
    )


def test_distinct_rejections_leave_no_trace(models_module):
    """Validate many distinct invalid values; no set anywhere retains them."""
    import gc

    marker = "javascript:alert(1)-retention-marker-"
    values = [f"{marker}{index}-{'x' * 512}" for index in range(200)]

    for value in values:
        models_module.ModelMeta(profile_image_url=value)

    retained_sets = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, set) and any(isinstance(item, str) and item in values for item in obj)
    ]
    assert retained_sets == [], (
        "rejected profile image URLs are being accumulated in a long-lived set; any "
        "signed-in user can grow process memory without bound this way"
    )


def test_no_warning_is_logged_per_rejection(models_module, caplog):
    caplog.set_level(logging.WARNING, logger="open_webui.models.models")

    for index in range(50):
        models_module.ModelMeta(profile_image_url=f"{INVALID_URL}-{index}")

    warnings = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert warnings == [], "a rejected profile image URL is still being logged"


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


def test_rejection_still_clears_the_value(models_module):
    meta = models_module.ModelMeta(profile_image_url=INVALID_URL)

    assert meta.profile_image_url is None, "the fix must keep clearing invalid URLs"


def test_an_invalid_svg_data_uri_is_still_rejected(models_module):
    meta = models_module.ModelMeta(profile_image_url=INVALID_DATA_URI)

    assert meta.profile_image_url is None


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct
# ---------------------------------------------------------------------------


def test_a_valid_url_is_preserved(models_module):
    meta = models_module.ModelMeta(profile_image_url=VALID_URL)

    assert meta.profile_image_url == VALID_URL


def test_an_unset_profile_image_stays_unset(models_module):
    meta = models_module.ModelMeta(profile_image_url=None)

    assert meta.profile_image_url is None


def test_an_invalid_background_image_still_raises(models_module):
    """Background images have no silent-clear fallback; that must not change."""
    if "background_image_url" not in models_module.ModelMeta.model_fields:
        pytest.skip("background_image_url does not exist on this ref")
    with pytest.raises(ValueError):
        models_module.ModelMeta(background_image_url=INVALID_URL)


def test_a_valid_background_image_is_preserved(models_module):
    if "background_image_url" not in models_module.ModelMeta.model_fields:
        pytest.skip("background_image_url does not exist on this ref")
    meta = models_module.ModelMeta(
        background_image_url="/api/v1/files/12345678-1234-1234-1234-123456789012/content"
    )

    assert (
        meta.background_image_url
        == "/api/v1/files/12345678-1234-1234-1234-123456789012/content"
    )
