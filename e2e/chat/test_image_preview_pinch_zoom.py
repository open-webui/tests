"""Regression: on a phone, pinch zoom in the image preview leaves the image lost or stuck.

open-webui/open-webui#34939. A person taps an image in a reply on a 390 by 844 touch screen and
zooms it in the full-screen preview with two fingers. The image follows the fingers: a quick pinch
together shrinks it to what the fingers did and never to nothing, a quick spread keeps it on the
screen with zoom still responding, and a slow pinch back to the starting finger distance brings it
back to the size it opened at. The fingers are Chromium touch points sent through CDP
(`utils/image_preview.py`); the same gestures were checked by hand in WebKit's iPhone emulation,
with the same results on both refs.

Discriminates: all three tests are red on dev 34888d690. A slow spread to 2.5 times the finger
distance zooms 20 times (every test's first spread), a quick pinch together sets the image's size
to zero, a quick spread throws it off the top left of the screen and zoom stops responding, and the
slow pinch back ends at about 0.6 of the fitted size. All three pass on fix/preview-pinch-zoom.
"""

from __future__ import annotations

import pytest

from utils.image_preview import (
    SCREEN_MIDDLE,
    Fingers,
    expect_on_screen_and_zoom,
    open_image_reply,
    open_preview,
    zoom_of,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def test_a_quick_pinch_together_never_shrinks_the_image_away(page_for, make_user):
    page = open_image_reply(page_for, make_user)
    fingers = Fingers(page)
    fitted = open_preview(page, fingers)

    fingers.pinch(SCREEN_MIDDLE, 100, 250, moves=15, pause_ms=30)
    expect_on_screen_and_zoom(page, fitted, 2, 3, "a slow spread to 2.5 times")
    # 250 to 60 px apart is a quarter of the distance: 2.5 times that is about 0.6
    fingers.pinch(SCREEN_MIDDLE, 250, 60, moves=2, pause_ms=4)
    expect_on_screen_and_zoom(page, fitted, 0.5, 0.7, "a quick pinch together")
    fingers.pinch(SCREEN_MIDDLE, 100, 200, moves=10, pause_ms=30)
    expect_on_screen_and_zoom(page, fitted, 1, 1.4, "a slow spread to twice")


def test_a_quick_spread_keeps_the_image_on_screen_and_zooming(page_for, make_user):
    page = open_image_reply(page_for, make_user)
    fingers = Fingers(page)
    fitted = open_preview(page, fingers)

    fingers.pinch(SCREEN_MIDDLE, 100, 330, moves=2, pause_ms=4)
    expect_on_screen_and_zoom(page, fitted, 2.6, 4, "a quick spread to 3.3 times")
    zoomed = zoom_of(page, fitted)
    fingers.pinch(SCREEN_MIDDLE, 100, 150, moves=10, pause_ms=30)
    expect_on_screen_and_zoom(
        page, fitted, zoomed * 1.3, zoomed * 1.7, "a slow spread to 1.5 times"
    )


def test_a_slow_pinch_back_returns_to_the_size_it_opened_at(page_for, make_user):
    page = open_image_reply(page_for, make_user)
    fingers = Fingers(page)
    fitted = open_preview(page, fingers)

    fingers.pinch(SCREEN_MIDDLE, 100, 250, moves=15, pause_ms=30)
    expect_on_screen_and_zoom(page, fitted, 2, 3, "a slow spread to 2.5 times")
    fingers.pinch(SCREEN_MIDDLE, 250, 100, moves=15, pause_ms=30)
    expect_on_screen_and_zoom(page, fitted, 0.97, 1.03, "the same pinch back")
