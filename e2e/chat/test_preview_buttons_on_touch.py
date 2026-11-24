"""Regression: on a touch screen, buttons over a zoomable preview do nothing when tapped.

open-webui/open-webui#34942. On a 390 by 844 touch screen a tap on Download in the full-screen
image preview saves the image, and Download as SVG and Copy to clipboard on a Mermaid diagram save
the SVG and copy its code. After zooming into an image, Close and Download stay on top, can be
seen against the image and still work: the person who reported open-webui/open-webui#34939 on an
iPhone lost them over a light image. Taps are Chromium touch points sent through CDP
(`utils/image_preview.py`); WebKit's iPhone emulation gave the same results by hand.

A pinch whose second finger lands on a button still zooms the image.

Discriminates: the three button tests are red on dev 34888d690. Download in the preview and
Download as SVG start no download, and after zooming into a light image the Close icon shows at
1.09:1 against it. The pinch test passes on dev and guards the fix: a first version of it
(7a1d5e981) left that pinch dead, with "zoom requires valid numbers" on every move. All four pass
on fix/preview-buttons-work-on-touch.
"""

from __future__ import annotations

import io
import math

import pytest
from PIL import Image
from playwright.sync_api import Locator, expect

from harness.chat_history import seed_chat
from utils.clipboard import read_text
from utils.image_preview import (
    SCREEN_MIDDLE,
    Fingers,
    centre,
    data_uri,
    grid_png,
    open_image_reply,
    open_preview,
    preview,
    screen_box,
    zoom_of,
)
from utils.phone import PHONE

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

# off-white, as many generated images are where the preview's buttons sit
LIGHT_DATA_URI = data_uri(grid_png(b"\xf5\xf5\xf0"))
DIAGRAM = "graph TD\n  A[Start] --> B[Middle]\n  B --> C[End]"


def tap_button(fingers: Fingers, button: Locator) -> None:
    expect(button).to_be_visible()
    fingers.tap(centre(screen_box(button)))


def icon_contrast(button: Locator) -> float:
    """Contrast between the button's brightest pixel (its white icon) and what most of the button
    shows behind it, as WCAG measures it."""

    def luminance(pixel: tuple[int, int, int]) -> float:
        def linear(value: int) -> float:
            value /= 255
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

        red, green, blue = pixel[:3]
        return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)

    shot = Image.open(io.BytesIO(button.screenshot())).convert("RGB")
    levels = sorted(luminance(pixel) for pixel in shot.getdata())
    icon, behind = levels[-1], levels[len(levels) // 2]
    return (max(icon, behind) + 0.05) / (min(icon, behind) + 0.05)


def test_close_and_download_stay_usable_after_zooming_in(page_for, make_user):
    page = open_image_reply(page_for, make_user, source=LIGHT_DATA_URI)
    fingers = Fingers(page)
    fitted = open_preview(page, fingers)

    fingers.pinch(SCREEN_MIDDLE, 100, 250, moves=15, pause_ms=30)
    assert zoom_of(page, fitted) > 1.5, "a slow spread did not zoom the image in"
    for name in ("Close", "Download"):
        button = preview(page).get_by_role("button", name=name)
        hit = button.evaluate(
            "(button, [x, y]) => button.contains(document.elementFromPoint(x, y))",
            list(centre(screen_box(button))),
        )
        assert hit, f"after zooming in, something covers the {name} button"
        contrast = icon_contrast(button)
        # 3:1 is the WCAG minimum for icons
        assert contrast >= 3, f"after zooming in, the {name} icon shows at {contrast:.2f}:1"

    with page.expect_download() as downloaded:
        tap_button(fingers, preview(page).get_by_role("button", name="Download"))
    assert downloaded.value.suggested_filename.endswith(".png")
    tap_button(fingers, preview(page).get_by_role("button", name="Close"))
    expect(preview(page)).to_have_count(0)


def test_download_in_the_image_preview_saves_the_image_on_a_tap(page_for, make_user):
    page = open_image_reply(page_for, make_user)
    fingers = Fingers(page)
    fitted = open_preview(page, fingers)
    download = preview(page).get_by_role("button", name="Download")

    with page.expect_download() as downloaded:
        tap_button(fingers, download)
    assert downloaded.value.suggested_filename.endswith(".png")
    expect(preview(page)).to_be_visible()

    # a quick second tap on the button is two taps on it, never a double tap on the image
    with page.expect_download():
        fingers.double_tap(centre(screen_box(download)))
    assert zoom_of(page, fitted) == 1, "a double tap on Download zoomed the image"


def test_a_pinch_with_one_finger_on_a_button_still_zooms(page_for, make_user):
    page = open_image_reply(page_for, make_user)
    fingers = Fingers(page)
    fitted = open_preview(page, fingers)
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    download = centre(screen_box(preview(page).get_by_role("button", name="Download")))

    # first finger on the image, second lands on Download, then the first slides away
    start, by = (195, 500), (-60, 180)
    fingers.spread_off(start, download, by=by)
    # the image grows as far as the fingers spread apart (cf100751e)
    spread = math.dist((start[0] + by[0], start[1] + by[1]), download) / math.dist(start, download)
    zoom = zoom_of(page, fitted)
    assert zoom == pytest.approx(spread, rel=0.1), (
        f"a pinch with a finger on Download zoomed to {zoom:.2f}, the fingers spread {spread:.2f}"
    )
    assert errors == [], f"the pinch raised page errors: {errors[:2]}"


def test_a_diagrams_download_and_copy_buttons_take_a_tap(page_for, make_user):
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Draw a flow"},
                {"role": "assistant", "content": f"```mermaid\n{DIAGRAM}\n```"},
            ],
        )
    page = page_for(
        owner, path=f"/c/{chat_id}", permissions=["clipboard-read", "clipboard-write"], **PHONE
    )
    fingers = Fingers(page)
    reply = page.locator("#messages-container")
    expect(reply.get_by_text("Middle", exact=True)).to_be_visible()

    with page.expect_download() as downloaded:
        tap_button(fingers, reply.get_by_role("button", name="Download as SVG"))
    assert downloaded.value.suggested_filename.endswith(".svg")

    tap_button(fingers, reply.get_by_role("button", name="Copy to clipboard"))
    expect(page.get_by_text("Copied to clipboard").first).to_be_visible()
    assert read_text(page) == DIAGRAM

    node = reply.get_by_text("Middle", exact=True)
    before = node.bounding_box()
    fingers.double_tap(centre(screen_box(reply.get_by_role("button", name="Copy to clipboard"))))
    assert node.bounding_box() == before, "a double tap on Copy to clipboard zoomed the diagram"
