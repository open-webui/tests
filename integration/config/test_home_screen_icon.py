"""Regression: the iOS home screen icon showed a thick black ring around the logo (#34762).

The new icons in open-webui 0.13.0 drew the logo on a rounded tile with a transparent margin, and
the apple-touch-icon got the same file. iOS keeps no transparency in home screen icons: it paints
transparent pixels black and rounds the square itself, so the margin showed as a black ring. The
fix ships a fully opaque square icon.

The test follows the icon the way Safari does: the page names it in its apple-touch-icon link
(the served page when the instance has a built frontend, else the page template it is built
from), and the instance serves the file at that address.

Discriminates: passes on fix/ios-pwa-icon-34762 661ded33e, fails on dev d3e87f8b1 (13121 of
32400 pixels are transparent).
"""

from __future__ import annotations

import io
import re

import pytest
from PIL import Image

from harness.instance import resolve_backend

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

APPLE_TOUCH_ICON_LINK = re.compile(r"<link\b[^>]*\brel=\"apple-touch-icon\"[^>]*>", re.S)


def _advertised_icon(instance) -> tuple[str, str]:
    """The apple-touch-icon link's href and sizes from the page Safari loads."""
    if instance.serves_frontend:
        with instance.client() as client:
            page = client.get("/")
        assert page.status_code == 200, f"the app page answered HTTP {page.status_code}"
        html = page.text
    else:
        html = (resolve_backend().parent / "src" / "app.html").read_text()
    link = APPLE_TOUCH_ICON_LINK.search(html)
    assert link, "the app page names no apple-touch-icon, so iOS falls back to a screenshot"
    href = re.search(r"\bhref=\"([^\"]+)\"", link.group(0)).group(1)
    sizes = re.search(r"\bsizes=\"([^\"]+)\"", link.group(0))
    return href, sizes.group(1) if sizes else ""


def test_the_ios_home_screen_icon_has_no_transparent_pixels(instance):
    href, sizes = _advertised_icon(instance)
    with instance.client() as client:
        served = client.get(href)
    assert served.status_code == 200, f"{href}: HTTP {served.status_code}"

    icon = Image.open(io.BytesIO(served.content)).convert("RGBA")
    alpha = icon.getchannel("A")
    transparent = sum(alpha.histogram()[:255])

    assert icon.width == icon.height, f"{href} is {icon.size}, iOS expects a square"
    if sizes:
        assert f"{icon.width}x{icon.height}" == sizes, (
            f"{href} is {icon.size}, the link says {sizes}"
        )
    assert transparent == 0, (
        f"{transparent} of {icon.width * icon.height} pixels of {href} are transparent; iOS paints "
        "them black, so the home screen icon shows a black ring (#34762)"
    )
