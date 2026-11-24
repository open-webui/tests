"""The full-screen image preview on a phone, driven by touch.

`Fingers` puts touch points on the screen through CDP, in screen pixels converted with the page
zoom at each event, as a real finger lands: `pinch`, `tap` and `double_tap`. `open_image_reply`
opens a phone page on a chat whose reply shows a grid image, and `open_preview` taps it open and
returns the box of the image as the preview fits it to the screen. `screen_box` reads an element's
box in screen pixels, and `expect_on_screen_and_zoom` checks the image against that fitted box.
"""

from __future__ import annotations

import base64
import struct
import zlib

from playwright.sync_api import Locator, Page, expect

from harness.chat_history import seed_chat
from utils.phone import PHONE

ALT = "Sun over a grid"
SCREEN_MIDDLE = (195, 420)

# the element's box in screen pixels, read through the page zoom
ON_SCREEN_BOX = """(element) => {
    const viewport = window.visualViewport;
    const box = element.getBoundingClientRect();
    const toScreen = (x, y) => [(x - viewport.offsetLeft) * viewport.scale,
                                (y - viewport.offsetTop) * viewport.scale];
    const [left, top] = toScreen(box.left, box.top);
    const [right, bottom] = toScreen(box.right, box.bottom);
    return { left, top, right, bottom, width: right - left, height: bottom - top,
             screenWidth: viewport.width * viewport.scale,
             screenHeight: viewport.height * viewport.scale };
}"""


def grid_png(fill: bytes, size: int = 512) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = []
    for y in range(size):
        row = bytearray(b"\x00")
        for x in range(size):
            row += b"\xff\xff\xff" if x % 32 < 2 or y % 32 < 2 else fill
        rows.append(bytes(row))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


DATA_URI = data_uri(grid_png(b"\x28\x5a\xa0"))
# off-white, as many generated images are where the preview's buttons sit
LIGHT_DATA_URI = data_uri(grid_png(b"\xf5\xf5\xf0"))


class Fingers:
    """Touch points on the screen, in screen pixels."""

    def __init__(self, page: Page):
        self.page = page
        self.session = page.context.new_cdp_session(page)

    def _send(self, kind: str, points: list[tuple[int, float, float]]) -> None:
        scale = self.page.evaluate("visualViewport.scale")
        touch_points = [{"id": id, "x": x / scale, "y": y / scale} for id, x, y in points]
        self.session.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": touch_points})

    def pinch(
        self, centre: tuple[float, float], start: float, end: float, moves: int, pause_ms: int
    ):
        """Two fingers side by side on `centre`, from `start` to `end` pixels apart."""
        x, y = centre

        def fingers(apart: float) -> list[tuple[int, float, float]]:
            return [(1, x - apart / 2, y), (2, x + apart / 2, y)]

        self._send("touchStart", fingers(start)[:1])
        self._send("touchStart", fingers(start))
        for move in range(1, moves + 1):
            self._send("touchMove", fingers(start + (end - start) * move / moves))
            self.page.wait_for_timeout(pause_ms)
        self._send("touchEnd", fingers(end)[1:])
        self._send("touchEnd", [])
        self.page.wait_for_timeout(500)

    def spread_off(
        self, moving: tuple[float, float], resting: tuple[float, float], by: tuple[float, float]
    ):
        """One finger rests on `resting` while the one on `moving` slides away by `by`."""
        (x, y), (dx, dy) = moving, by
        self._send("touchStart", [(1, x, y)])
        self._send("touchStart", [(1, x, y), (2, *resting)])
        for move in range(1, 16):
            self._send("touchMove", [(1, x + dx * move / 15, y + dy * move / 15), (2, *resting)])
            self.page.wait_for_timeout(16)
        self._send("touchEnd", [])
        self.page.wait_for_timeout(500)

    def tap(self, point: tuple[float, float]) -> None:
        self._send("touchStart", [(1, *point)])
        self.page.wait_for_timeout(40)
        self._send("touchEnd", [])

    def double_tap(self, point: tuple[float, float]) -> None:
        self.tap(point)
        self.page.wait_for_timeout(120)
        self.tap(point)
        self.page.wait_for_timeout(700)


def open_image_reply(page_for, make_user, source: str = DATA_URI) -> Page:
    """A phone page on a new chat whose reply shows the image `source`."""
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Draw a sun over a grid"},
                {"role": "assistant", "content": f"Here it is.\n\n![{ALT}]({source})"},
            ],
        )
    page = page_for(owner, path=f"/c/{chat_id}", **PHONE)
    image = page.locator("#messages-container").get_by_alt_text(ALT)
    expect(image).to_be_visible()
    page.wait_for_function(
        "image => image.complete && image.naturalWidth > 0", arg=image.element_handle()
    )
    return page


def screen_box(locator: Locator) -> dict:
    return locator.evaluate(ON_SCREEN_BOX)


def centre(box: dict) -> tuple[float, float]:
    return ((box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2)


def preview(page: Page) -> Locator:
    return page.get_by_role("dialog", name=ALT)


def open_preview(page: Page, fingers: Fingers) -> dict:
    """Taps the reply's image and returns the box of the image the preview fits to the screen."""
    fingers.tap(centre(screen_box(page.locator("#messages-container").get_by_alt_text(ALT))))
    image = preview(page).get_by_alt_text(ALT)
    expect(image).to_be_visible()
    page.wait_for_timeout(500)
    return screen_box(image)


def zoom_of(page: Page, fitted: dict) -> float:
    return screen_box(preview(page).get_by_alt_text(ALT))["width"] / fitted["width"]


def expect_on_screen_and_zoom(page: Page, fitted: dict, low: float, high: float, step: str):
    box = screen_box(preview(page).get_by_alt_text(ALT))
    zoom = box["width"] / fitted["width"]
    middle = (box["screenWidth"] / 2, box["screenHeight"] / 2)
    covers_middle = (
        box["left"] <= middle[0] <= box["right"] and box["top"] <= middle[1] <= box["bottom"]
    )
    assert covers_middle, f"after {step} the image left the middle of the screen: {box}"
    assert low <= zoom <= high, f"after {step} the image is at {zoom:.3g} of its fitted size"
