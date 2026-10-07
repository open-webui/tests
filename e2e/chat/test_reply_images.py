"""Journey: images in an assistant reply are shown when they load and replaced when they do not.

A markdown image in a reply, whether its source is a data URI or an http URL, shows as an image
that loaded, with its alt text. An image that fails to load (a 404 or data that is not an image)
is replaced by an "Image unavailable" placeholder while the rest of the reply still renders.
Clicking a shown image opens a full-screen preview that Escape closes. A source with a scheme
that is neither http(s), data nor a relative path is not loaded: the image falls back to the
app's own placeholder. The same images are shown again after a reload of the chat.

Discriminates: passes on dev ebc6add67; in a frontend build where the image failure handler does
nothing, every source is passed through unchecked and Escape does not close the preview, seven
tests go red (both placeholder tests, the preview test, the three scheme cases and the reload
test) and the data URI and http image tests stay green.
"""

from __future__ import annotations

import base64
import struct
import zlib

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LOADED = "image => image.complete && image.naturalWidth > 0"


def tiny_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\x00" + b"\x20\x80\xe0" * 2 for _ in range(2))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


PNG = tiny_png()
DATA_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()
NOT_AN_IMAGE = "data:text/plain;base64," + base64.b64encode(b"just words").decode()


def ask_for(page: Page, upstream, answer: str) -> Locator:
    upstream.queue(reply.text(answer, match=reply.answering("show me a picture")))
    send(page, "show me a picture")
    expect_reply(page, "That is all.")
    return last_reply(page)


def serve_png(listener, path: str = "/tide.png") -> str:
    listener.route("GET", path, (200, {"Content-Type": "image/png"}, PNG))
    return f"{listener.base_url}{path}"


def expect_loaded(image: Locator) -> None:
    expect(image).to_be_visible()
    image.page.wait_for_function(LOADED, arg=image.element_handle())


def test_a_data_uri_image_is_shown_with_its_alt_text(page_for, make_user, upstream):
    page = page_for(make_user())

    box = ask_for(page, upstream, f"Here it is.\n\n![Harbour chart]({DATA_URI})\n\nThat is all.")

    expect_loaded(box.get_by_alt_text("Harbour chart"))


def test_an_http_image_is_loaded_from_its_server(page_for, make_user, upstream, listener):
    url = serve_png(listener)
    page = page_for(make_user())

    box = ask_for(page, upstream, f"Here it is.\n\n![Tide chart]({url})\n\nThat is all.")

    expect_loaded(box.get_by_alt_text("Tide chart"))
    assert [request.method for request in listener.requests_to("/tide.png")] == ["GET"]


def test_an_image_the_server_cannot_find_shows_the_placeholder(
    page_for, make_user, upstream, listener
):
    listener.route("GET", "/gone.png", (404, {"Content-Type": "text/plain"}, b""))
    page = page_for(make_user())

    box = ask_for(
        page,
        upstream,
        f"Before.\n\n![Missing chart]({listener.base_url}/gone.png)\n\nThat is all.",
    )

    expect(box.get_by_text("Image unavailable")).to_be_visible()
    expect(box.get_by_alt_text("Missing chart")).to_have_count(0)
    expect(box.get_by_role("button", name="Show image preview")).to_have_count(0)
    expect(box).to_contain_text("Before.")
    assert listener.requests_to("/gone.png")


def test_a_data_uri_that_is_not_an_image_shows_the_placeholder(page_for, make_user, upstream):
    page = page_for(make_user())

    box = ask_for(page, upstream, f"Before.\n\n![Words]({NOT_AN_IMAGE})\n\nThat is all.")

    expect(box.get_by_text("Image unavailable")).to_be_visible()
    expect(box.get_by_alt_text("Words")).to_have_count(0)
    expect(box).to_contain_text("Before.")


def test_clicking_an_image_opens_a_preview_that_escape_closes(page_for, make_user, upstream):
    page = page_for(make_user())
    box = ask_for(page, upstream, f"Here it is.\n\n![Harbour chart]({DATA_URI})\n\nThat is all.")
    expect_loaded(box.get_by_alt_text("Harbour chart"))

    box.get_by_role("button", name="Show image preview").click()

    preview = page.locator("body > div.modal").get_by_alt_text("Harbour chart")
    expect(preview).to_be_visible()
    assert preview.get_attribute("src") == DATA_URI

    page.keyboard.press("Escape")

    expect(page.locator("body > div.modal")).to_have_count(0)
    expect(box.get_by_alt_text("Harbour chart")).to_be_visible()


@pytest.mark.parametrize(
    "source",
    ["javascript:alert(1)", "file:///etc/hostname", "ftp://example.invalid/chart.png"],
)
def test_an_image_with_another_scheme_falls_back_to_the_placeholder(
    page_for, make_user, upstream, source
):
    page = page_for(make_user())

    box = ask_for(page, upstream, f"Before.\n\n![Odd chart]({source})\n\nThat is all.")

    image = box.get_by_alt_text("Odd chart")
    expect(image).to_have_attribute("src", "/favicon.png")
    expect_loaded(image)


def test_the_images_are_shown_again_after_a_reload(page_for, make_user, listener):
    url = serve_png(listener)
    listener.route("GET", "/gone.png", (404, {"Content-Type": "text/plain"}, b""))
    account = make_user()
    answer = (
        f"![Harbour chart]({DATA_URI})\n\n![Tide chart]({url})\n\n"
        f"![Missing chart]({listener.base_url}/gone.png)\n\nThat is all."
    )
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "show me a picture"},
                {"role": "assistant", "content": answer},
            ],
        )
    page = page_for(account)
    page.goto(f"/c/{chat_id}")
    expect_reply(page, "That is all.")
    expect_loaded(last_reply(page).get_by_alt_text("Harbour chart"))

    page.reload()

    box = last_reply(page)
    expect_loaded(box.get_by_alt_text("Harbour chart"))
    expect_loaded(box.get_by_alt_text("Tide chart"))
    expect(box.get_by_text("Image unavailable")).to_have_count(1)
