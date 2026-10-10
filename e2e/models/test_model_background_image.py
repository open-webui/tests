"""Journey: a model's background image chosen in the editor is checked by Pillow when it is saved.

The model editor checks a chosen background in the browser (type by name, size, a decode) and
uploads it on save; the server then opens it with Pillow, refuses what Pillow cannot fully decode
and types the file after what Pillow read. A PNG whose pixel data stops short decodes in Chrome
but not in Pillow, so saving it shows "Invalid background image." and keeps the model without a
background. A JPEG named `.png` saves, and its file is typed as the JPEG it is.

Twin of integration/deps/test_image_validation.py, which drives the same checks over HTTP.

Retargeted for d4879a98b, whose editor picks the background from the header (Add background,
then Change background): passes on dev 0401b7522 (3 of 3), and the refusal test fails in a
backend copy of it without the `load()` after `verify()`.

Discriminates: passes on dev ef67cc3fa; in a backend copy without the `load()` after `verify()`
the short PNG is saved and the refusal test fails, and in one whose `Image.open` fails on every
input the JPEG is refused.
"""

from __future__ import annotations

import io
import re
import struct
import uuid
import zlib

import pytest
from playwright.sync_api import expect

from harness.upstream import MOCK_MODEL_ID

Image = pytest.importorskip("PIL.Image")

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _picture(format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (20, 90, 160)).save(buffer, format=format)
    return buffer.getvalue()


def _png_short_of_pixels() -> bytes:
    """A well-formed PNG, every checksum right, whose pixel data ends ten bytes in."""
    png = _picture("PNG")
    rebuilt, position = bytearray(png[:8]), 8
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        kind = png[position + 4 : position + 8]
        data = png[position + 8 : position + 8 + length]
        if kind == b"IDAT":
            data = zlib.compress(b"\x00" * 10)
        rebuilt += struct.pack(">I", len(data)) + kind + data
        rebuilt += struct.pack(">I", zlib.crc32(kind + data))
        position += 12 + length
    return bytes(rebuilt)


@pytest.fixture
def model_id(admin):
    created_id = f"background-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": created_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": "Harbour guide",
                "meta": {},
                "params": {},
            },
        )
        assert created.status_code == 200, created.text
        yield created_id
        client.post("/api/v1/models/model/delete", json={"id": created_id})


def _choose_background_and_save(page, model_id: str, content: bytes) -> None:
    page.goto(f"/workspace/models/edit?id={model_id}")
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Add background").click()
    chooser.value.set_files({"name": "harbour.png", "mimeType": "image/png", "buffer": content})
    expect(page.get_by_role("button", name="Change background")).to_be_visible()
    page.get_by_role("button", name="Save & Update").click()


def _stored_model(admin, model_id: str) -> dict:
    with admin.client() as client:
        return client.get("/api/v1/models/model", params={"id": model_id}).json()


def test_a_background_pillow_cannot_decode_is_refused_on_save(page_for, admin, model_id):
    page = page_for(admin)

    _choose_background_and_save(page, model_id, _png_short_of_pixels())

    expect(page.get_by_text("Invalid background image.")).to_be_visible()
    assert not _stored_model(admin, model_id)["meta"].get("background_image_url")


def test_a_jpeg_named_png_is_saved_and_typed_as_a_jpeg(page_for, admin, model_id):
    page = page_for(admin)

    _choose_background_and_save(page, model_id, _picture("JPEG"))

    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))
    background_url = _stored_model(admin, model_id)["meta"]["background_image_url"]
    with admin.client() as client:
        stored_file = client.get(background_url.removesuffix("/content")).json()
    assert stored_file["meta"]["content_type"] == "image/jpeg"
