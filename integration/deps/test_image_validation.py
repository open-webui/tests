"""Dependency smoke: Pillow reads the images users hand Open WebUI, wherever it decides on them.

Saving a model whose `meta.background_image_url` points at an uploaded file runs
`validate_background_image` in `utils/validate.py`: Pillow opens the bytes, names the format,
checks the size, verifies and fully decodes them, and the file is typed after what Pillow read.
PNG, JPEG, WebP and GIF are accepted whatever name or type the upload claimed; a BMP, a file that
only claims to be a PNG, a PNG whose pixel data stops short (which only a full decode finds), a
picture over 25 megapixels and a decompression bomb Pillow will not open are refused with 400.

Attaching a link (`/api/v1/retrieval/process/url`) opens a fetched body with Pillow too: an
answer claiming to be an image that Pillow cannot read is refused, and a JPEG served as
`application/octet-stream` without an extension is stored as the JPEG it is. Before an image
edit goes to an OpenAI-compatible engine, a JPEG turned by its EXIF orientation is turned
upright and saved again as JPEG, and a plain PNG is sent as it was.

Twin of unit/deps/test_pillow.py.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose `Image.open` fails on every
input every image that should pass fails (the backgrounds, the linked JPEG, both edits) and the
BMP and the large picture are refused for the wrong reason, while the garbage cases stay green.
Mapping WEBP out of the background formats fails its case, dropping the `load()` after `verify()`
lets the short PNG through, dropping the pixel check lets the large picture through, no longer
catching `DecompressionBombError` answers the bomb with a 500, a fetch that ignores what Pillow
read stores the linked JPEG as a file, and an edit that skips `exif_transpose` sends the sideways
picture as it was.
"""

from __future__ import annotations

import base64
import email.parser
import email.policy
import io
import struct
import uuid
import zlib

import httpx
import pytest

from harness.image_engines import (
    IMAGES_CONFIG,
    PNG_BASE64,
    image_connection,
    image_settings,
    save_image_settings,
)
from harness.listener import json_answer
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import LOCAL_WEB_FETCH

Image = pytest.importorskip("PIL.Image")

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

ORIENTATION = 274  # the EXIF tag
TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}


def _picture(format: str, size: tuple[int, int] = (64, 32), **options) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (20, 90, 160)).save(buffer, format=format, **options)
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


def _bomb() -> bytes:
    """200 megapixels in 24 KB: past the ceiling where Pillow refuses to open an image."""
    buffer = io.BytesIO()
    Image.new("1", (20000, 10000)).save(buffer, format="PNG")
    return buffer.getvalue()


def _uploaded(client: httpx.Client, content: bytes) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "false"},
        files={"file": ("background.png", content, "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return f"/api/v1/files/{uploaded.json()['id']}/content"


def _save_model_with_background(client: httpx.Client, background_url: str) -> httpx.Response:
    model_id = f"background-{uuid.uuid4().hex[:8]}"
    saved = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": "Background model",
            "meta": {"background_image_url": background_url},
            "params": {},
        },
    )
    client.post("/api/v1/models/model/delete", json={"id": model_id})
    return saved


@pytest.mark.parametrize("format", TYPES)
def test_each_background_format_is_accepted_and_typed_as_what_it_is(admin, format):
    with admin.client() as client:
        background_url = _uploaded(client, _picture(format))
        saved = _save_model_with_background(client, background_url)
        stored = client.get(background_url.removesuffix("/content"))

    assert saved.status_code == 200, saved.text
    assert stored.json()["meta"]["content_type"] == TYPES[format]


@pytest.mark.parametrize(
    ("content", "refusal"),
    [
        pytest.param(_picture("BMP"), "must be PNG, JPEG, WebP, or GIF", id="bmp"),
        pytest.param(
            b"\x89PNG\r\n\x1a\n" + b"not really an image" * 8, "Invalid background", id="garbage"
        ),
        pytest.param(_png_short_of_pixels(), "Invalid background", id="short-of-pixels"),
        pytest.param(_picture("PNG", (6000, 5000)), "at most 25 megapixels", id="30-megapixels"),
        pytest.param(_bomb(), "Invalid background", id="decompression-bomb"),
    ],
)
def test_a_background_pillow_cannot_accept_is_refused(admin, content, refusal):
    with admin.client() as client:
        saved = _save_model_with_background(client, _uploaded(client, content))

    assert saved.status_code == 400, saved.text
    assert refusal in saved.text


# ---------------------------------------------------------------- a linked image


@pytest.fixture(scope="module")
def local_fetch(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


def _attach_link(instance, url: str) -> httpx.Response:
    with instance.client() as client:
        return client.post("/api/v1/retrieval/process/url", json={"url": url})


@pytest.mark.slow
def test_a_linked_jpeg_is_stored_as_one_whatever_the_server_calls_it(local_fetch, listener):
    served = (200, {"Content-Type": "application/octet-stream"}, _picture("JPEG"))
    listener.route("GET", "/photo", served)

    attached = _attach_link(local_fetch, f"{listener.base_url}/photo")

    assert attached.status_code == 200, attached.text
    assert attached.json()["type"] == "image"
    assert attached.json()["name"] == "photo.jpg"
    assert attached.json()["file"]["meta"]["content_type"] == "image/jpeg"


@pytest.mark.slow
def test_a_link_claiming_an_image_pillow_cannot_read_is_refused(local_fetch, listener):
    listener.route("GET", "/photo.png", (200, {"Content-Type": "image/png"}, b"<html>no</html>"))

    attached = _attach_link(local_fetch, f"{listener.base_url}/photo.png")

    assert attached.status_code == 400, attached.text
    assert "Invalid image content" in attached.text


# ---------------------------------------------------------------- image edits


@pytest.fixture
def edit_engine(admin, preserve, listener):
    preserve(IMAGES_CONFIG)
    listener.route("POST", "/images/edits", json_answer({"data": [{"b64_json": PNG_BASE64}]}))
    with admin.client() as client:
        editor = image_connection(
            "editor",
            "openai",
            listener.base_url,
            "edit-key",
            models=("gpt-image-1",),
            generates=False,
        )
        save_image_settings(client, **image_settings(editor))
    return listener


def _edit(actor, content: bytes, content_type: str) -> httpx.Response:
    data_url = f"data:{content_type};base64,{base64.b64encode(content).decode()}"
    with actor.client() as client:
        return client.post("/api/v1/images/edit", json={"image": data_url, "prompt": "brighter"})


def _picture_sent(edit_engine) -> tuple[str, bytes]:
    """(content type, bytes) of the image part of the one edit request."""
    [call] = edit_engine.requests_to("/images/edits")
    head = f"Content-Type: {call.headers['Content-Type']}\r\n\r\n".encode()
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + call.body)
    [part] = [
        part
        for part in message.iter_parts()
        if part.get_param("name", header="content-disposition") == "image"
    ]
    return part.get_content_type(), part.get_payload(decode=True)


def test_a_sideways_jpeg_is_sent_upright_for_an_edit(edit_engine, make_user):
    image = Image.new("RGB", (80, 40), (200, 30, 30))
    exif = image.getexif()
    exif[ORIENTATION] = 6  # shown turned a quarter clockwise
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    edited = _edit(make_user(), buffer.getvalue(), "image/jpeg")

    assert edited.status_code == 200, edited.text
    content_type, sent = _picture_sent(edit_engine)
    with Image.open(io.BytesIO(sent)) as upright:
        assert upright.format == "JPEG" and content_type == "image/jpeg"
        assert upright.size == (40, 80)
        assert upright.getexif().get(ORIENTATION) in (None, 1)


def test_a_png_is_sent_for_an_edit_as_it_was(edit_engine, make_user):
    original = _picture("PNG")

    edited = _edit(make_user(), original, "image/png")

    assert edited.status_code == 200, edited.text
    assert _picture_sent(edit_engine) == ("image/png", original)
