"""Regression: a generated JPEG or WebP image was stored, named and served as a PNG.

Fix 0a2e9a42e (open-webui/open-webui#30359, issue open-webui/open-webui#29948). An engine that
answers with bare base64 (Automatic1111, OpenAI's `b64_json`, Gemini's `bytesBase64Encoded` and
`inlineData`) carries no format, and Open WebUI labelled every such image `image/png` and named
it `generated-image.png`, so the download name and the served content type were wrong. The type
is now read from the image bytes. The engines here are local stand-ins answering with a JPEG or
a WebP image.

Discriminates: passes on dev efe63bd34, fails with 0a2e9a42e reverted (every JPEG and WebP
image is stored as `image/png` with a `.png` name).
"""

from __future__ import annotations

import base64
import io

import pytest

from harness import image_engines
from harness.image_engines import (
    GEMINI_IMAGE_MODEL,
    IMAGEN_MODEL,
    IMAGES_CONFIG,
    PNG_BASE64,
    image_connection,
    save_image_settings,
    serve_automatic1111,
    serve_gemini,
)
from harness.listener import json_answer

Image = pytest.importorskip("PIL.Image")

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

EXTENSIONS = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/png": ".png"}
PILLOW_FORMATS = {"image/jpeg": "JPEG", "image/webp": "WEBP", "image/png": "PNG"}


def encoded_image(content_type: str) -> tuple[bytes, str]:
    """A 1x1 picture in `content_type`: its bytes and their bare base64."""
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (200, 30, 30)).save(buffer, PILLOW_FORMATS[content_type])
    return buffer.getvalue(), base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def image_settings(admin, preserve):
    """`image_settings(**settings)` points the admin's image settings at a stand-in."""
    preserve(IMAGES_CONFIG)

    def configure(**settings) -> None:
        with admin.client() as client:
            save_image_settings(client, **settings)

    return configure


def automatic1111_answering(listener, encoded: str) -> dict:
    settings = serve_automatic1111(listener)
    listener.route("POST", "/sdapi/v1/txt2img", json_answer({"images": [encoded], "info": "{}"}))
    return settings


def openai_answering(listener, encoded: str) -> dict:
    listener.route("POST", "/images/generations", json_answer({"data": [{"b64_json": encoded}]}))
    openai = image_connection(
        "openai", "openai", listener.base_url, "sk-images", models=("dall-e-2",), size="512x512"
    )
    return image_engines.image_settings(openai, edit=False)


def gemini_predict_answering(listener, encoded: str) -> dict:
    settings = serve_gemini(listener)
    prediction = {"bytesBase64Encoded": encoded, "mimeType": "image/png"}
    listener.route(
        "POST",
        f"/models/{IMAGEN_MODEL}:predict",
        json_answer({"predictions": [prediction]}),
    )
    return settings


ENGINES = {
    "automatic1111": automatic1111_answering,
    "openai": openai_answering,
    "gemini": gemini_predict_answering,
}


def generate(actor) -> dict:
    with actor.client() as client:
        generated = client.post("/api/v1/images/generations", json={"prompt": "a harbour", "n": 1})
    assert generated.status_code == 200, generated.text
    [image] = generated.json()
    return image


def stored(actor, image: dict) -> tuple[dict, str, bytes]:
    """The stored file's record, the content type it is served with and its bytes."""
    file_id = image["url"].rstrip("/").split("/")[-2]
    with actor.client() as client:
        record = client.get(f"/api/v1/files/{file_id}")
        served = client.get(image["url"])
    assert record.status_code == 200, record.text
    assert served.status_code == 200, served.text
    return record.json(), served.headers["content-type"], served.content


def assert_stored_as(actor, image: dict, content_type: str, picture: bytes) -> None:
    record, served_type, served_bytes = stored(actor, image)
    extension = EXTENSIONS[content_type]
    labels = (record["meta"].get("content_type"), record["meta"].get("name"), served_type)
    assert served_bytes == picture
    assert record["meta"].get("content_type") == content_type, (
        f"a generated {content_type} image was stored as {labels} (#29948)"
    )
    assert record["meta"]["name"].endswith(extension), labels
    assert served_type.split(";")[0] == content_type, labels


@pytest.mark.parametrize("content_type", ["image/jpeg", "image/webp"])
def test_an_automatic1111_image_keeps_its_real_type(
    image_settings, listener, make_user, content_type
):
    picture, encoded = encoded_image(content_type)
    image_settings(**automatic1111_answering(listener, encoded))
    painter = make_user()

    image = generate(painter)

    assert_stored_as(painter, image, content_type, picture)


@pytest.mark.parametrize("engine", ["openai", "gemini"])
@pytest.mark.parametrize("content_type", ["image/jpeg", "image/webp"])
def test_every_bare_base64_engine_keeps_the_real_type(
    image_settings, listener, make_user, engine, content_type
):
    picture, encoded = encoded_image(content_type)
    image_settings(**ENGINES[engine](listener, encoded))
    painter = make_user()

    image = generate(painter)

    assert_stored_as(painter, image, content_type, picture)


def test_a_gemini_edit_answered_as_jpeg_is_stored_as_jpeg(image_settings, listener, make_user):
    picture, encoded = encoded_image("image/jpeg")
    settings = serve_gemini(listener)
    part = {"inlineData": {"mimeType": "image/png", "data": encoded}}
    listener.route(
        "POST",
        f"/models/{GEMINI_IMAGE_MODEL}:generateContent",
        json_answer({"candidates": [{"content": {"parts": [part]}}]}),
    )
    image_settings(**settings)
    editor = make_user()

    with editor.client() as client:
        edited = client.post(
            "/api/v1/images/edit",
            json={"prompt": "warmer light", "image": f"data:image/png;base64,{PNG_BASE64}"},
        )

    assert edited.status_code == 200, edited.text
    [image] = edited.json()
    assert_stored_as(editor, image, "image/jpeg", picture)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_png_image_is_still_stored_as_png(image_settings, listener, make_user, engine):
    picture, encoded = encoded_image("image/png")
    image_settings(**ENGINES[engine](listener, encoded))
    painter = make_user()

    image = generate(painter)

    assert_stored_as(painter, image, "image/png", picture)


def test_an_answer_that_is_not_an_image_fails_the_generation(image_settings, listener, make_user):
    not_an_image = base64.b64encode(b"this is not a picture at all").decode()
    image_settings(**automatic1111_answering(listener, not_an_image))

    with make_user().client() as client:
        generated = client.post("/api/v1/images/generations", json={"prompt": "a harbour", "n": 1})

    assert generated.status_code == 400, generated.text
