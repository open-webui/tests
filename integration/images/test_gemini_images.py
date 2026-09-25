"""Image generation and editing on the Gemini engine, against a local stand-in for the Gemini API.

The admin picks Gemini for both; a user then generates a picture through Imagen's `:predict`,
through Gemini's `:generateContent` and edits one of their own. The stand-in refuses any other
API key, so the tests read what Open WebUI sent it (prompt, key, the picture to edit) and fetch
the stored result back.

Discriminates: fails with the generation headers in `image_generations` built without
`x-goog-api-key` (both generation paths answer 400) and with the edit picture left out of the
Gemini edit request (the stand-in never gets it).
"""

from __future__ import annotations

import base64

import pytest

from harness.image_engines import (
    GEMINI_API_KEY,
    GEMINI_IMAGE_MODEL,
    IMAGEN_MODEL,
    IMAGES_CONFIG,
    PNG_BASE64,
    gemini_calls,
    save_image_settings,
    serve_gemini,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

GENERATED_PNG = base64.b64decode(PNG_BASE64)
# a 1x1 green PNG, the picture a user edits
SOURCE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)


@pytest.fixture
def gemini(admin, preserve, listener):
    """`gemini(**overrides)` points the admin's image settings at the stand-in."""
    preserve(IMAGES_CONFIG)
    settings = serve_gemini(listener)

    def configure(**overrides):
        with admin.client() as client:
            save_image_settings(client, **{**settings, **overrides})

    return configure


def stored_picture(actor, image: dict) -> bytes:
    with actor.client() as client:
        fetched = client.get(image["url"])
    assert fetched.status_code == 200, fetched.text
    return fetched.content


def test_imagen_predict_generates_a_stored_picture(gemini, listener, make_user):
    gemini()
    painter = make_user()

    with painter.client() as client:
        generated = client.post(
            "/api/v1/images/generations", json={"prompt": "a lighthouse at dusk", "n": 1}
        )

    assert generated.status_code == 200, generated.text
    [image] = generated.json()
    assert stored_picture(painter, image) == GENERATED_PNG
    [call] = gemini_calls(listener, IMAGEN_MODEL, "predict")
    assert call.headers.get("x-goog-api-key") == GEMINI_API_KEY
    assert call.json()["instances"] == {"prompt": "a lighthouse at dusk"}
    assert call.json()["parameters"]["sampleCount"] == 1


def test_generate_content_generates_a_stored_picture(gemini, listener, make_user):
    gemini(
        IMAGE_GENERATION_MODEL=GEMINI_IMAGE_MODEL, IMAGES_GEMINI_ENDPOINT_METHOD="generateContent"
    )
    painter = make_user()

    with painter.client() as client:
        generated = client.post("/api/v1/images/generations", json={"prompt": "a red kite"})

    assert generated.status_code == 200, generated.text
    [image] = generated.json()
    assert stored_picture(painter, image) == GENERATED_PNG
    [call] = gemini_calls(listener, GEMINI_IMAGE_MODEL, "generateContent")
    assert call.headers.get("x-goog-api-key") == GEMINI_API_KEY
    assert call.json() == {"contents": [{"parts": [{"text": "a red kite"}]}]}


def test_an_edit_sends_the_picture_and_stores_the_result(gemini, listener, make_user):
    gemini()
    painter = make_user()

    with painter.client() as client:
        edited = client.post(
            "/api/v1/images/edit",
            json={
                "image": f"data:image/png;base64,{SOURCE_PNG_BASE64}",
                "prompt": "make it blue",
            },
        )

    assert edited.status_code == 200, edited.text
    [image] = edited.json()
    assert stored_picture(painter, image) == GENERATED_PNG
    [call] = gemini_calls(listener, GEMINI_IMAGE_MODEL, "generateContent")
    assert call.headers.get("x-goog-api-key") == GEMINI_API_KEY
    assert call.json()["contents"][0]["parts"] == [
        {"text": "make it blue"},
        {"inline_data": {"mime_type": "image/png", "data": SOURCE_PNG_BASE64}},
    ]


def test_a_key_gemini_refuses_is_a_failed_generation(gemini, listener, make_user):
    gemini(IMAGES_GEMINI_API_KEY="revoked-key")

    with make_user().client() as client:
        generated = client.post("/api/v1/images/generations", json={"prompt": "a lighthouse"})

    assert generated.status_code == 400
    [call] = gemini_calls(listener, IMAGEN_MODEL, "predict")
    assert call.headers.get("x-goog-api-key") == "revoked-key"
