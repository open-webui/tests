"""Dependency smoke: Pillow decides whether an upload may be a model's background image.

Saving a model whose `meta.background_image_url` points at an uploaded file runs
`validate_background_image` in `utils/validate.py`: Pillow opens the bytes, reports the format,
verifies and fully decodes them. A real PNG is accepted and a file that only claims to be one is
refused with 400. A Pillow bump that stops decoding PNGs refuses every background; one that
stops raising on garbage lets it through.

Discriminates: passes on dev bbfa876af; in a backend copy whose `Image.open` fails on every
input the PNG is refused with 400, while the garbage test stays green.
"""

from __future__ import annotations

import io
import uuid

import httpx
import pytest

from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (20, 90, 160)).save(buffer, format="PNG")
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


def test_a_png_is_accepted_as_a_background(admin):
    with admin.client() as client:
        saved = _save_model_with_background(client, _uploaded(client, _png()))

    assert saved.status_code == 200, saved.text


def test_bytes_that_only_claim_to_be_a_png_are_refused(admin):
    with admin.client() as client:
        garbage = b"\x89PNG\r\n\x1a\n" + b"not really an image" * 8
        saved = _save_model_with_background(client, _uploaded(client, garbage))

    assert saved.status_code == 400, saved.text
    assert "Invalid background image" in saved.text
