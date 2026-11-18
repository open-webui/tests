"""Regression: a large model profile image was carried by every read of the model.

Issue open-webui/open-webui#34656.

A profile image picked in the model editor was stored as base64 text inside the model's settings,
so an animated GIF of a few MB was read, parsed and sent back on every read of the model. The fix
(branch perf/model-profile-image-files) has the editor upload the picture as a file and the model
keep only its `/api/v1/files/<id>/content` link, checked like a background image on every save.
Before it, the server dropped such a link on save, so the model fell back to the default logo.

Over HTTP the fix shows as: a model saved with a file link keeps it, its reads stay a few hundred
bytes, its profile image route serves the file to whoever may read the model and the default logo
to anyone else, a file the saver cannot read is refused, and the export still carries the picture
inline. A profile image saved as a data URI keeps loading as before.

Discriminates: passes on perf/model-profile-image-files, fails on dev 2e0e9f25c (the file link is
dropped on save, so the model reads back without it, the reader gets the default logo, the foreign
file is accepted with 200 and the export carries no picture).
"""

from __future__ import annotations

import base64
import io
import random
import uuid

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

Image = pytest.importorskip("PIL.Image")

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DEFAULT_LOGO = "/static/favicon.png"
# A model read carries its settings; with the picture inline it would be several MB.
MODEL_READ_LIMIT = 64 * 1024


def _large_animated_gif() -> bytes:
    """About 3.5 MB of noise frames, under the 5 MiB limit for an image file."""
    noise = random.Random(7)
    frames = [
        Image.frombytes("L", (256, 256), noise.randbytes(256 * 256)).convert("P") for _ in range(40)
    ]
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:], duration=80, loop=0
    )
    return buffer.getvalue()


LARGE_GIF = _large_animated_gif()


def _upload(client: httpx.Client, content: bytes) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "false"},
        files={"file": ("mascot.gif", content, "image/gif")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return f"/api/v1/files/{uploaded.json()['id']}/content"


def _model_form(model_id: str, profile_image_url: str, access_grants: list[dict]) -> dict:
    return {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Harbour mascot",
        "meta": {"profile_image_url": profile_image_url},
        "params": {},
        "access_grants": access_grants,
    }


def _create(actor: Actor, profile_image_url: str, access_grants: list[dict] = ()) -> str:
    model_id = f"mascot-{uuid.uuid4().hex[:8]}"
    with actor.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json=_model_form(model_id, profile_image_url, list(access_grants)),
        )
    assert created.status_code == 200, created.text
    return model_id


def _profile_image(actor: Actor, model_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get("/api/v1/models/model/profile/image", params={"id": model_id})


def _assert_serves(response: httpx.Response, content: bytes, content_type: str, who: str) -> None:
    assert response.status_code == 200, (
        f"{who} got {response.status_code} {response.headers.get('location')} "
        "instead of the model's profile image"
    )
    assert response.headers["content-type"] == content_type
    assert response.content == content, f"{who} got {len(response.content)} other bytes"


def test_a_profile_image_saved_as_a_file_loads_and_stays_out_of_model_reads(admin, make_user):
    reader, stranger = make_user(), make_user()
    with admin.client() as client:
        file_url = _upload(client, LARGE_GIF)
    model_id = _create(admin, file_url, [grant("user", reader.id, "read")])

    with reader.client() as client:
        read = client.get("/api/v1/models/model", params={"id": model_id})
        file_for_reader = client.get(file_url)
    assert read.status_code == 200, read.text
    stored = read.json()["meta"].get("profile_image_url")
    assert stored == file_url, f"the model was saved without its profile image file: {stored!r}"
    assert len(read.content) < MODEL_READ_LIMIT, (
        f"reading the model sent {len(read.content)} bytes for a {len(LARGE_GIF)}-byte picture"
    )
    _assert_serves(_profile_image(reader, model_id), LARGE_GIF, "image/gif", "a reader")
    _assert_serves(_profile_image(admin, model_id), LARGE_GIF, "image/gif", "the owner")
    assert file_for_reader.status_code == 200, "a reader of the model cannot open its picture"

    denied = _profile_image(stranger, model_id)
    assert denied.status_code == 302 and denied.headers["location"] == DEFAULT_LOGO, (
        f"someone without access got {denied.status_code} instead of the default logo"
    )
    with stranger.client() as client:
        opened = client.get(file_url)
    assert opened.status_code != 200, "someone without access to the model opened its picture"


def test_a_file_the_saver_cannot_read_is_refused_as_a_profile_image(admin, make_user):
    builder = make_user()
    make_group(admin, [builder], permissions={"workspace": {"models": True}})
    with admin.client() as client:
        foreign_url = _upload(client, LARGE_GIF)
    with builder.client() as client:
        own_url = _upload(client, LARGE_GIF)
    model_id = _create(builder, own_url)

    with builder.client() as client:
        attempts = {
            "create": client.post(
                "/api/v1/models/create",
                json=_model_form(f"mascot-{uuid.uuid4().hex[:8]}", foreign_url, []),
            ),
            "update": client.post(
                "/api/v1/models/model/update",
                params={"id": model_id},
                json=_model_form(model_id, foreign_url, []),
            ),
        }
        kept = client.get("/api/v1/models/model", params={"id": model_id}).json()
    refused = {route: response.status_code for route, response in attempts.items()}
    assert refused == {"create": 403, "update": 403}, (
        f"saving someone else's file as a profile image was answered {refused}"
    )
    assert kept["meta"]["profile_image_url"] == own_url


def test_an_export_carries_the_picture_inline_and_a_data_uri_still_loads(admin):
    with admin.client() as client:
        file_model = _create(admin, _upload(client, LARGE_GIF))
    inline_url = "data:image/gif;base64," + base64.b64encode(LARGE_GIF).decode()
    inline_model = _create(admin, inline_url)

    with admin.client() as client:
        exported = client.get("/api/v1/models/export", params={"ids": [file_model]})
    assert exported.status_code == 200, exported.text
    exported_url = exported.json()[0]["meta"].get("profile_image_url")
    assert exported_url == inline_url, (
        f"the export carries {str(exported_url)[:60]!r} instead of the picture inline"
    )
    _assert_serves(_profile_image(admin, inline_model), LARGE_GIF, "image/gif", "the owner")
