"""Regression: image edit reads a file URL of Open WebUI's own from the file store, not over HTTP.

open-webui 0.11.2 fix `50413f348` (withheld advisory): `image_edits` sent an absolute URL naming
Open WebUI's own `/api/v1/files/<id>/content` through the outbound HTTP client, so the server
requested from itself a file it can open directly, unauthenticated, and the web fetch guard
refused its own loopback address. The fix reads the file through `get_file_content_by_id`, which
keeps the ownership checks; since PR #29691 the content path is recognised on any host, because a
network fetch of it can never authenticate.

A local service plays the OpenAI edit endpoint, so the tests read which picture it was sent.

Twin of unit/security/test_image_edit_own_origin_file_url.py.

Discriminates: passes on dev bbfa876af; without the content-path branch the own-origin and
foreign-host URLs reach the fetch guard, which refuses loopback, so the edit is a 400 and the
edit service never gets the picture. Matching on the host alone fails the foreign-host case and
the nearby file-metadata URL, which it reads from the store.
"""

from __future__ import annotations

import base64

import pytest

from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.listener import json_answer, text_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

STORED_PNG = base64.b64decode(PNG_BASE64)
INLINE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)
INLINE_DATA_URL = f"data:image/png;base64,{base64.b64encode(INLINE_PNG).decode()}"


@pytest.fixture
def edit_service(admin, preserve, listener):
    preserve(IMAGES_CONFIG)
    edited = {"data": [{"b64_json": base64.b64encode(INLINE_PNG).decode()}]}
    listener.route("POST", "/images/edits", json_answer(edited))
    with admin.client() as client:
        save_image_settings(
            client,
            ENABLE_IMAGE_EDIT=True,
            IMAGE_EDIT_ENGINE="openai",
            IMAGE_EDIT_MODEL="gpt-image-1",
            IMAGES_EDIT_OPENAI_API_BASE_URL=listener.base_url,
            IMAGES_EDIT_OPENAI_API_KEY="edit-key",
        )
    return listener


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def file_id(owner):
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/?process=false", files={"file": ("photo.png", STORED_PNG, "image/png")}
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def edit(actor, image):
    with actor.client() as client:
        return client.post("/api/v1/images/edit", json={"image": image, "prompt": "make it blue"})


def pictures_sent(edit_service):
    return [call.body for call in edit_service.requests_to("/images/edits")]


def test_own_origin_file_url_is_sent_as_the_stored_file(edit_service, instance, owner, file_id):
    response = edit(owner, f"{instance.base_url}/api/v1/files/{file_id}/content")

    assert response.status_code == 200, response.text
    [sent] = pictures_sent(edit_service)
    assert STORED_PNG in sent


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.2:9/api/v1/files/{id}/content",  # same path on a host that is not us
        "{base}/api/v1/files/{id}/content?attachment=true",
        "{base}/api/v1/files/{id}/content/html",
    ],
)
def test_every_file_content_url_is_read_from_the_store(edit_service, instance, owner, file_id, url):
    response = edit(owner, url.format(base=instance.base_url, id=file_id))

    assert response.status_code == 200, response.text
    [sent] = pictures_sent(edit_service)
    assert STORED_PNG in sent


def test_own_origin_file_url_in_a_list_is_read_from_the_store(
    edit_service, instance, owner, file_id
):
    own_origin = f"{instance.base_url}/api/v1/files/{file_id}/content"

    response = edit(owner, [own_origin, INLINE_DATA_URL])

    assert response.status_code == 200, response.text
    [sent] = pictures_sent(edit_service)
    assert STORED_PNG in sent and INLINE_PNG in sent


def test_another_users_file_is_refused(edit_service, instance, file_id, make_user):
    response = edit(make_user(), f"{instance.base_url}/api/v1/files/{file_id}/content")

    assert response.status_code == 404
    assert pictures_sent(edit_service) == []


def test_own_origin_url_outside_the_content_route_is_not_read_from_the_store(
    edit_service, instance, owner, file_id
):
    response = edit(owner, f"{instance.base_url}/api/v1/files/{file_id}")

    assert response.status_code == 400
    assert pictures_sent(edit_service) == []


def test_a_loopback_picture_url_is_refused_by_the_fetch_guard(edit_service, owner):
    edit_service.route("GET", "/photo.png", text_answer("not reached", content_type="image/png"))

    response = edit(owner, f"{edit_service.base_url}/photo.png")

    assert response.status_code == 400
    assert edit_service.requests_to("/photo.png") == []
    assert pictures_sent(edit_service) == []


@pytest.mark.parametrize("form", ["file id", "relative path"])
def test_a_file_named_without_a_host_is_read_from_the_store(edit_service, owner, file_id, form):
    image = file_id if form == "file id" else f"/api/v1/files/{file_id}/content"

    response = edit(owner, image)

    assert response.status_code == 200, response.text
    [sent] = pictures_sent(edit_service)
    assert STORED_PNG in sent


def test_a_data_url_is_sent_as_it_is(edit_service, owner):
    response = edit(owner, INLINE_DATA_URL)

    assert response.status_code == 200, response.text
    [sent] = pictures_sent(edit_service)
    assert INLINE_PNG in sent
