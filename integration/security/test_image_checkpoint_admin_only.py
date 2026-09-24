"""Regression: a regular user's image request must not switch the shared Automatic1111 checkpoint.

open-webui 0.11.0 fix `8becf9443` (#27244): the Automatic1111 branch of `image_generations`
called `set_image_model` whenever the request carried a `model`. That call is not scoped to the
request: it saves `image_generation.model` to the global config and POSTs `sd_model_checkpoint`
to the Automatic1111 server, which holds one checkpoint for every user. Anyone allowed to
generate images could repoint the image model for the whole instance. Only an admin's request
switches it now.

A local service plays Automatic1111, so the tests read what it was sent and what the admin's
image settings say afterwards.

Twin of unit/security/test_image_checkpoint_admin_only.py.

Discriminates: passes on dev bbfa876af; without the `user.role == 'admin'` gate a regular user's
request POSTs the checkpoint switch and rewrites the admin's model, so the narrow and broad
generation cases fail.
"""

from __future__ import annotations

import pytest

from harness.image_engines import (
    CHECKPOINT,
    IMAGES_CONFIG,
    checkpoint_switches,
    save_image_settings,
    serve_automatic1111,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

REQUESTED = "requested.safetensors"


@pytest.fixture
def automatic1111(admin, preserve, listener):
    """`automatic1111(engine)` points the admin's image settings at the listener."""
    preserve(IMAGES_CONFIG)
    settings = serve_automatic1111(listener)

    def configure(engine="automatic1111"):
        with admin.client() as client:
            save_image_settings(client, **{**settings, "IMAGE_GENERATION_ENGINE": engine})

    return configure


def generate(actor, **request):
    with actor.client() as client:
        response = client.post("/api/v1/images/generations", json={"prompt": "a cat", **request})
    assert response.status_code == 200, response.text
    return response.json()


def image_settings(admin):
    with admin.client() as client:
        return client.get(IMAGES_CONFIG[0]).json()


def txt2img_payload(listener):
    [call] = listener.requests_to("/sdapi/v1/txt2img")
    return call.json()


def test_regular_user_model_does_not_switch_the_checkpoint(
    automatic1111, listener, admin, make_user
):
    automatic1111()

    generate(make_user(), model=REQUESTED)

    assert checkpoint_switches(listener) == [], "a regular user switched the shared checkpoint"
    assert image_settings(admin)["IMAGE_GENERATION_MODEL"] == CHECKPOINT


@pytest.mark.parametrize("engine", ["automatic1111", ""])  # "" is Automatic1111 too
def test_regular_user_request_changes_no_image_setting(
    automatic1111, listener, admin, make_user, engine
):
    automatic1111(engine)
    before = image_settings(admin)

    generate(make_user(), model=REQUESTED, size="1024x1024", n=2, steps=7, negative_prompt="blurry")

    assert image_settings(admin) == before
    assert checkpoint_switches(listener) == []


def test_image_settings_are_admin_only(automatic1111, listener, admin, make_user):
    automatic1111()
    settings = image_settings(admin)

    with make_user().client() as client:
        refused = [
            client.get(IMAGES_CONFIG[0]),
            client.post(IMAGES_CONFIG[1], json=settings),
            client.post("/api/v1/images/verify", json={"url": listener.base_url}),
        ]

    assert [response.status_code for response in refused] == [401, 401, 401]


def test_admin_model_still_switches_the_checkpoint(automatic1111, listener, admin):
    automatic1111()

    generate(admin, model="admin-choice.safetensors")

    assert [switch["sd_model_checkpoint"] for switch in checkpoint_switches(listener)] == [
        "admin-choice.safetensors"
    ]
    assert image_settings(admin)["IMAGE_GENERATION_MODEL"] == "admin-choice.safetensors"


def test_regular_user_request_values_reach_automatic1111(automatic1111, listener, make_user):
    automatic1111()

    images = generate(
        make_user(), model=REQUESTED, size="768x1024", n=3, steps=7, negative_prompt="blurry"
    )

    assert txt2img_payload(listener) == {
        "prompt": "a cat",
        "batch_size": 3,
        "width": 768,
        "height": 1024,
        "steps": 7,
        "negative_prompt": "blurry",
    }
    assert [image["url"] for image in images] == [f"/api/v1/files/{images[0]['id']}/content"]


def test_regular_user_gets_the_configured_size_and_steps(automatic1111, listener, make_user):
    automatic1111()

    generate(make_user())

    payload = txt2img_payload(listener)
    assert (payload["width"], payload["height"], payload["steps"]) == (512, 512, 20)
