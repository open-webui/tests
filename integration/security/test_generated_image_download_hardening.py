"""Regression: an image an engine answers as a link was downloaded without the SSRF-safe session.

Fix `b4ebd0d62` (#31623). When an OpenAI-compatible engine answers a generation or an edit with a
`url` in place of the image bytes, Open WebUI checks the link once with the URL safety check and
then downloads it. The download used the shared session, which follows redirects and never judges
the address it connects to, so a link on an allowed host could lead the server to a host the
operator's `WEB_FETCH_FILTER_LIST` blocks. It now goes through the SSRF-safe session, which checks
the host and the resolved address of every request, redirect hops included. A link on the
configured ComfyUI base URL stays trusted, and a private address is still refused up front.

The instance fetches local pages and blocks `127.0.0.2`: the engine on `127.0.0.1` is allowed and
redirects to a listener on the blocked address, so only the session can refuse the hop.

Discriminates: passes on dev a5bc78300; with the download back on the shared session, the
redirect to the blocked address is followed, the blocked listener is reached and the image is
stored (generation and edit). The refusal without local fetch, the unblocked redirect, the
direct link and ComfyUI rows pass on both.
"""

from __future__ import annotations

import base64

import pytest

from harness.actors import admin_of, create_user
from harness.comfyui import PNG, FakeComfyUI, comfyui_settings, serving, workflow
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, openai_settings, save_image_settings
from harness.listener import json_answer, listening

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

BLOCKED_ADDRESS = "127.0.0.2"
LOCAL_FETCH_BLOCKING_ONE_ADDRESS = {
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "WEB_FETCH_FILTER_LIST": f"!{BLOCKED_ADDRESS}/32",
}
ENGINE_KEY = "sk-images"
PICTURE = base64.b64decode(PNG_BASE64)
PICTURE_ANSWER = (200, {"Content-Type": "image/png"}, PICTURE)
PNG_DATA_URL = f"data:image/png;base64,{PNG_BASE64}"
ACTIONS = {
    "generate": ("/api/v1/images/generations", {"prompt": "a lighthouse at dusk"}),
    "edit": ("/api/v1/images/edit", {"prompt": "add fog", "image": PNG_DATA_URL}),
}
ENGINE_PATH = {"generate": "/images/generations", "edit": "/images/edits"}


@pytest.fixture(scope="module")
def fetching(instance_with):
    return instance_with(LOCAL_FETCH_BLOCKING_ONE_ADDRESS)


def _engine_answering_with_link(instance, preserve, listener, link: str):
    """Point the instance's image engine at `listener`, which answers every request with `link`."""
    preserve(IMAGES_CONFIG, on=instance)
    answer = json_answer({"data": [{"url": link}]})
    listener.route("POST", "/images/generations", answer)
    listener.route("POST", "/images/edits", answer)
    with admin_of(instance).client() as client:
        save_image_settings(client, **openai_settings(listener.base_url, ENGINE_KEY))


def _request_image(actor, action: str):
    route, payload = ACTIONS[action]
    with actor.client() as client:
        return client.post(route, json=payload)


def _stored_files(actor) -> list[dict]:
    with actor.client() as client:
        listed = client.get("/api/v1/files/")
    assert listed.status_code == 200, listed.text
    return listed.json()["items"]


def _redirect_to(location: str):
    return 302, {"Location": location}, b""


@pytest.mark.slow
@pytest.mark.parametrize("action", ACTIONS)
def test_a_link_that_redirects_to_a_blocked_address_is_not_followed(
    fetching, preserve, listener, action
):
    painter = create_user(fetching)
    with listening(host=BLOCKED_ADDRESS) as blocked:
        blocked.route("GET", "/secret.png", PICTURE_ANSWER)
        listener.route("GET", "/start.png", _redirect_to(f"{blocked.base_url}/secret.png"))
        _engine_answering_with_link(fetching, preserve, listener, f"{listener.base_url}/start.png")

        response = _request_image(painter, action)
        reached = list(blocked.received)

    assert reached == [], (
        f"the {action} download followed a redirect to {BLOCKED_ADDRESS}, which "
        f"`!{BLOCKED_ADDRESS}/32` blocks, because it ran on a session that judges no hop (#31623)"
    )
    assert response.status_code != 200, response.text
    assert _stored_files(painter) == []


@pytest.mark.slow
@pytest.mark.parametrize("action", ACTIONS)
def test_a_link_on_the_engine_itself_is_downloaded(fetching, preserve, listener, action):
    painter = create_user(fetching)
    listener.route("GET", "/drawn.png", PICTURE_ANSWER)
    _engine_answering_with_link(fetching, preserve, listener, f"{listener.base_url}/drawn.png")

    response = _request_image(painter, action)

    assert response.status_code == 200, response.text
    [image] = response.json()
    with painter.client() as client:
        assert client.get(image["url"]).content == PICTURE


@pytest.mark.slow
@pytest.mark.parametrize("action", ACTIONS)
def test_a_redirect_to_an_unblocked_host_is_still_followed(fetching, preserve, listener, action):
    painter = create_user(fetching)
    listener.route("GET", "/start.png", _redirect_to(f"{listener.base_url}/landing.png"))
    listener.route("GET", "/landing.png", PICTURE_ANSWER)
    _engine_answering_with_link(fetching, preserve, listener, f"{listener.base_url}/start.png")

    response = _request_image(painter, action)

    assert response.status_code == 200, response.text
    assert len(response.json()) == 1


@pytest.mark.parametrize("action", ACTIONS)
def test_a_link_to_a_private_address_is_not_fetched(
    instance, preserve, listener, make_user, action
):
    painter = make_user()
    listener.route("GET", "/internal.png", PICTURE_ANSWER)
    _engine_answering_with_link(instance, preserve, listener, f"{listener.base_url}/internal.png")

    response = _request_image(painter, action)

    assert listener.requests_to("/internal.png") == [], (
        f"the {action} download fetched a loopback link with local web fetch off"
    )
    assert response.status_code != 200, response.text
    assert _stored_files(painter) == []


@pytest.fixture
def comfyui(admin, preserve):
    preserve(IMAGES_CONFIG)
    fake = FakeComfyUI()
    with serving(fake) as base_url, admin.client() as client:
        save_image_settings(client, **comfyui_settings(base_url, workflow("SaveImage")))
        yield fake


@pytest.mark.parametrize("action", ACTIONS)
def test_comfyui_on_its_configured_address_still_delivers_the_image(comfyui, make_user, action):
    painter = make_user()

    response = _request_image(painter, action)

    assert response.status_code == 200, response.text
    [image] = response.json()
    with painter.client() as client:
        assert client.get(image["url"]).content == PNG
