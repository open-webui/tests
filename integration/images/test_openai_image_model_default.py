"""Regression: with no image model set, OpenAI image generation asked for the shut-down DALL·E 2.

Fix d780d386e (PR open-webui/open-webui#33694, issue #33578). With the OpenAI engine and the
Model field left empty, every generation went out for `dall-e-2`, which OpenAI no longer serves,
so image generation failed out of the box, and the model list still offered DALL·E 2 and 3. The
default is now `gpt-image-2` and the list offers only the GPT image models. A local service plays
OpenAI's images API, so the tests read the request it got.

The default request still carries `response_format`, which Open WebUI leaves out for every model
matching `IMAGE_URL_RESPONSE_MODELS_REGEX_PATTERN` (`^gpt-image`), as the docs promise, and does
leave out when the admin picks `gpt-image-2` by name: the check reads the empty setting instead of
the model it sends (issue open-webui/open-webui#33874). OpenAI documents `response_format` as
unsupported for GPT image models.
`test_the_default_request_matches_the_default_model_chosen_by_name` stays red until that fix merges.

Discriminates: on dev f659c0d12 the default and model list tests pass and fail on a backend copy
with d780d386e reverted (the request asks for `dall-e-2`, the list offers both DALL·E models); the
explicit model test passes on both. The default request test fails on dev 1fc5b8f6d (the default
request carries `response_format`) and passes with the #33874 fix.
"""

from __future__ import annotations

import pytest

from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DEFAULT_MODEL = "gpt-image-2"


@pytest.fixture
def openai_images(admin, preserve, listener):
    """`openai_images(model)` points the admin's image settings at a stand-in OpenAI."""
    preserve(IMAGES_CONFIG)
    listener.route("POST", "/images/generations", json_answer({"data": [{"b64_json": PNG_BASE64}]}))

    def configure(model: str) -> None:
        with admin.client() as client:
            save_image_settings(
                client,
                ENABLE_IMAGE_GENERATION=True,
                IMAGE_GENERATION_ENGINE="openai",
                IMAGES_OPENAI_API_BASE_URL=listener.base_url,
                IMAGES_OPENAI_API_KEY="sk-images",
                IMAGE_GENERATION_MODEL=model,
                IMAGE_SIZE="1024x1024",
            )

    return configure


def generation_sent(actor, listener, prompt: str) -> dict:
    """Generate an image as `actor`; returns the request the stand-in OpenAI got for it."""
    with actor.client() as client:
        generated = client.post("/api/v1/images/generations", json={"prompt": prompt, "n": 1})
    assert generated.status_code == 200, generated.text
    sent = [call.json() for call in listener.requests_to("/images/generations")]
    [request] = [body for body in sent if body.get("prompt") == prompt]
    return request


def test_with_no_model_set_openai_is_asked_for_gpt_image_2(openai_images, listener, make_user):
    openai_images("")

    sent = generation_sent(make_user(), listener, "a lighthouse at dusk")

    assert sent["model"] == DEFAULT_MODEL, (
        f"with no image model set the request asked OpenAI for {sent['model']!r}; DALL·E 2 is "
        "shut down, so image generation fails out of the box (#33578)"
    )


def test_the_openai_model_list_offers_no_shut_down_model(openai_images, make_user):
    openai_images("")

    with make_user().client() as client:
        listed = client.get("/api/v1/images/models")

    assert listed.status_code == 200, listed.text
    offered = [model["id"] for model in listed.json()]
    assert DEFAULT_MODEL in offered, offered
    assert not [model for model in offered if model.startswith("dall-e")], (
        f"the model list still offers DALL·E models OpenAI no longer serves: {offered}"
    )


def test_a_model_chosen_by_name_is_sent_as_it_is(openai_images, listener, make_user):
    openai_images("gpt-image-1")

    sent = generation_sent(make_user(), listener, "a harbour crane")

    assert sent["model"] == "gpt-image-1"
    assert "response_format" not in sent, sent


def test_the_default_request_matches_the_default_model_chosen_by_name(
    openai_images, listener, make_user
):
    painter = make_user()
    openai_images(DEFAULT_MODEL)
    chosen = generation_sent(painter, listener, "a tugboat, chosen model")
    openai_images("")
    defaulted = generation_sent(painter, listener, "a tugboat, default model")

    assert "response_format" not in chosen, chosen
    assert {**defaulted, "prompt": None} == {**chosen, "prompt": None}, (
        "with no model set, the request for the default GPT image model differs from the same "
        f"model picked by name: {defaulted} against {chosen}"
    )
