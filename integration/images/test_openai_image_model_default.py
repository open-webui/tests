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

Image editing had the same check (fix d2a8374f3, PR open-webui/open-webui#33875): the format was
picked from the saved edit model, so a request naming a GPT image model was still asked for
base64 and one naming DALL·E 2 over a saved GPT image model was not. The edit test passes on dev
e8c0ad85e and fails in both cases on a backend copy with d2a8374f3 reverted.
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


@pytest.fixture
def openai_edits(admin, preserve, listener):
    """`openai_edits(model)` points the admin's image edit settings at a stand-in OpenAI."""
    preserve(IMAGES_CONFIG)
    listener.route("POST", "/images/edits", json_answer({"data": [{"b64_json": PNG_BASE64}]}))

    def configure(model: str) -> None:
        with admin.client() as client:
            save_image_settings(
                client,
                ENABLE_IMAGE_EDIT=True,
                IMAGE_EDIT_ENGINE="openai",
                IMAGE_EDIT_MODEL=model,
                IMAGES_EDIT_OPENAI_API_BASE_URL=listener.base_url,
                IMAGES_EDIT_OPENAI_API_KEY="sk-edits",
            )

    return configure


def edit_sent(actor, listener, prompt: str, model: str) -> bytes:
    """Edit an image as `actor`, naming `model`; returns the form the stand-in OpenAI got."""
    form = {"image": f"data:image/png;base64,{PNG_BASE64}", "prompt": prompt, "model": model}
    with actor.client() as client:
        edited = client.post("/api/v1/images/edit", json=form)
    assert edited.status_code == 200, edited.text
    sent = [call.body for call in listener.requests_to("/images/edits")]
    [request] = [body for body in sent if prompt.encode() in body]
    return request


@pytest.mark.parametrize(
    ("saved", "requested", "asks_for_base64"),
    [("", "gpt-image-1", False), ("gpt-image-1", "dall-e-2", True)],
    ids=["gpt-image-named-in-the-request", "dall-e-named-in-the-request"],
)
def test_an_edit_asks_for_base64_only_from_the_model_it_sends(
    openai_edits, listener, make_user, saved, requested, asks_for_base64
):
    openai_edits(saved)

    sent = edit_sent(make_user(), listener, f"paint it red for {requested}", requested)

    assert f'name="model"\r\n\r\n{requested}'.encode() in sent, sent[:600]
    assert (b'name="response_format"' in sent) == asks_for_base64, (
        f"an edit with {requested} named in the request and {saved!r} saved "
        f"{'did not ask' if asks_for_base64 else 'asked'} for base64 output; the format must "
        "follow the model the request sends, GPT image models refuse it (#33874)"
    )
