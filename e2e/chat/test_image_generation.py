"""Journey: image generation in the chat, from the admin's engine settings to the stored picture.

The admin picks the OpenAI engine in Admin Settings > Images and points it at a local stand-in
for OpenAI's image endpoints (`harness.channel_chat.serve_openai_images`). A user then turns on
Image under Integrations, the model calls `generate_image` (native tool calling, the default)
and the picture shows in the reply and is still there after a reload. Asked for a change, the
model calls `edit_image` on the picture it drew. The Image switch is offered only while the
admin has image generation on and the user holds the image generation permission, a model with
image generation on by default starts with the chip lit, and an engine that fails leaves no
picture and shows its error in the tool result.

`test_a_model_with_image_generation_by_default_starts_with_the_chip_on` is red now and then on dev
7b7dba6ee: since de73bb830 a live update to the reply can write back an older copy of it, so the
picture drops out of the reply, or the reply stays blank, until a reload
(open-webui/open-webui#32091). With the chat list answering 1.5 s late, dev 22102e4a2 lost the
reply and its picture in 5 of 5 runs and a build without the per-update copy in none.

Discriminates: passes on dev 176d31d1d with its built frontend. In a backend copy, with
`/api/v1/images/config/update` ignoring `IMAGES_OPENAI_API_KEY` the settings test fails (the key
is empty after the reload), with `generate_image` not storing its files on the chat message the
reload test fails (the picture is gone after the reload), with `edit_image` left out of the
builtin tools the edit test fails (no edited picture) and with `generate_image` answering an
empty success on errors the failing engine test fails. In a frontend build, with the composer
ignoring the admin switch and the user permission the two toggle tests fail, and with the model's
default features ignored the default feature test fails (no Image chip).
"""

from __future__ import annotations

import base64
import json
import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import drawing_model, serve_openai_images
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.listener import json_answer
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    conversation,
    expect_reply,
    last_reply,
    replies,
    send,
)
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DRAWN_PNG = base64.b64decode(PNG_BASE64)
# a 1x1 green PNG, what the stand-in answers for an edit
EDITED_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)
FILE_CONTENT_URL = re.compile(r"/api/v1/files/[^/]+/content$")


@pytest.fixture
def image_engine(admin, preserve, listener):
    """`image_engine(**overrides)` points the admin's image settings at the OpenAI stand-in."""
    preserve(IMAGES_CONFIG)
    settings = serve_openai_images(listener)

    def configure(**overrides) -> None:
        with admin.client() as client:
            save_image_settings(client, **{**settings, **overrides})

    return configure


@pytest.fixture
def denied_image_generation(admin, preserve):
    """Take the image generation permission away from every account by default."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["image_generation"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    saved.raise_for_status()


def image_settings(page: Page) -> Locator:
    """Admin Settings > Images, freshly loaded from the server."""
    page.goto("/admin/settings/images")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()
    return settings


def open_chat(page: Page) -> Page:
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return page


def integrations_menu(page: Page) -> Locator:
    page.get_by_label("Integrations").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button").first).to_be_visible()
    return menu


def turn_on_image(page: Page) -> None:
    integrations_menu(page).get_by_role("button", name="Image", exact=True).click()
    page.keyboard.press("Escape")


def image_chip(page: Page) -> Locator:
    """The lit Image chip beside the chat input."""
    page.wait_for_function(
        "() => [...document.querySelectorAll('form button')]"
        ".some((button) => button.parentElement?._tippy?.props.content === 'Image')"
    )
    return tooltip_button(page.locator("form"), "Image")


def reply_image(page: Page, reply_index: int = -1) -> Locator:
    return replies(page).nth(reply_index).get_by_role("img")


def stored_picture(actor, url: str) -> bytes:
    with actor.client() as client:
        fetched = client.get(url)
    assert fetched.status_code == 200, fetched.text
    return fetched.content


def tool_results(upstream, prompt: str) -> list[str]:
    follow_ups = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    return [entry["content"] for entry in follow_ups[-1]["messages"] if entry["role"] == "tool"]


def draw(page: Page, upstream, prompt: str, answer: str) -> None:
    upstream.queue(
        reply.tool_call("generate_image", {"prompt": prompt}, match=reply.answering(prompt)),
        reply.text(answer, match=reply.answering(prompt)),
    )
    send(page, prompt)
    expect_reply(page, answer)


def test_an_admin_sets_up_the_openai_engine_and_it_draws(
    page_for, make_user, preserve, listener, upstream
):
    preserve(IMAGES_CONFIG)
    serve_openai_images(listener)
    admin_page = page_for(make_user(role="admin"))
    settings = image_settings(admin_page)
    settings.get_by_role("switch", name="Image Generation").click()
    settings.get_by_role("combobox", name="Select Engine").first.select_option(
        label="Default (Open AI)"
    )
    settings.get_by_role("combobox", name="Select a model").fill("dall-e-2")
    # the first pair is image generation's, the second image editing's
    settings.get_by_role("textbox", name="API Base URL").first.fill(listener.base_url)
    settings.get_by_role("textbox", name="API Key").first.fill("sk-drawn")
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    settings = image_settings(admin_page)
    expect(settings.get_by_role("switch", name="Image Generation")).to_be_checked()
    expect(settings.get_by_role("combobox", name="Select Engine").first).to_have_value("openai")
    expect(settings.get_by_role("textbox", name="API Base URL").first).to_have_value(
        listener.base_url
    )
    expect(settings.get_by_role("textbox", name="API Key").first).to_have_value("sk-drawn")

    admin_page.goto("/")
    turn_on_image(open_chat(admin_page))
    draw(admin_page, upstream, "draw a lighthouse", "Here is your lighthouse.")
    expect(reply_image(admin_page)).to_be_visible()
    [generation] = listener.requests_to("/images/generations")
    assert generation.headers.get("Authorization") == "Bearer sk-drawn"
    assert generation.json()["prompt"] == "draw a lighthouse"


def test_the_image_switch_follows_the_admin_setting(page_for, make_user, image_engine):
    image_engine(ENABLE_IMAGE_GENERATION=False, ENABLE_IMAGE_EDIT=False)
    page = open_chat(page_for(make_user()))
    image_switch = integrations_menu(page).get_by_role("button", name="Image", exact=True)
    expect(image_switch).to_have_count(0)

    image_engine()
    page.reload()
    expect(
        integrations_menu(open_chat(page)).get_by_role("button", name="Image", exact=True)
    ).to_be_visible()


def test_a_user_without_the_permission_gets_no_image_switch(
    page_for, make_user, image_engine, denied_image_generation
):
    image_engine()
    page = open_chat(page_for(make_user()))
    image_switch = integrations_menu(page).get_by_role("button", name="Image", exact=True)
    expect(image_switch).to_have_count(0)


def test_a_drawn_picture_is_stored_and_shows_after_a_reload(
    page_for, make_user, image_engine, listener, upstream
):
    image_engine()
    person = make_user()
    page = open_chat(page_for(person))
    turn_on_image(page)
    expect(image_chip(page)).to_be_visible()
    draw(page, upstream, "draw a red square", "Here is your red square.")
    expect(reply_image(page)).to_be_visible()
    source = reply_image(page).get_attribute("src")
    assert FILE_CONTENT_URL.search(source), f"the picture is not a stored file: {source}"
    assert listener.requests_to("/images/generations"), "the engine was never asked"

    expect(page).to_have_url(re.compile(r"/c/"))
    page.reload()
    expect_reply(page, "Here is your red square.")
    expect(reply_image(page)).to_be_visible()
    expect(reply_image(page)).to_have_attribute("src", source)
    assert stored_picture(person, source) == DRAWN_PNG


def test_the_model_edits_the_picture_it_drew(page_for, make_user, image_engine, listener, upstream):
    image_engine()
    listener.route(
        "POST", "/images/edits", json_answer({"data": [{"b64_json": EDITED_PNG_BASE64}]})
    )
    person = make_user()
    page = open_chat(page_for(person))
    turn_on_image(page)
    draw(page, upstream, "draw a red square", "Here is your red square.")
    [drawn] = json.loads(tool_results(upstream, "draw a red square")[0])["images"]

    change = "now make it green"
    upstream.queue(
        reply.tool_call(
            "edit_image",
            {"prompt": "make it green", "image_urls": [drawn["url"]]},
            match=reply.answering(change),
        ),
        reply.text("Now it is green.", match=reply.answering(change)),
    )
    send(page, change)
    expect_reply(page, "Now it is green.")
    expect(reply_image(page)).to_be_visible()
    edited_source = reply_image(page).get_attribute("src")
    assert edited_source != reply_image(page, 0).get_attribute("src")
    assert stored_picture(person, edited_source) == base64.b64decode(EDITED_PNG_BASE64)
    [edit] = listener.requests_to("/images/edits")
    assert DRAWN_PNG in edit.body, "the edit engine never got the drawn picture"


def test_a_model_with_image_generation_by_default_starts_with_the_chip_on(
    page_for, make_user, admin, image_engine, listener, upstream
):
    image_engine()
    with admin.client() as client, drawing_model(client) as model_id:
        page = page_for(make_user(role="admin"))
        page.goto(f"/?models={model_id}")
        open_chat(page)
        expect(image_chip(page)).to_be_visible()
        draw(page, upstream, "draw a blue circle", "Here is your blue circle.")
        expect(reply_image(page)).to_be_visible()
    assert listener.requests_to("/images/generations"), "the engine was never asked"


def test_a_failing_engine_shows_its_error_and_no_picture(
    page_for, make_user, image_engine, listener, upstream
):
    image_engine()
    listener.route("POST", "/images/generations", json_answer({"error": "engine down"}, 500))
    page = open_chat(page_for(make_user()))
    turn_on_image(page)
    draw(page, upstream, "draw a red square", "I could not draw that.")

    [result] = tool_results(upstream, "draw a red square")
    assert "error" in json.loads(result), f"the model was not told the engine failed: {result}"
    expect(reply_image(page)).to_have_count(0)
    conversation(page).get_by_text("View Result from generate_image").click()
    expect(last_reply(page).get_by_text(re.compile("Internal Server Error"))).to_be_visible()
