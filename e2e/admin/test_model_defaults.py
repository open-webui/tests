"""Journey: the Model Defaults an admin sets in Admin Settings > Models reach every model's chats.

Model Defaults holds what a model without settings of its own starts from. A prompt suggestion
added there shows on a user's new chat and sends its prompt when pressed; a parameter set there
reaches the provider with the user's chat; a capability switched off there takes that ability
from the model, so an image attached to a chat with it is refused where it attached before.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Model Defaults save sends the
stored defaults back in place of the edited ones, every test but the untouched attach fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
# a PNG of one pixel, enough for the chat input to treat the file as an image
PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8ffff3f0005fe02fea7d6a4b40000000049454e44ae426082"
)


@pytest.fixture
def defaults_restored(admin, preserve):
    """Puts the model defaults and the default prompt suggestions back afterwards."""
    preserve(MODELS_CONFIG)
    with admin.client() as client:
        shown = client.get("/api/config").json()
    suggestions = shown.get("default_prompt_suggestions")
    localized = shown.get("default_prompt_suggestions_i18n") or {}
    yield
    with admin.client() as client:
        restored = client.post(
            "/api/v1/configs/suggestions", json={"suggestions": suggestions, "i18n": localized}
        )
    restored.raise_for_status()


def open_model_defaults(page: Page) -> Locator:
    page.goto("/admin/settings/models")
    settings = page.get_by_role("dialog")
    settings.get_by_role("button", name="Model Defaults").click()
    expect(settings.get_by_role("button", name="Model Capabilities")).to_be_visible()
    return settings


def open_section(settings: Locator, name: str) -> None:
    settings.get_by_role("button", name=name).click()


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Models configuration saved successfully")).to_be_visible()


def test_a_default_prompt_suggestion_shows_on_a_new_chat_and_sends(
    page_for, admin, make_user, upstream, defaults_restored
):
    title = f"Tide times {uuid.uuid4().hex[:6]}"
    content = f"When is high tide in the harbour today? {uuid.uuid4().hex[:6]}"
    admin_page = page_for(admin)
    settings = open_model_defaults(admin_page)
    open_section(settings, "Prompt Suggestions")
    settings.get_by_role("button", name="Add prompt suggestion").click()
    settings.get_by_role("textbox", name="Title", exact=True).last.fill(title)
    settings.get_by_role("textbox", name="Content").last.fill(content)
    save(admin_page, settings)

    page = page_for(make_user())
    upstream.queue(reply.text("At noon.", match=reply.answering(content)))
    page.get_by_role("listitem").filter(has_text=title).click()

    expect_reply(page, "At noon.")


def test_a_default_parameter_reaches_the_provider(
    page_for, admin, make_user, upstream, defaults_restored
):
    question = "How warm is the lake today?"
    admin_page = page_for(admin)
    settings = open_model_defaults(admin_page)
    open_section(settings, "Model Parameters")
    temperature = settings.get_by_text("Temperature", exact=True)
    temperature.locator("xpath=ancestor::div[.//button][1]").get_by_role(
        "button", name="Default"
    ).click()
    settings.get_by_role("spinbutton", name="Temperature").fill("0.3")
    save(admin_page, settings)

    page = page_for(make_user())
    upstream.queue(reply.text("Nineteen degrees.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Nineteen degrees.")

    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    assert sent["temperature"] == 0.3


def test_a_default_capability_switched_off_refuses_an_image(
    page_for, admin, make_user, defaults_restored
):
    admin_page = page_for(admin)
    settings = open_model_defaults(admin_page)
    open_section(settings, "Model Capabilities")
    vision = settings.get_by_role("checkbox", name="Vision")
    expect(vision).to_be_checked()
    vision.click()
    expect(vision).not_to_be_checked()
    save(admin_page, settings)

    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": "pier.png", "mimeType": "image/png", "buffer": PIXEL_PNG})

    expect(page.get_by_text("Selected model(s) do not support image inputs")).to_be_visible()
    expect(page.get_by_role("button", name="pier.png")).to_have_count(0)


def test_an_image_attaches_while_the_default_capabilities_allow_it(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": "pier.png", "mimeType": "image/png", "buffer": PIXEL_PNG})

    expect(page.get_by_role("button", name="pier.png")).to_be_visible()
