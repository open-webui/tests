"""Journey: image compression set by a person or by the admin shrinks the image the model is sent.

Image Compression switched on in Settings > Interface, with a maximum size set in its Manage
dialog, scales an image attached to a chat down to that size before it is sent. The admin's
Image Compression Width and Height in Admin Settings > Documents do the same for every account.
Without either, the model gets the image at its own size.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose chat input ignores both
compression settings, the person's and the admin's tests fail while the plain one passes.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "What colour is this buoy?"


def buoy_png(width: int = 96, height: int = 64) -> bytes:
    picture = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(picture, format="PNG")
    return picture.getvalue()


def attach_and_ask(page: Page, upstream) -> tuple[int, int]:
    """The size of the image the model was sent with the question."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": "buoy.png", "mimeType": "image/png", "buffer": buoy_png()})
    # an attached image shows as a chip or a thumbnail, both with a remove button
    expect(page.get_by_label("Remove file").first).to_be_attached()
    upstream.queue(reply.text("It is red.", match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, "It is red.")
    sent = next(filter(reply.answering(QUESTION), upstream.chat_requests()))
    [image] = [
        part["image_url"]["url"]
        for message in sent["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    return Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1]))).size


def test_an_image_reaches_the_model_at_its_own_size(page_for, make_user, upstream):
    assert attach_and_ask(page_for(make_user()), upstream) == (96, 64)


def test_a_persons_image_compression_shrinks_the_image_sent(page_for, make_user, upstream):
    page = page_for(make_user())
    page.goto("/?settings=interface")
    tab = page.locator("#tab-interface")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        tab.get_by_role("switch", name="Image Compression", exact=True).click()
    tab.get_by_role("button", name="Open Modal To Manage Image Compression").click()
    dialog = page.get_by_role("dialog").filter(has_text="Image Max Compression Size")
    dialog.get_by_placeholder("Width").fill("24")
    dialog.get_by_placeholder("Height").fill("16")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        dialog.get_by_role("button", name="Save").click()

    width, height = attach_and_ask(page, upstream)

    assert width <= 24 and height <= 16, (width, height)


def test_the_admins_image_compression_shrinks_every_accounts_image(
    page_for, make_user, upstream, preserve
):
    preserve(RETRIEVAL_CONFIG)
    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/settings/documents")
    settings = admin_page.get_by_role("dialog")
    for label in ("Image Compression Width", "Image Compression Height"):
        field = settings.get_by_text(label, exact=True)
        field.locator("xpath=following-sibling::div//input").fill("20")
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    width, height = attach_and_ask(page_for(make_user()), upstream)

    assert width <= 20 and height <= 20, (width, height)
