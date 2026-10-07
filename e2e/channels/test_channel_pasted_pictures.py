"""Journey: a picture pasted into the channel input posts with the message, compressed on request.

A picture pasted with Control+V into a channel's input is attached and posts with the message,
and the other member sees it on that message. Image Compression in Settings > Interface shrinks
it only when Compress Images in Channels is on as well; with that switch off the channel gets the
picture at its own size. The size is read from the picture the other member is shown.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose channel input ignores
the person's compression settings, the compressed case fails; one whose paste handler drops
pictures fails both cases.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.channel_quotes import enable_channels, group_channel
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]

COPY_PICTURE = """async (encoded) => {
    const bytes = Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
    const picture = new Blob([bytes], { type: 'image/png' });
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': picture })]);
}"""


def buoy_png() -> str:
    picture = io.BytesIO()
    Image.new("RGB", (96, 64), (200, 30, 30)).save(picture, format="PNG")
    return base64.b64encode(picture.getvalue()).decode()


@pytest.fixture
def channels_on(admin, preserve) -> None:
    preserve("admin_config")
    enable_channels(admin)


def open_channel(page_for, account: Actor, channel_id: str) -> Page:
    page = page_for(account, permissions=CLIPBOARD)
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def message(page: Page, text: str) -> Locator:
    posted = page.locator("[id^='message-']:not(#message-input-container)")
    return posted.filter(has_text=text).first


@pytest.mark.parametrize(
    ("in_channels", "expected_size"),
    [(False, [96, 64]), (True, [24, 16])],
    ids=["compression-outside-channels", "compression-in-channels"],
)
def test_a_pasted_picture_posts_at_the_size_the_persons_compression_asks_for(
    channels_on, make_user, page_for, in_channels, expected_size
):
    sender, member = make_user(), make_user()
    compression = {
        "imageCompression": True,
        "imageCompressionSize": {"width": 24, "height": 16},
        "imageCompressionInChannels": in_channels,
    }
    with sender.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": compression})
    assert saved.status_code == 200, saved.text
    channel_id = group_channel(sender, member)
    page = open_channel(page_for, sender, channel_id)

    page.evaluate(COPY_PICTURE, buoy_png())
    chat_input(page).click()
    page.keyboard.press("Control+V")
    # the spinner is the only sign the upload is still running
    expect(page.locator("#message-input-container .spinner_ajPY")).to_have_count(0)
    page.keyboard.type("the new buoy")
    page.keyboard.press("Enter")
    expect(message(page, "the new buoy")).to_be_visible()

    member_page = open_channel(page_for, member, channel_id)
    picture = message(member_page, "the new buoy").locator("img").last
    expect(picture).to_be_visible()
    expect(picture).to_have_js_property("complete", True)
    assert picture.evaluate("image => [image.naturalWidth, image.naturalHeight]") == expected_size
