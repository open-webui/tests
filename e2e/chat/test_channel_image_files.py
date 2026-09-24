"""An image a model generates in a channel shows in that channel.

0.11.2 `aeb126b95`: a channel turn passed `channel:<id>` along as a chat id and handed on only
the image URL, so the generated file never reached the channel message and the channel showed
the model's words without the picture. The model is mentioned through the API; the page is
what a member opening the channel sees.

Twin of unit/chat/test_channel_image_files.py.

Discriminates: passes on dev bbfa876af; with the channel emitter ignoring `files` events the
image never appears in the channel.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import channel_chat
from harness import upstream as reply
from harness.image_engines import IMAGES_CONFIG, save_image_settings

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def test_a_generated_image_shows_in_the_channel(
    page_for, admin, admin_token, upstream, listener, preserve
):
    preserve("admin_config", IMAGES_CONFIG)
    upstream.queue(reply.tool_call("generate_image", {"prompt": "a cat"}), reply.text("a cat"))
    with admin.client() as client, channel_chat.drawing_model(client) as model:
        channel_chat.enable_channels(client, reply_mode="channel")
        save_image_settings(client, **channel_chat.serve_openai_images(listener))
        channel_id, _ = channel_chat.mention(client, model, "draw a cat")
        channel_chat.model_reply(client, channel_id)

        page = page_for(admin)
        page.goto(f"/channels/{channel_id}")

        expect(page.get_by_text("a cat", exact=True)).to_be_visible()
        expect(page.get_by_role("img", name="generated-image.png")).to_be_visible()
