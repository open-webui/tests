"""Regression: an iPhone photo (HEIC) reached a vision model as HEIC, which it refused.

Fix `6c322941d` (PR #31649, issue #28411). The chat input converted a photo to JPEG only when
the browser typed it exactly `image/heic`; Firefox types it `image/heif` and some browsers leave
the type empty, so those photos went up unconverted, an empty type even as a document. When the
conversion did run, the JPEG still went up under the HEIC name and type, so the model was told
it was HEIC. A HEIC or HEIF photo is now converted whatever type the browser reports and goes up
as a `.jpg` typed `image/jpeg`. The photo is attached through the chat's Upload Files menu and
the test reads the image the scripted provider was sent.

Discriminates: passes on the 015dbc861 build; with `6c322941d` reverted in the chat input (the
015dbc861 mutation build) both cases go red: the photo goes up under its HEIC name and type.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import base64
import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, expect_reply, open_input_menu, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

# a 32x32 red photo in HEIC, 8-bit HEVC, written by ImageMagick
RED_HEIC = base64.b64decode(
    "AAAAHGZ0eXBoZWljAAAAAG1pZjFoZWljbWlhZgAAAWltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABwaWN0AAAAAAAA"
    "AAAAAAAAAAAAACJpbG9jAAAAAERAAAEAAQAAAAABjQABAAAAAAAAAB4AAAAjaWluZgAAAAAAAQAAABVpbmZlAgAA"
    "AAABAABodmMxAAAAAA5waXRtAAAAAAABAAAA6WlwcnAAAADKaXBjbwAAAHZodmNDAQNwAAAAAAAAAAAAHvAA/P34"
    "+AAADwNgAAEAGEABDAH//wNwAAADAJAAAAMAAAMAHroCQGEAAQAqQgEBA3AAAAMAkAAAAwAAAwAeoCCBBZbq5Ka5"
    "uAhoMCAAAAMDIAAAAwAhYgABAAZEAcFzwIkAAAAUaXNwZQAAAAAAAABAAAAAQAAAAChjbGFwAAAAIAAAAAEAAAAg"
    "AAAAAf///+AAAAAC////4AAAAAIAAAAQcGl4aQAAAAADCAgIAAAAF2lwbWEAAAAAAAAAAQABBIECBIMAAAAmbWRh"
    "dAAAABooAa8TgPgQ4af//E0o//p1zw/TzHvzopDPgA=="
)

JPEG_MAGIC = b"\xff\xd8\xff"


def attach(chat: ChatSurface, name: str, content: bytes, mime_type: str) -> None:
    expect(chat_input(chat)).to_be_visible()
    menu = open_input_menu(chat)
    with chat.page.expect_file_chooser() as chooser:
        menu.get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": content})


def images_sent(upstream, prompt: str) -> list[str]:
    answered = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    question = reply.conversation(answered[-1])[-1]["content"]
    parts = question if isinstance(question, list) else []
    return [part["image_url"]["url"] for part in parts if part.get("type") == "image_url"]


# Chromium types a file it is given untyped from its extension, so an empty type cannot be played
@pytest.mark.parametrize("reported_type", ["image/heic", "image/heif"], ids=["heic", "heif"])
def test_a_heic_photo_reaches_the_model_as_a_jpeg(chat_for, make_user, upstream, reported_type):
    account = make_user()
    chat = chat_for(account)
    prompt = f"what colour is this photo ({reported_type})?"
    upstream.queue(reply.text("It is red.", match=reply.answering(prompt)))

    attach(chat, "IMG_0001.HEIC", RED_HEIC, reported_type)
    expect(chat.panel.get_by_role("button", name="Show image preview")).to_be_visible()
    send(chat, prompt)
    expect_reply(chat, "It is red.")

    with account.client() as client:
        uploaded = client.get("/api/v1/files/", params={"content": False}).json()["items"]
    assert [(file["filename"], file["meta"]["content_type"]) for file in uploaded] == [
        ("IMG_0001.jpg", "image/jpeg")
    ], f"the photo did not go up as a JPEG (#31649): {uploaded}"

    images = images_sent(upstream, prompt)
    assert len(images) == 1, f"the provider got {len(images)} images for the photo (#31649)"
    assert images[0].startswith("data:image/jpeg;base64,"), (
        f"the photo was not sent as a JPEG (#31649): {images[0][:40]}"
    )
    sent_bytes = base64.b64decode(re.sub(r"^data:[^,]*,", "", images[0]))
    assert sent_bytes.startswith(JPEG_MAGIC)
