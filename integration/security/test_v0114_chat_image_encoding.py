"""Regression: a chat image whose host forces a Content-Encoding must reach the model as its link.

open-webui 0.11.4 fix `6fb68e43c` (PR #29623): before a chat goes to the model, every image link
in it is fetched and inlined as a data URL (`get_image_base64_from_url`). The fetch sent aiohttp's
default Accept-Encoding and never looked at the response's Content-Encoding, so an image from a
host that compresses whatever it is asked (an S3 or MinIO object stored with that metadata, a
compressing proxy) was inlined from a body the client had already decoded, and the reply broke.
The fetch now asks for `Accept-Encoding: identity` and leaves an image that still comes back
encoded as its original link, the way an unreachable image already was.

The image host is a local listener on an instance that may fetch from 127.0.0.1; the scripted
provider records the image part the model was sent.

Twin of unit/security/test_v0114_chat_image_encoding.py.

Discriminates: passes on dev bbfa876af, fails with `6fb68e43c` reverted (the fetch advertises
gzip, deflate and br, and a gzip or deflate encoded image reaches the model as a data URL).
"""

from __future__ import annotations

import base64
import gzip
import time
import uuid
import zlib

import pytest

from harness.actors import create_user
from harness.chat import send_message, wait_for_reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LOCAL_FETCH = {"ENABLE_LOCAL_WEB_FETCH": "true"}
IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + b"not really a picture, but the bytes are what matter"
ENCODERS = {"gzip": gzip.compress, "deflate": zlib.compress}


@pytest.fixture
def local(instance_with):
    return instance_with(LOCAL_FETCH)


@pytest.fixture
def chatter(local):
    return create_user(local)


def serve_image(listener, content_encoding: str | None, content_type: str = "image/png") -> str:
    """The URL of an image whose response carries `content_encoding` (encoded when we can)."""
    path = f"/images/{uuid.uuid4().hex}.png"
    headers = {"Content-Type": content_type}
    body = IMAGE_BYTES
    if content_encoding is not None:
        headers["Content-Encoding"] = content_encoding
        body = ENCODERS.get(content_encoding.lower(), lambda raw: raw)(IMAGE_BYTES)
    listener.route("GET", path, (200, headers, body))
    return f"{listener.base_url}{path}"


def image_the_model_was_sent(actor, local, image_url: str) -> str:
    """Attach the image the way the web client stores one on a user message, and chat."""
    user_message = {
        "id": str(uuid.uuid4()),
        "parentId": None,
        "role": "user",
        "content": "what is in this picture?",
        "files": [{"type": "image", "url": image_url}],
        "models": [MOCK_MODEL_ID],
        "timestamp": int(time.time()),
    }
    with actor.client() as client:
        turn = send_message(client, user_message["content"], user_message=user_message)
        wait_for_reply(client, turn)
    prompt = local.upstream.chat_requests()[-1]["messages"][-1]["content"]
    image_parts = [part for part in prompt if part.get("type") == "image_url"]
    assert len(image_parts) == 1, f"the model was not sent the attached image: {prompt}"
    return image_parts[0]["image_url"]["url"]


def data_url(content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(IMAGE_BYTES).decode()}"


def test_a_gzip_encoded_image_reaches_the_model_as_its_link(chatter, local, listener):
    image_url = serve_image(listener, "gzip")

    sent = image_the_model_was_sent(chatter, local, image_url)

    assert sent == image_url, (
        "a gzip-encoded image was inlined from the body the client had already decoded, "
        f"instead of reaching the model as its link (#29623); the model got {sent[:80]!r}"
    )


def test_the_image_fetch_asks_for_an_unencoded_body(chatter, local, listener):
    image_url = serve_image(listener, None)
    image_path = image_url.removeprefix(listener.base_url)

    image_the_model_was_sent(chatter, local, image_url)

    fetches = listener.requests_to(image_path)
    assert fetches, "the chat never fetched the image, so nothing here was exercised"
    advertised = {name.lower(): value for name, value in fetches[-1].headers.items()}
    assert advertised.get("accept-encoding") == "identity", (
        "the image fetch did not ask for an identity-encoded body (#29623); it advertised "
        f"{advertised.get('accept-encoding')!r}"
    )


@pytest.mark.parametrize("content_encoding", ["deflate", "br", "zstd", "GZIP", "identity, gzip"])
def test_every_real_content_encoding_is_left_as_a_link(chatter, local, listener, content_encoding):
    image_url = serve_image(listener, content_encoding)

    sent = image_the_model_was_sent(chatter, local, image_url)

    assert sent == image_url, (
        f"an image sent with Content-Encoding {content_encoding!r} was inlined"
    )


@pytest.mark.parametrize("content_encoding", [None, "", "identity", "IDENTITY"])
def test_an_unencoded_image_is_still_inlined(chatter, local, listener, content_encoding):
    image_url = serve_image(listener, content_encoding, content_type="image/webp")

    sent = image_the_model_was_sent(chatter, local, image_url)

    assert sent == data_url("image/webp"), (
        f"an image sent with Content-Encoding {content_encoding!r} must still be inlined with "
        f"its own content type; the model got {sent[:80]!r}"
    )
