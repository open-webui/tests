"""Regression: a remote image URL in a chat message must not be downloaded without a limit.

open-webui 0.11.2 fix `756241b34` (withheld security advisory). Before a chat reaches the model,
`convert_url_images_to_base64` inlines every `image_url` part as a data URL through
`get_image_base64_from_url`, which did `await response.read()`: the whole body of any URL a user
could name was pulled into memory and encoded. The fix streams the body in 64 KiB chunks against
the upload size limit (`FILE_MAX_SIZE`, in megabytes, unset meaning no limit) and gives up the
moment the running total passes it, so the part goes to the model as the plain URL.

Twin of unit/security/test_image_url_download_limit.py.

Discriminates: passes on dev `bbfa876af`; with the chunked read replaced by `response.read()`
the oversized image reaches the model inlined and the server downloads the whole body.
"""

from __future__ import annotations

import base64
import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from harness.instance import free_port
from harness.listener import text_answer
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MB = 1024 * 1024
LIMIT_MB = 1
LOCAL_FETCH_WITH_LIMIT = {"ENABLE_LOCAL_WEB_FETCH": "true", "RAG_FILE_MAX_SIZE": str(LIMIT_MB)}


@pytest.fixture(scope="module")
def limited(instance_with):
    return instance_with(LOCAL_FETCH_WITH_LIMIT)


def _image_answer(size: int, content_type: str = "image/png"):
    return 200, {"Content-Type": content_type}, b"x" * size


def _image_part_sent_to_the_model(instance, image_url: str) -> str:
    instance.upstream.reset()
    content = [
        {"type": "text", "text": "what is in this picture?"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    with instance.client() as client:
        response = client.post(
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": content}],
                "stream": False,
            },
        )
    assert response.status_code == 200, response.text
    sent_content = instance.upstream.chat_requests()[-1]["messages"][-1]["content"]
    (image_part,) = [part for part in sent_content if part["type"] == "image_url"]
    return image_part["image_url"]["url"]


def _is_inlined(sent_url: str) -> bool:
    return sent_url.startswith("data:")


@contextlib.contextmanager
def _file_max_size(instance, value: str) -> Iterator[None]:
    with instance.client() as client:
        changed = client.post("/api/v1/retrieval/config/update", json={"FILE_MAX_SIZE": value})
        assert changed.status_code == 200, changed.text
        try:
            yield
        finally:
            client.post("/api/v1/retrieval/config/update", json={"FILE_MAX_SIZE": LIMIT_MB})


@contextlib.contextmanager
def _streamed_image(total_bytes: int) -> Iterator[tuple[str, dict]]:
    """Serves one large image in chunks and counts what the client took before hanging up."""
    delivery = {"bytes": 0, "done": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(total_bytes))
            self.end_headers()
            chunk = b"x" * (64 * 1024)
            try:
                while delivery["bytes"] < total_bytes:
                    self.wfile.write(chunk)
                    delivery["bytes"] += len(chunk)
            except OSError:
                pass  # the client hung up
            finally:
                delivery["done"].set()

    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}/huge.png", delivery
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("content_type", ["image/png", "image/svg+xml", "text/html"])
def test_an_image_over_the_limit_reaches_the_model_as_its_url(limited, listener, content_type):
    listener.route("GET", "/big", _image_answer(2 * MB, content_type))
    image_url = f"{listener.base_url}/big"

    sent = _image_part_sent_to_the_model(limited, image_url)

    assert not _is_inlined(sent), (
        f"a 2 MB {content_type} body under a {LIMIT_MB} MB limit was downloaded whole and "
        "inlined, so any URL a user names is buffered into server memory"
    )
    assert sent == image_url


def test_the_download_stops_once_the_limit_is_passed(limited):
    total = 32 * MB
    with _streamed_image(total) as (image_url, delivery):
        sent = _image_part_sent_to_the_model(limited, image_url)
        assert delivery["done"].wait(10), "the image download never ended"

    assert not _is_inlined(sent), "a 32 MB image was inlined past the limit"
    assert delivery["bytes"] < total // 2, (
        f"the server took {delivery['bytes'] // MB} MB of a {total // MB} MB image before "
        f"refusing it; the body was read whole instead of stopping past {LIMIT_MB} MB"
    )


def test_an_image_one_byte_over_the_limit_is_not_inlined(limited, listener):
    listener.route("GET", "/over", _image_answer(LIMIT_MB * MB + 1))
    image_url = f"{listener.base_url}/over"

    assert not _is_inlined(_image_part_sent_to_the_model(limited, image_url)), (
        "an image one byte over the limit was inlined"
    )


def test_an_image_exactly_at_the_limit_is_inlined(limited, listener):
    listener.route("GET", "/exact", _image_answer(LIMIT_MB * MB))

    sent = _image_part_sent_to_the_model(limited, f"{listener.base_url}/exact")

    assert sent.startswith("data:image/png;base64,")


def test_a_small_image_is_inlined_with_its_content_type(limited, listener):
    body = b"small image bytes"
    listener.route("GET", "/small.webp", text_answer(body.decode(), content_type="image/webp"))

    sent = _image_part_sent_to_the_model(limited, f"{listener.base_url}/small.webp")

    assert sent == f"data:image/webp;base64,{base64.b64encode(body).decode()}"


def test_without_a_limit_a_large_image_is_inlined(limited, listener):
    listener.route("GET", "/big", _image_answer(2 * MB))

    with _file_max_size(limited, ""):
        sent = _image_part_sent_to_the_model(limited, f"{listener.base_url}/big")

    assert sent.startswith("data:image/png;base64,")
    assert len(base64.b64decode(sent.split(",", 1)[1])) == 2 * MB


def test_a_private_address_is_still_refused_without_local_fetch(instance, listener):
    listener.route("GET", "/small.png", _image_answer(1024))
    image_url = f"{listener.base_url}/small.png"

    assert _image_part_sent_to_the_model(instance, image_url) == image_url
    assert listener.received == [], "a loopback image was fetched with local web fetch off"
