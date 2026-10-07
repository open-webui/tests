"""Regression: uploading a GGUF model to Ollama pushed the file but never created the model.

Issue open-webui/open-webui#31861, fix PR open-webui/open-webui#31862. In Admin > Settings >
Connections > Manage Ollama, File Mode asked Ollama for a streamed create and closed the connection
without reading the reply, and Ollama stops creating a model when its client disconnects. URL Mode
never created a model on the server at all: it relied on a create call of the dialog that failed.
Both routes now create the model after pushing the blob, named after the file without its extension,
and report it in their last event.

The Ollama stand-in keeps a pushed blob and adds a model only for a create that stays connected to
the end of its stream (or asked for `stream: false`). The file URL is served by
`harness/model_hub.py` on an instance of its own, as the route only downloads from Hugging Face or
GitHub. Twin of e2e/models/test_ollama_gguf_upload.py.

The URL Mode download used to hash and push the file while its last write was still in the file
buffer, so a file whose last chunk arrives on its own and is under 8 KiB reached Ollama without that
chunk (open-webui/open-webui#31956), fixed in dev 1e7561450.

Discriminates: passes on dev 1711059db; the short last chunk test fails on dev c01d4825a, before
1e7561450. With 1b2ceedd6 reverted in a backend copy, the File Mode upload leaves no model on the
server and the URL Mode download never calls `/api/create`.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from harness.actors import admin_of
from harness.model_hub import model_hub_env, serving_model_hub
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

UPLOADED = b"GGUF the file picked in the dialog"
# sent in one piece larger than the download's write buffer, so none of it waits there
DOWNLOADED = b"GGUF" + bytes(range(256)) * 100
DOWNLOAD_PATH = "/ggml-org/tiny-gguf/resolve/main/tiny-url.gguf"
LARGE_BODY, SHORT_TAIL = b"GGUF" + bytes(64 * 1024), b"the last few bytes of the file"
TAILED_PATH = "/ggml-org/tiny-gguf/resolve/main/tiny-tail.gguf"


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _events(streamed) -> list[dict]:
    lines = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
    return [json.loads(line.removeprefix("data: ")) for line in lines]


def _listed(client) -> list[str]:
    tags = client.get("/ollama/api/tags/0")  # the uncached list the settings page reads
    assert tags.status_code == 200, tags.text
    return [model["name"] for model in tags.json()["models"]]


@pytest.fixture(scope="module")
def hub():
    with serving_model_hub() as serving:
        serving.files[DOWNLOAD_PATH] = DOWNLOADED
        serving.files[TAILED_PATH] = [LARGE_BODY, SHORT_TAIL]
        yield serving


@pytest.fixture
def ollama(instance_with, hub, preserve, listener):
    managed = instance_with(model_hub_env(hub))
    preserve(OLLAMA_CONFIG, on=managed)
    server = serve_ollama(listener)
    with admin_of(managed).client() as client:
        connect_ollama(client, listener)
        yield server, client


def test_an_uploaded_model_file_creates_the_model(ollama):
    server, client = ollama

    uploaded = client.post(
        "/ollama/models/upload/0",
        files={"file": ("tiny-file.gguf", UPLOADED, "application/octet-stream")},
    )

    assert uploaded.status_code == 200, uploaded.text
    assert "tiny-file:latest" in _listed(client)
    assert _events(uploaded)[-1] == {
        "done": True,
        "blob": _digest(UPLOADED),
        "name": "tiny-file.gguf",
        "model_created": "tiny-file",
    }
    assert server.blobs == {_digest(UPLOADED): UPLOADED}
    assert server.sent("/api/create") == [
        {"model": "tiny-file", "files": {"tiny-file.gguf": _digest(UPLOADED)}, "stream": False}
    ]


def test_a_model_file_downloaded_by_url_creates_the_model(ollama, hub):
    server, client = ollama

    downloaded = client.post("/ollama/models/download/0", json={"url": hub.url(DOWNLOAD_PATH)})

    assert downloaded.status_code == 200, downloaded.text
    assert "tiny-url:latest" in _listed(client)
    assert _events(downloaded)[-1] == {
        "done": True,
        "blob": _digest(DOWNLOADED),
        "name": "tiny-url.gguf",
        "model_created": "tiny-url",
    }
    assert hub.downloads == [DOWNLOAD_PATH]
    assert server.blobs == {_digest(DOWNLOADED): DOWNLOADED}
    assert server.sent("/api/create") == [
        {"model": "tiny-url", "files": {"tiny-url.gguf": _digest(DOWNLOADED)}, "stream": False}
    ]


def test_a_model_file_whose_last_chunk_is_short_reaches_ollama_whole(ollama, hub):
    server, client = ollama

    downloaded = client.post("/ollama/models/download/0", json={"url": hub.url(TAILED_PATH)})

    assert downloaded.status_code == 200, downloaded.text
    pushed = b"".join(server.blobs.values())
    assert pushed == LARGE_BODY + SHORT_TAIL, (
        f"the blob lost the file's last {len(LARGE_BODY + SHORT_TAIL) - len(pushed)} bytes: the "
        "download is hashed and pushed before its last write leaves the file buffer (#31956)"
    )
