"""Regression: a document sent to Docling carried the full path it is stored under on the server.

Fix `f956f7adc` (open-webui/open-webui#30357, issue open-webui/open-webui#30352): the Docling
loader put the upload's storage path in the multipart `filename`, so every conversion told the
Docling server where Open WebUI keeps its uploads. It now sends only the file name.

Docling is a local service here; each upload is converted before the response returns.

Discriminates: passes on dev efe63bd34; with f956f7adc reverted in a backend copy the narrow test
fails (the filename is the absolute path under the data directory). The nearby test passes on
both.
"""

from __future__ import annotations

import re
import uuid

import pytest

from harness.docling_server import CONVERT, serve_docling

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
OCTET_STREAM = "application/octet-stream"


@pytest.fixture
def docling(admin, preserve, listener):
    """The admin's client with Docling as the extraction engine, served by `listener`."""
    preserve(RETRIEVAL_CONFIG)
    serve_docling(listener, {"status": "success", "document": {"md_content": "converted"}})
    with admin.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"CONTENT_EXTRACTION_ENGINE": "docling", "DOCLING_SERVER_URL": listener.base_url},
        )
        assert saved.status_code == 200, saved.text
        yield client


def _name_docling_was_sent(client, listener, filename: str) -> str:
    uploaded = client.post(
        "/api/v1/files/?process_in_background=false",
        files={"file": (filename, b"binary-ish", OCTET_STREAM)},
    )
    assert uploaded.status_code == 200, uploaded.text
    stored = client.get(f"/api/v1/files/{uploaded.json()['id']}").json()
    assert stored["data"].get("content") == "converted", stored["data"]
    [conversion] = listener.requests_to(CONVERT)
    names = re.findall(rb'filename="([^"]*)"', conversion.body)
    assert len(names) == 1, f"expected one file part, got {names}"
    return names[0].decode()


def test_docling_is_sent_the_file_name_without_the_storage_path(docling, listener, instance):
    filename = f"drawing-{uuid.uuid4().hex[:6]}.dxf"

    sent = _name_docling_was_sent(docling, listener, filename)

    assert "/" not in sent and "\\" not in sent, (
        f"Docling was sent {sent!r}, the path the upload is stored under on the server (#30352)"
    )
    assert str(instance.data_dir) not in sent
    assert sent.endswith(filename)


def test_a_name_with_spaces_reaches_docling_whole(docling, listener):
    filename = f"quarterly report {uuid.uuid4().hex[:6]}.dxf"

    sent = _name_docling_was_sent(docling, listener, filename)

    assert sent.endswith(filename)
