"""Regression: an upload with no content reports the real reason.

open-webui 0.11.4 fix `6786ae179`. Adding a web page without any text to a knowledge base failed
with a bare "Error uploading file": the storage layer refuses empty content with
`ERROR_MESSAGES.EMPTY_CONTENT`, and `upload_file_handler`'s generic except replaced that message.
The fix passes the empty-content message through as the 400 detail.

Twin of unit/security/test_empty_upload_reason.py.

Discriminates: passes on dev `bbfa876af`; with the handler's detail back to the generic message
the empty uploads answer "[ERROR: Error uploading file]".
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

GENERIC_REASON = "Error uploading file"


def _upload(client, content: bytes, process: bool = True, metadata: str | None = None):
    return client.post(
        "/api/v1/files/",
        params={"process": str(process).lower()},
        files={"file": ("page.txt", content, "text/plain")},
        data={"metadata": metadata} if metadata else None,
    )


@pytest.mark.parametrize("process", [True, False])
def test_an_empty_upload_says_the_content_is_empty(make_user, process):
    with make_user().client() as client:
        response = _upload(client, b"", process=process)

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "content provided is empty" in detail, (
        f"an empty upload answered {detail!r}, which hides why it failed (6786ae179)"
    )


def test_an_upload_with_content_still_succeeds(make_user):
    with make_user().client() as client:
        response = _upload(client, b"some page text", process=False)

    assert response.status_code == 200, response.text
    assert response.json()["meta"]["size"] == len(b"some page text")


def test_another_upload_failure_keeps_the_generic_reason(make_user):
    """Only the empty-content refusal is passed through; a malformed metadata list is not."""
    with make_user().client() as client:
        response = _upload(client, b"some page text", metadata="[1, 2]")

    assert response.status_code == 400, response.text
    assert GENERIC_REASON in response.json()["detail"]
