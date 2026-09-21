"""Regression: an upload with no content reports the real reason.

open-webui 0.11.4 fix `6786ae179`: adding a web address to a knowledge base
that came back without any text failed with a bare "Error uploading file".
The web fetch path hands the page body to `upload_file_handler`; when the
body is empty, `LocalStorageProvider.upload_file` raises
`ValueError(ERROR_MESSAGES.EMPTY_CONTENT)`, and the handler's generic
except turned it into `ERROR_MESSAGES.DEFAULT('Error uploading file')`,
hiding the actual reason from the caller. The fix passes the
`EMPTY_CONTENT` message through as the HTTP error detail, so the UI can
show it and remove the pending row (the row removal is frontend and is not
pinned here).

The test drives the real `upload_file_handler` with the storage boundary
raising exactly what the local provider raises for empty content.

Discriminates: passes on v0.11.4, fails on v0.11.3 (the handler answers
"[ERROR: Error uploading file]" instead of the empty-content message).
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException

pytestmark = pytest.mark.regression


@pytest.fixture
def files_router(owui_module):
    return owui_module("open_webui.routers.files")


def _upload(file_bytes=b"real bytes"):
    return SimpleNamespace(
        filename="page.txt",
        content_type="text/plain",
        file=SimpleNamespace(read=lambda: file_bytes),
    )


async def _upload_page(router, file):
    return await router.upload_file_handler(
        request=None,
        file=file,
        metadata={"source_url": "https://example.test/empty"},
        process=True,
        process_in_background=False,
        user=SimpleNamespace(id="user-1", role="user", email="e", name="n"),
        background_tasks=None,
        db=None,
    )


@contextmanager
def _boundary(router, upload_result):
    config = SimpleNamespace(get=AsyncMock(return_value=None))
    patches = [
        patch.object(router.Config, "get", config.get),
        patch.object(router.Storage, "upload_file", upload_result),
        patch.object(router, "publish_event", AsyncMock()),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


# ── Narrow: the bug itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_page_upload_reports_the_empty_content_reason(files_router):
    """The exact bug: the storage layer says "content is empty", the handler
    must pass that wording on rather than masking it."""
    error_messages = files_router.ERROR_MESSAGES

    def empty_content(*args, **kwargs):
        raise ValueError(error_messages.EMPTY_CONTENT)

    with _boundary(files_router, empty_content):
        with pytest.raises(HTTPException) as raised:
            await _upload_page(files_router, _upload(b""))

    assert raised.value.status_code == 400
    assert str(raised.value.detail) == str(error_messages.EMPTY_CONTENT), (
        f"an empty page upload answered {raised.value.detail!r} instead of the "
        "empty-content reason, so the user could not tell why it failed (#6786ae179)"
    )


@pytest.mark.asyncio
async def test_other_upload_errors_keep_the_generic_reason(files_router):
    """Nearby: an unrelated failure still gets the generic message."""
    error_messages = files_router.ERROR_MESSAGES

    def broken_storage(*args, **kwargs):
        raise RuntimeError("disk on fire")

    with _boundary(files_router, broken_storage):
        with pytest.raises(HTTPException) as raised:
            await _upload_page(files_router, _upload(b"x"))

    assert raised.value.status_code == 400
    assert str(raised.value.detail) == str(error_messages.DEFAULT("Error uploading file")), (
        f"an unrelated upload error changed its wording to {raised.value.detail!r}"
    )


@pytest.mark.asyncio
async def test_a_value_error_with_a_different_message_stays_generic(files_router):
    """Nearby: only the exact empty-content sentinel is special-cased, so a
    ValueError from somewhere else is not misreported as empty content."""
    error_messages = files_router.ERROR_MESSAGES

    def other_value_error(*args, **kwargs):
        raise ValueError("something else entirely")

    with _boundary(files_router, other_value_error):
        with pytest.raises(HTTPException) as raised:
            await _upload_page(files_router, _upload(b"x"))

    assert str(raised.value.detail) == str(error_messages.DEFAULT("Error uploading file")), (
        f"a ValueError with another message was reported as {raised.value.detail!r}"
    )
