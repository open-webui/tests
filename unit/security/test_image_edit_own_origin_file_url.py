"""Regression: image edit must read its own files directly, not fetch them over HTTP.

open-webui 0.11.2 fix `50413f348` (withheld security advisory): `image_edits` in
`open_webui/routers/images.py` sent an absolute URL pointing at Open WebUI's own
`/api/v1/files/<id>/content` endpoint through the outbound HTTP client, so the
server made a request back at itself to read a file it can open locally.

The fix recognises such a URL by netloc plus path and recurses with
`load_url_image(parsed.path)`, which resolves the file id through
`get_file_content_by_id` and its ownership checks.

Discriminates: passes on v0.11.2, fails on v0.11.1 (the own-origin URL is fetched
through the SSRF-validating HTTP path, so the outbound session is used).

No network: the outbound session is a fake that records every request.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from fastapi.responses import FileResponse

pytestmark = pytest.mark.regression

BASE_URL = "http://localhost:8080/"
FILE_ID = "file-abc"
OWN_ORIGIN_URL = f"http://localhost:8080/api/v1/files/{FILE_ID}/content"
FILE_BYTES = b"\x89PNG\r\n\x1a\n local image bytes"
REMOTE_BYTES = b"bytes fetched over the network"


class FakeResponse:
    def __init__(self) -> None:
        self.headers = {"content-type": "image/jpeg"}

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return REMOTE_BYTES

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class FakeSession:
    def __init__(self) -> None:
        self.requested_urls: list[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.requested_urls.append(url)
        return FakeResponse()

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


@pytest.fixture(scope="session")
def images_module(owui_module):
    """`open_webui.routers.images` (image_edits)."""
    return owui_module("open_webui.routers.images")


@pytest.fixture
def edit(images_module, monkeypatch, tmp_path):
    """Drive the real image_edits with every outbound boundary faked out."""
    stored = tmp_path / "stored.png"
    stored.write_bytes(FILE_BYTES)

    session = FakeSession()
    validated: list[str] = []
    served_file_ids: list[str] = []

    async def fake_image_config():
        # An engine nothing matches: image_edits loads the images then falls through.
        return SimpleNamespace(
            IMAGE_EDIT_SIZE=None, IMAGE_EDIT_MODEL="edit-model", IMAGE_EDIT_ENGINE="none"
        )

    async def fake_get_file_content(file_id, user=None, *args, **kwargs):
        served_file_ids.append(file_id)
        return FileResponse(str(stored))

    monkeypatch.setattr(images_module, "get_image_config", fake_image_config)
    monkeypatch.setattr(images_module, "get_ssrf_safe_session", lambda: session)
    monkeypatch.setattr(images_module, "validate_url", validated.append)
    monkeypatch.setattr(images_module, "get_file_content_by_id", fake_get_file_content)

    async def _edit(image, base_url=BASE_URL):
        form_data = images_module.EditImageForm(image=image, prompt="make it blue")
        request = SimpleNamespace(base_url=base_url)
        user = SimpleNamespace(id="user-1", role="user")
        await images_module.image_edits(request, form_data, user=user)
        return SimpleNamespace(
            image=form_data.image,
            session=session,
            validated=validated,
            served_file_ids=served_file_ids,
        )

    return _edit


def local_data_url(content_type: str = "image/png") -> str:
    return f"data:{content_type};base64,{base64.b64encode(FILE_BYTES).decode()}"


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_own_origin_file_url_never_reaches_the_http_client(edit):
    outcome = await edit(OWN_ORIGIN_URL)

    assert outcome.session.requested_urls == []
    assert outcome.validated == []


@pytest.mark.asyncio
async def test_own_origin_file_url_is_served_from_the_file_store(edit):
    outcome = await edit(OWN_ORIGIN_URL)

    assert outcome.served_file_ids == [FILE_ID]
    assert outcome.image == local_data_url()


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin,url",
    [
        ("https://chat.example.com/", f"https://chat.example.com/api/v1/files/{FILE_ID}/content"),
        ("http://localhost:8080/", f"http://localhost:8080/api/v1/files/{FILE_ID}/content?x=1"),
        ("http://localhost:8080/", f"http://localhost:8080/api/v1/files/{FILE_ID}/content/head"),
    ],
)
@pytest.mark.asyncio
async def test_every_shape_of_own_origin_content_url_stays_local(edit, origin, url):
    outcome = await edit(url, base_url=origin)

    assert outcome.session.requested_urls == []
    assert outcome.served_file_ids == [FILE_ID]


@pytest.mark.asyncio
async def test_own_origin_url_in_a_list_stays_local(edit):
    outcome = await edit([OWN_ORIGIN_URL, "data:image/png;base64,AAAA"])

    assert outcome.session.requested_urls == []
    assert outcome.image == [local_data_url(), "data:image/png;base64,AAAA"]


@pytest.mark.asyncio
async def test_foreign_host_with_the_same_path_is_not_treated_as_own_origin(edit):
    foreign = f"https://evil.example.com/api/v1/files/{FILE_ID}/content"
    outcome = await edit(foreign)

    assert outcome.session.requested_urls == [foreign]
    assert outcome.served_file_ids == []


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/static/logo.png",
        "http://localhost:8080/api/v1/files/",
        f"http://localhost:8080/api/v1/files/{FILE_ID}",
    ],
)
@pytest.mark.asyncio
async def test_own_origin_url_outside_the_content_route_still_goes_over_http(edit, url):
    """Same host is not enough: matching on netloc alone would send any path at the file store."""
    outcome = await edit(url)

    assert outcome.session.requested_urls == [url]
    assert outcome.served_file_ids == []


@pytest.mark.asyncio
async def test_data_url_is_passed_through_untouched(edit):
    outcome = await edit("data:image/png;base64,AAAA")

    assert outcome.image == "data:image/png;base64,AAAA"
    assert outcome.session.requested_urls == []
    assert outcome.served_file_ids == []


@pytest.mark.asyncio
async def test_bare_file_id_is_resolved_through_the_file_store(edit):
    outcome = await edit(FILE_ID)

    assert outcome.served_file_ids == [FILE_ID]
    assert outcome.image == local_data_url()


@pytest.mark.asyncio
async def test_relative_content_path_is_resolved_through_the_file_store(edit):
    outcome = await edit(f"/api/v1/files/{FILE_ID}/content")

    assert outcome.served_file_ids == [FILE_ID]
    assert outcome.image == local_data_url()


@pytest.mark.asyncio
async def test_external_url_is_ssrf_validated_and_fetched(edit):
    outcome = await edit("https://example.com/photo.jpg")

    assert outcome.validated == ["https://example.com/photo.jpg"]
    assert outcome.session.requested_urls == ["https://example.com/photo.jpg"]
    assert outcome.image == f"data:image/jpeg;base64,{base64.b64encode(REMOTE_BYTES).decode()}"
