"""Regression: a file fetched from a link must not be buffered whole in memory.

open-webui 0.11.1 fix `6b438f1a7` (PR #28945): the binary branch of the web
fetch, `_extract_text_from_binary_response` in
`open_webui/retrieval/utils.py`, wrote `response.content` into a temp file.
`requests` materialises that attribute by reading the entire body into memory
first, and no size limit was applied, so any link the user attached could push
an arbitrarily large download through the server's RAM. The temp file was also
created outside the `try`, so a download that died partway left it on disk.

The fix streams the body in 64 KiB blocks through `iter_content`, charges each
block against the configured `rag.file.max_size` limit (surfaced to the loader
config as `file_max_size`) and aborts the moment the running total passes it,
and creates the temp file inside the `try` so the existing `finally` removes it
when the download fails.

Discriminates: the narrow tests pass on v0.11.1 and fail on v0.11.0, where the
whole body is read up front, the limit does not exist, and the loader runs on
the oversized download instead of the fetch being refused.

Everything here drives a fake `requests.Response`; no network, and the fake
bodies are 16 MB at most so the buggy checkout cannot exhaust anything either.
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

MB = 1024 * 1024

# Small enough that the whole fixture body stays trivial for the buggy checkout
# (which really does buffer it), large enough to be several stream blocks.
MAX_SIZE_MB = 1


class FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content
        self.metadata: dict = {}


class FakeLoader:
    """Stand-in for the extraction pipeline, recording what it was asked to load."""

    def __init__(self, pages: list[str] | None = None) -> None:
        self.pages = pages if pages is not None else ['extracted text']
        self.calls: list[tuple[str, str, str]] = []

    def load(self, filename: str, content_type: str, path: str):
        self.calls.append((filename, content_type, path))
        return [FakeDocument(page) for page in self.pages]


class FakeResponse:
    """Fake `requests.Response` that can be streamed, or buffered via `.content`.

    Records whether `.content` was touched and how many blocks the caller
    actually pulled, which is what separates streaming from buffering.
    `fail_after` makes the body die partway, the way a dropped connection does.
    """

    def __init__(
        self,
        chunks: list[bytes],
        headers: dict | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._chunks = chunks
        self.headers = headers or {'Content-Type': 'application/pdf'}
        self.fail_after = fail_after
        self.content_accessed = False
        self.chunks_pulled = 0

    @property
    def content(self) -> bytes:
        self.content_accessed = True
        if self.fail_after is not None:
            self.chunks_pulled = self.fail_after
            raise OSError('connection reset while downloading')
        self.chunks_pulled = len(self._chunks)
        return b''.join(self._chunks)

    def iter_content(self, chunk_size: int = 1):
        for index, chunk in enumerate(self._chunks):
            if self.fail_after is not None and index == self.fail_after:
                raise OSError('connection reset while downloading')
            self.chunks_pulled = index + 1
            yield chunk


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Point `tempfile` at an empty directory so leftovers are visible."""
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    return tmp_path


@pytest.fixture
def loader(retrieval_utils_module, monkeypatch):
    """Replace the extraction pipeline; these tests are about the download only."""
    fake = FakeLoader()
    monkeypatch.setattr(
        retrieval_utils_module, 'build_loader_from_config', lambda request, config: fake
    )
    return fake


def loader_config(max_size_mb=MAX_SIZE_MB) -> dict:
    return {'file_max_size': max_size_mb, 'CONTENT_EXTRACTION_ENGINE': ''}


def extract(retrieval_utils_module, response, url='https://example.com/report.pdf', config=None):
    return retrieval_utils_module._extract_text_from_binary_response(
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
        response,
        url,
        loader_config() if config is None else config,
    )


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


def test_oversized_download_is_cut_off_mid_stream(retrieval_utils_module, temp_dir, loader):
    """The limit trips while the body is still arriving, not after it all landed."""
    # 16 MB offered against a 1 MB limit; a streaming reader stops after ~1 MB.
    response = FakeResponse([b'x' * (MB // 4)] * 64)

    with pytest.raises(ValueError) as excinfo:
        extract(retrieval_utils_module, response, config=loader_config())

    assert 'too large' in str(excinfo.value).lower()
    assert f'{MAX_SIZE_MB} MB' in str(excinfo.value)
    # 1 MB limit / 256 KB blocks: refused on the 5th, nowhere near all 64.
    assert response.chunks_pulled <= 8, (
        f'read {response.chunks_pulled} of 64 blocks; the limit was applied '
        'after the fact, not mid-stream'
    )


def test_oversized_download_never_reaches_the_loader(retrieval_utils_module, temp_dir, loader):
    """A refused download must not be handed to the extraction pipeline."""
    response = FakeResponse([b'x' * (MB // 4)] * 64)

    with pytest.raises(ValueError):
        extract(retrieval_utils_module, response, config=loader_config())

    assert loader.calls == [], 'the oversized download was extracted anyway'


def test_body_is_streamed_instead_of_buffered(retrieval_utils_module, temp_dir, loader):
    """`.content` reads the whole body into memory, so it must stay untouched."""
    response = FakeResponse([b'a' * 1024] * 4)

    content, docs = extract(retrieval_utils_module, response)

    assert content == 'extracted text'
    assert not response.content_accessed, (
        'the response body was buffered whole via .content instead of streamed'
    )


def test_failed_download_leaves_no_partial_file(retrieval_utils_module, temp_dir, loader):
    """A download that dies partway must clean up the temp file it started."""
    response = FakeResponse([b'a' * 1024] * 8, fail_after=3)

    with pytest.raises(OSError):
        extract(retrieval_utils_module, response)

    leftovers = list(temp_dir.iterdir())
    assert leftovers == [], f'partial download left behind: {leftovers}'


def test_download_at_the_limit_is_accepted(retrieval_utils_module, temp_dir, loader):
    """Exactly the configured size is allowed; the check is on exceeding it."""
    response = FakeResponse([b'x' * (MB // 4)] * 4)

    content, docs = extract(retrieval_utils_module, response, config=loader_config())

    assert content == 'extracted text'
    assert len(loader.calls) == 1
    assert not response.content_accessed, (
        'the body was buffered whole via .content instead of streamed'
    )


def test_unset_limit_downloads_without_a_cap(retrieval_utils_module, temp_dir, loader):
    """No configured limit means no cap, matching the sibling upload path."""
    response = FakeResponse([b'x' * (MB // 4)] * 12)

    content, docs = extract(
        retrieval_utils_module,
        response,
        config={'file_max_size': None, 'CONTENT_EXTRACTION_ENGINE': ''},
    )

    assert content == 'extracted text'
    assert response.chunks_pulled == 12
    assert not response.content_accessed, (
        'the body was buffered whole via .content instead of streamed'
    )


# ---------------------------------------------------------------------------
# broad: behaviour of the binary branch that must survive the rewrite
# ---------------------------------------------------------------------------


def test_extracted_docs_carry_the_source_url(retrieval_utils_module, temp_dir, loader):
    loader.pages = ['first page', 'second page']
    url = 'https://example.com/docs/report.pdf'

    content, docs = extract(retrieval_utils_module, FakeResponse([b'a' * 16]), url=url)

    assert content == 'first page second page'
    assert [doc.metadata['source'] for doc in docs] == [url, url]


def test_filename_and_suffix_come_from_the_url_path(retrieval_utils_module, temp_dir, loader):
    extract(
        retrieval_utils_module,
        FakeResponse([b'a' * 16]),
        url='https://example.com/files/quarterly.PDF',
    )

    filename, content_type, path = loader.calls[0]
    assert filename == 'quarterly.PDF'
    assert content_type == 'application/pdf'
    assert path.endswith('.pdf'), f'temp file kept no usable suffix: {path}'


def test_filename_falls_back_to_content_disposition(retrieval_utils_module, temp_dir, loader):
    response = FakeResponse(
        [b'a' * 16],
        headers={
            'Content-Type': 'application/pdf',
            'Content-Disposition': 'attachment; filename="statement.pdf"',
        },
    )

    extract(retrieval_utils_module, response, url='https://example.com/download')

    assert loader.calls[0][0] == 'statement.pdf'


def test_temp_file_is_removed_after_a_successful_extraction(
    retrieval_utils_module, temp_dir, loader
):
    extract(retrieval_utils_module, FakeResponse([b'a' * 16]))

    assert list(temp_dir.iterdir()) == []


def test_downloaded_bytes_are_what_the_loader_reads(retrieval_utils_module, temp_dir, monkeypatch):
    """Whatever path the body takes, the file on disk is the full body."""
    body = b''.join([bytes([index % 256]) * 4096 for index in range(32)])
    seen: dict[str, bytes] = {}

    class RecordingLoader(FakeLoader):
        def load(self, filename, content_type, path):
            with open(path, 'rb') as handle:
                seen['bytes'] = handle.read()
            return super().load(filename, content_type, path)

    monkeypatch.setattr(
        retrieval_utils_module,
        'build_loader_from_config',
        lambda request, config: RecordingLoader(),
    )
    chunks = [body[i : i + 4096] for i in range(0, len(body), 4096)]
    extract(retrieval_utils_module, FakeResponse(chunks))

    assert seen['bytes'] == body


def test_loader_failure_still_removes_the_temp_file(retrieval_utils_module, temp_dir, monkeypatch):
    class ExplodingLoader:
        def load(self, filename, content_type, path):
            raise RuntimeError('extraction engine unavailable')

    monkeypatch.setattr(
        retrieval_utils_module,
        'build_loader_from_config',
        lambda request, config: ExplodingLoader(),
    )

    with pytest.raises(RuntimeError):
        extract(retrieval_utils_module, FakeResponse([b'a' * 16]))

    assert list(temp_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# nearby: the routing that decides a fetch is a binary download at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'content_type',
    ['text/html', 'text/plain; charset=utf-8', 'application/json', 'application/xml', ''],
)
def test_text_content_types_stay_on_the_web_loader(retrieval_utils_module, content_type):
    assert retrieval_utils_module._is_text_content_type(content_type) is True


@pytest.mark.parametrize(
    'content_type',
    ['application/pdf', 'application/zip', 'image/png', 'application/octet-stream'],
)
def test_binary_content_types_take_the_download_branch(retrieval_utils_module, content_type):
    assert retrieval_utils_module._is_text_content_type(content_type) is False
