"""Regressions in the web-loader and URL-attachment paths, fixed in v0.11.1.

Six independent defects, all in the "attach a link / search the web" flow:

- Tavily web loader (PR #27636, commit 104a0f2f1): `SafeTavilyLoader.__init__`
  kept a required `api_base_url` parameter that `get_web_loader` never passed,
  so selecting Tavily as the web loader raised before any request was made.
- Microsoft Web IQ loader (issue #28688, commits 6dcc2d5269 + 140d2cf4b5):
  `SafeMicrosoftWebIQLoader.__init__` did not accept `api_base_url` at all,
  while `get_web_loader` always passes it. Broken since the loader shipped.
- Content-type sniffing (commit 886248de36): `_is_text_content_type` matched
  any type merely *containing* 'xml'/'json'/'javascript', so Office archives
  (`...openxmlformats...`) were pushed through the text web loader.
- Attached page content (issue #28378, commit 9c21d4ed3b): `process_web`
  returned the extracted text only under `file.data.content`, but the caller
  reads the top-level `content`, so the model received an empty attachment.
- YouTube transcript errors (issue #28361, PR #28362, commit 121f2404e): the
  loader swallowed the transcript-API exception and returned `[]`, and
  `process_web` blamed the knowledge base for anything it could not fetch.
  Now `YoutubeTranscriptError` carries a reason and the URL is named.
- Web search errors (PR #28942/#28948, commit 945c521ed): `WEB_SEARCH_ERROR`
  was a passthrough lambda, so `WEB_SEARCH_ERROR(e)` put the Exception object
  itself into `HTTPException.detail`, which does not serialise and the user
  saw a blank error.

Discriminates: passes on v0.11.1, fails on v0.11.0 (pre-fix the Tavily and
Web IQ loaders cannot be constructed, Office archives sniff as text, the
`process_web` response has no top-level `content`, YouTube failures return an
empty list and are reported as knowledge-base errors, and the web-search
error detail is an Exception object rather than a string).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.regression


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def retrieval_router_module(owui_module):
    """`open_webui.routers.retrieval` (process_web, process_web_search)."""
    return owui_module("open_webui.routers.retrieval")


@pytest.fixture(scope="session")
def youtube_loader_module(owui_module):
    """`open_webui.retrieval.loaders.youtube`."""
    return owui_module("open_webui.retrieval.loaders.youtube")


@pytest.fixture(scope="session")
def constants_module(owui_module):
    """`open_webui.constants` (ERROR_MESSAGES)."""
    return owui_module("open_webui.constants")


class _AsyncReturn:
    """Awaitable stand-in for an async I/O boundary."""

    def __init__(self, value):
        self._value = value

    async def __call__(self, *_args, **_kwargs):
        return self._value


class _AsyncRaise:
    def __init__(self, error: Exception):
        self._error = error

    async def __call__(self, *_args, **_kwargs):
        raise self._error


def _user(role: str = "admin"):
    return SimpleNamespace(id="user-1", role=role, name="tester", email="t@example.com")


# =============================================================================
# get_web_loader: Tavily (#27636) and Microsoft Web IQ (#28688)
# =============================================================================


def _build_loader(module, engine: str, extra: dict | None = None):
    """Run the real `get_web_loader`, skipping only the DNS/SSRF check."""
    loader_config = {"web_loader_engine": engine}
    loader_config.update(extra or {})
    with patch.object(module, "safe_validate_urls", lambda urls: list(urls)):
        return module.get_web_loader(["https://example.com/page"], loader_config=loader_config)


def test_tavily_web_loader_can_be_constructed(retrieval_web_utils_module) -> None:
    """Regression for PR #27636: Tavily page reading failed on every attempt.

    `get_web_loader` never passes `api_base_url`, so the vestigial required
    parameter made the constructor raise before any URL was fetched.
    """
    loader = _build_loader(
        retrieval_web_utils_module,
        "tavily",
        {"tavily_api_key": "tvly-key", "tavily_extract_depth": "advanced"},
    )

    assert isinstance(loader, retrieval_web_utils_module.SafeTavilyLoader)
    assert loader.api_key == "tvly-key"
    assert loader.extract_depth == "advanced"
    assert loader.web_paths == ["https://example.com/page"]


def test_microsoft_web_iq_loader_keeps_configured_api_base_url(retrieval_web_utils_module) -> None:
    """Regression for issue #28688: the Web IQ loader rejected `api_base_url`.

    `get_web_loader` always passes the admin-configured base URL, so the
    loader could never be built and Web IQ browsing was dead on arrival.
    """
    loader = _build_loader(
        retrieval_web_utils_module,
        "microsoft_web_iq",
        {
            "microsoft_web_iq_api_base_url": "https://web-iq.internal/v3",
            "microsoft_web_iq_api_key": "iq-key",
            "microsoft_web_iq_language": "de",
        },
    )

    assert isinstance(loader, retrieval_web_utils_module.SafeMicrosoftWebIQLoader)
    assert loader.api_base_url == "https://web-iq.internal/v3"
    assert loader.api_key == "iq-key"
    assert loader.language == "de"


@pytest.mark.parametrize(
    ("engine", "extra"),
    [
        ("safe_web", {}),
        ("firecrawl", {"firecrawl_api_key": "fc-key", "firecrawl_api_url": "https://fc.example"}),
        ("tavily", {"tavily_api_key": "tvly-key"}),
        (
            "microsoft_web_iq",
            {
                "microsoft_web_iq_api_base_url": "https://web-iq.internal/v3",
                "microsoft_web_iq_api_key": "iq-key",
            },
        ),
        (
            "external",
            {
                "external_web_loader_url": "https://loader.example",
                "external_web_loader_api_key": "x",
            },
        ),
    ],
)
def test_every_selectable_web_loader_engine_constructs(
    retrieval_web_utils_module, engine: str, extra: dict
) -> None:
    """Broad: every engine `get_web_loader` offers must accept what it passes.

    Both loader bugs in this file are the same defect: the constructor and its
    only caller disagreed about the argument list.
    """
    loader = _build_loader(retrieval_web_utils_module, engine, extra)
    # ExternalWebLoader names the same list `urls`.
    stored_urls = getattr(loader, "web_paths", None) or loader.urls
    assert stored_urls == ["https://example.com/page"]


def test_unknown_web_loader_engine_still_rejected(retrieval_web_utils_module) -> None:
    """Nearby: an unrecognised engine name is an error, not a silent fallback."""
    with pytest.raises(ValueError, match="Invalid WEB_LOADER_ENGINE"):
        _build_loader(retrieval_web_utils_module, "not-an-engine")


def test_web_loader_rejects_urls_that_fail_validation(retrieval_web_utils_module) -> None:
    """Nearby: no loader is built when every URL is blocked."""
    with patch.object(retrieval_web_utils_module, "safe_validate_urls", lambda urls: []):
        with pytest.raises(ValueError):
            retrieval_web_utils_module.get_web_loader(
                ["https://example.com/page"], loader_config={"web_loader_engine": "safe_web"}
            )


# =============================================================================
# _is_text_content_type (commit 886248de36)
# =============================================================================


@pytest.mark.parametrize(
    "content_type",
    [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document; charset=binary",
    ],
)
def test_office_archives_are_not_treated_as_text(retrieval_utils_module, content_type: str) -> None:
    """Regression for commit 886248de36: substring matching caught archives.

    'application/vnd.openxmlformats-...' contains 'xml', so DOCX/XLSX/PPTX
    attachments were handed to the text web loader instead of the binary
    extractor and arrived as mojibake.
    """
    assert retrieval_utils_module._is_text_content_type(content_type) is False


@pytest.mark.parametrize(
    "content_type",
    [
        "text/html",
        "text/plain; charset=utf-8",
        "TEXT/HTML; charset=UTF-8",
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-javascript",
        "application/atom+xml",
        "application/ld+json",
        "",
        "   ",
    ],
)
def test_text_content_types_still_go_to_the_web_loader(
    retrieval_utils_module, content_type: str
) -> None:
    """Nearby: real text types, structured suffixes and the missing header."""
    assert retrieval_utils_module._is_text_content_type(content_type) is True


@pytest.mark.parametrize(
    "content_type",
    ["application/pdf", "application/zip", "image/png", "video/mp4", "application/octet-stream"],
)
def test_binary_content_types_still_bypass_the_web_loader(
    retrieval_utils_module, content_type: str
) -> None:
    """Nearby: types that were already handled correctly stay that way."""
    assert retrieval_utils_module._is_text_content_type(content_type) is False


# =============================================================================
# process_web (#28378, #28361/#28362)
# =============================================================================


def _process_web_config():
    return SimpleNamespace(BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True)


async def _run_process_web(module, *, content="extracted page text", error=None, process=True):
    from langchain_core.documents import Document

    if error is not None:
        boundary = _AsyncRaise(error)
    else:
        boundary = _AsyncReturn((content, [Document(page_content=content, metadata={})]))

    form_data = module.ProcessUrlForm(url="https://example.com/article", collection_name=None)
    with (
        patch.object(module, "get_retrieval_config", _AsyncReturn(_process_web_config())),
        patch.object(module, "get_content_from_url", boundary),
    ):
        return await module.process_web(
            request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            form_data=form_data,
            process=process,
            overwrite=True,
            user=_user(),
        )


@pytest.mark.asyncio
async def test_process_web_returns_content_at_top_level(retrieval_router_module) -> None:
    """Regression for issue #28378: attached pages reached the model empty.

    The caller reads the top-level `content`; pre-fix the text lived only
    under `file.data.content`.
    """
    result = await _run_process_web(retrieval_router_module)

    assert result["content"] == "extracted page text"


@pytest.mark.asyncio
async def test_process_web_still_nests_content_under_file_data(retrieval_router_module) -> None:
    """Nearby: the nested copy other consumers read is untouched."""
    result = await _run_process_web(retrieval_router_module)

    assert result["status"] is True
    assert result["filename"] == "https://example.com/article"
    assert result["file"]["data"]["content"] == "extracted page text"
    assert result["file"]["meta"]["source"] == "https://example.com/article"


@pytest.mark.asyncio
async def test_process_web_without_processing_returns_content(retrieval_router_module) -> None:
    """Nearby: the preview path already returned top-level content."""
    result = await _run_process_web(retrieval_router_module, process=False)

    assert result == {"status": True, "content": "extracted page text"}


@pytest.mark.asyncio
async def test_unreadable_link_error_names_the_link(retrieval_router_module) -> None:
    """Regression for PR #28362: fetch failures blamed the knowledge base.

    Everything from fetching the URL to writing the vector store sat in one
    try, so a page that could not be read produced 'Error querying knowledge
    base' without naming the URL.
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        await _run_process_web(retrieval_router_module, error=RuntimeError("connection reset"))

    assert excinfo.value.status_code == 400
    assert "https://example.com/article" in str(excinfo.value.detail)
    assert "knowledge base" not in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_youtube_transcript_error_reaches_the_user_verbatim(
    retrieval_router_module, youtube_loader_module
) -> None:
    """Regression for issue #28361: the transcript reason never reached the UI."""
    from fastapi import HTTPException

    reason = "Transcripts are disabled for the YouTube video dQw4w9WgXcQ."
    with pytest.raises(HTTPException) as excinfo:
        await _run_process_web(
            retrieval_router_module,
            error=youtube_loader_module.YoutubeTranscriptError(reason),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == reason


@pytest.mark.asyncio
async def test_process_web_passes_through_http_exceptions(retrieval_router_module) -> None:
    """Nearby: an HTTPException raised downstream keeps its own status."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        await _run_process_web(
            retrieval_router_module, error=HTTPException(status_code=403, detail="nope")
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "nope"


# =============================================================================
# YouTube transcript errors (PR #28362)
# =============================================================================


class _FakeTranscript:
    is_generated = False

    def __init__(self, pieces):
        self._pieces = pieces

    def fetch(self):
        return self._pieces


class _FakeTranscriptList:
    def __init__(self, transcript=None, find_error=None):
        self._transcript = transcript
        self._find_error = find_error

    def find_transcript(self, _languages):
        if self._find_error is not None:
            raise self._find_error
        return self._transcript

    def find_manually_created_transcript(self, _languages):
        return self._transcript


def _fake_transcript_api(*, list_error=None, transcript_list=None):
    """Replaces the youtube_transcript_api client, the only I/O boundary here."""

    class _Api:
        def __init__(self, *_args, **_kwargs):
            pass

        def list(self, _video_id):
            if list_error is not None:
                raise list_error
            return transcript_list

    return _Api


@pytest.mark.parametrize(
    ("error_name", "expected_fragment"),
    [
        ("RequestBlocked", "Youtube Proxy URL"),
        ("IpBlocked", "Youtube Proxy URL"),
        ("TranscriptsDisabled", "Transcripts are disabled"),
        ("AgeRestricted", "age restricted"),
        ("VideoUnavailable", "is unavailable"),
        ("VideoUnplayable", "is unavailable"),
        ("InvalidVideoId", "is unavailable"),
        ("PoTokenRequired", "additional verification"),
        ("SomethingElseEntirely", "Could not retrieve a transcript"),
    ],
)
def test_transcript_error_message_names_the_cause(
    youtube_loader_module, error_name: str, expected_fragment: str
) -> None:
    """Regression for PR #28362: every transcript failure gets its own reason.

    The mapping is by exception class *name*, so a locally defined class with
    the same name exercises the real branch.
    """
    error = type(error_name, (Exception,), {})()

    message = youtube_loader_module._transcript_error_message(error, "dQw4w9WgXcQ")

    assert expected_fragment in message
    assert "dQw4w9WgXcQ" in message


def test_blocked_transcript_raises_instead_of_returning_nothing(youtube_loader_module) -> None:
    """Regression for issue #28361: a blocked request returned an empty list.

    The empty list then failed downstream with a generic knowledge-base error,
    so the actual reason (and the proxy hint that fixes it) was gone.
    """
    import youtube_transcript_api
    from youtube_transcript_api import RequestBlocked

    loader = youtube_loader_module.YoutubeLoader("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with patch.object(
        youtube_transcript_api,
        "YouTubeTranscriptApi",
        _fake_transcript_api(list_error=RequestBlocked("dQw4w9WgXcQ")),
    ):
        with pytest.raises(youtube_loader_module.YoutubeTranscriptError) as excinfo:
            loader.load()

    assert "Youtube Proxy URL" in str(excinfo.value)
    assert "dQw4w9WgXcQ" in str(excinfo.value)


def test_missing_language_transcript_reports_the_languages_tried(youtube_loader_module) -> None:
    """Broad: exhausting every requested language is also a named failure."""
    import youtube_transcript_api
    from youtube_transcript_api import NoTranscriptFound

    loader = youtube_loader_module.YoutubeLoader(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", language=["de"]
    )
    transcript_list = _FakeTranscriptList(
        find_error=NoTranscriptFound("dQw4w9WgXcQ", ["de", "en"], [])
    )
    with patch.object(
        youtube_transcript_api,
        "YouTubeTranscriptApi",
        _fake_transcript_api(transcript_list=transcript_list),
    ):
        with pytest.raises(youtube_loader_module.YoutubeTranscriptError) as excinfo:
            loader.load()

    assert "de" in str(excinfo.value)


def test_successful_transcript_still_loads(youtube_loader_module) -> None:
    """Nearby: the happy path is unchanged by the new error handling."""
    import youtube_transcript_api

    transcript = _FakeTranscript(
        [SimpleNamespace(text="never gonna"), SimpleNamespace(text="give you up")]
    )
    with patch.object(
        youtube_transcript_api,
        "YouTubeTranscriptApi",
        _fake_transcript_api(transcript_list=_FakeTranscriptList(transcript=transcript)),
    ):
        docs = youtube_loader_module.YoutubeLoader(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ).load()

    assert len(docs) == 1
    assert docs[0].page_content == "never gonna give you up"


# =============================================================================
# Web search error detail (PR #28942 / #28948)
# =============================================================================


async def _run_failing_web_search(module):
    """Drive process_web_search with no engine configured, the reported case."""
    from fastapi import HTTPException

    config = SimpleNamespace(
        ENABLE_WEB_SEARCH=True,
        USER_PERMISSIONS={},
        WEB_SEARCH_ENGINE="",
        WEB_SEARCH_CONCURRENT_REQUESTS=0,
    )
    with patch.object(module, "get_retrieval_config", _AsyncReturn(config)):
        try:
            await module.process_web_search(
                request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
                form_data=module.SearchForm(queries=["open webui"]),
                user=_user(),
            )
        except HTTPException as e:
            return e
    raise AssertionError("process_web_search did not fail")


@pytest.mark.asyncio
async def test_failing_web_search_returns_a_readable_message(retrieval_router_module) -> None:
    """Regression for PR #28942: the user saw a blank web-search error.

    `WEB_SEARCH_ERROR` was a passthrough lambda, so `WEB_SEARCH_ERROR(e)`
    handed the Exception object itself to `HTTPException.detail`.
    """
    error = await _run_failing_web_search(retrieval_router_module)

    assert error.status_code == 400
    assert error.detail == "[ERROR: Something went wrong while searching the web.]"


@pytest.mark.asyncio
async def test_web_search_error_detail_is_json_serialisable(retrieval_router_module) -> None:
    """Broad: a 400 body must survive serialisation, or it becomes a 500."""
    error = await _run_failing_web_search(retrieval_router_module)

    assert isinstance(error.detail, str)
    assert json.loads(json.dumps({"detail": error.detail}))["detail"] == error.detail


def test_web_search_error_constant_is_a_plain_message(constants_module) -> None:
    """Regression for PR #28948: the constant is a string, not a lambda."""
    assert str(constants_module.ERROR_MESSAGES.WEB_SEARCH_ERROR) == (
        "Something went wrong while searching the web."
    )
    assert not callable(constants_module.ERROR_MESSAGES.WEB_SEARCH_ERROR)


@pytest.mark.asyncio
async def test_web_search_disabled_still_returns_403(retrieval_router_module) -> None:
    """Nearby: the permission gate ahead of the search is unchanged."""
    from fastapi import HTTPException

    config = SimpleNamespace(ENABLE_WEB_SEARCH=False, USER_PERMISSIONS={})
    with patch.object(retrieval_router_module, "get_retrieval_config", _AsyncReturn(config)):
        with pytest.raises(HTTPException) as excinfo:
            await retrieval_router_module.process_web_search(
                request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
                form_data=retrieval_router_module.SearchForm(queries=["open webui"]),
                user=_user(),
            )

    assert excinfo.value.status_code == 403
