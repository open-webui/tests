"""Regressions in building web loaders and reading YouTube transcripts, fixed in v0.11.1.

- Tavily web loader (PR #27636, commit 104a0f2f1): `SafeTavilyLoader.__init__` kept a required
  `api_base_url` that `get_web_loader` never passed, so choosing Tavily raised before any
  request. Every engine `get_web_loader` offers must accept what it passes (the Web IQ loader had
  the same defect the other way round, issue #28688).
- YouTube transcript errors (issue #28361, PR #28362, commit 121f2404e): the loader caught the
  transcript API's exception and returned `[]`, so the reason, and the proxy hint that fixes a
  blocked server, never reached the user. `YoutubeTranscriptError` now carries a reason per cause.

Tavily's API base URL is environment-only and the transcript API talks to YouTube, so these stay
here; the transcript API client is the stand-in boundary, specced from the real classes. The
attached-page, Web IQ, content sniffing and web search error cases moved to
integration/retrieval/test_web_loaders.py.

Discriminates: passes on dev bbfa876af; a required `api_base_url` on the Tavily loader, a Web IQ
loader that takes none and a refused transcript returning `[]` each fail their tests.
"""

from __future__ import annotations

from unittest.mock import create_autospec, patch

import pytest
import youtube_transcript_api
from youtube_transcript_api import (
    AgeRestricted,
    FetchedTranscript,
    FetchedTranscriptSnippet,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    Transcript,
    TranscriptList,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApi,
)

pytestmark = pytest.mark.regression

# a public address literal: validated without a DNS lookup and never fetched
PAGE = "https://93.184.216.34/page"
VIDEO_ID = "dQw4w9WgXcQ"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.fixture(scope="session")
def youtube_loader_module(owui_module):
    return owui_module("open_webui.retrieval.loaders.youtube")


# --- building web loaders ----------------------------------------------------------------------


def build(module, engine: str, **settings):
    return module.get_web_loader(
        urls=[PAGE], loader_config={"web_loader_engine": engine, **settings}
    )


def test_the_tavily_loader_is_built_from_the_saved_settings(retrieval_web_utils_module):
    loader = build(
        retrieval_web_utils_module,
        "tavily",
        tavily_api_key="tvly-key",
        tavily_extract_depth="advanced",
    )

    assert isinstance(loader, retrieval_web_utils_module.SafeTavilyLoader)
    assert (loader.api_key, loader.extract_depth, loader.web_paths) == (
        "tvly-key",
        "advanced",
        [PAGE],
    )


@pytest.mark.parametrize(
    ("engine", "settings"),
    [
        ("safe_web", {}),
        ("firecrawl", {"firecrawl_api_key": "fc-key", "firecrawl_api_url": "https://fc.example"}),
        ("tavily", {"tavily_api_key": "tvly-key"}),
        (
            "microsoft_web_iq",
            {
                "microsoft_web_iq_api_base_url": "https://iq.example/v3",
                "microsoft_web_iq_api_key": "k",
            },
        ),
        ("external", {"external_web_loader_url": "https://loader.example"}),
    ],
)
def test_every_selectable_engine_is_built(retrieval_web_utils_module, engine, settings):
    """Both loader bugs were a constructor disagreeing with its only caller."""
    loader = build(retrieval_web_utils_module, engine, **settings)

    assert (getattr(loader, "web_paths", None) or loader.urls) == [PAGE]


def test_an_unknown_engine_is_refused(retrieval_web_utils_module):
    with pytest.raises(ValueError, match="Invalid WEB_LOADER_ENGINE"):
        build(retrieval_web_utils_module, "not-an-engine")


def test_no_loader_is_built_when_every_url_is_blocked(retrieval_web_utils_module):
    with pytest.raises(ValueError):
        retrieval_web_utils_module.get_web_loader(
            urls=["http://169.254.169.254/latest"], loader_config={"web_loader_engine": "safe_web"}
        )


# --- YouTube transcripts -----------------------------------------------------------------------


def transcript_api(listed: Exception | TranscriptList):
    """The transcript client, specced from the real class: `list` raises or returns `listed`."""
    api_class = create_autospec(YouTubeTranscriptApi)
    if isinstance(listed, Exception):
        api_class.return_value.list.side_effect = listed
    else:
        api_class.return_value.list.return_value = listed
    return patch.object(youtube_transcript_api, "YouTubeTranscriptApi", api_class)


def load_transcript(module, listed, **options):
    with transcript_api(listed):
        return module.YoutubeLoader(VIDEO_URL, **options).load()


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (RequestBlocked(VIDEO_ID), "Youtube Proxy URL"),
        (IpBlocked(VIDEO_ID), "Youtube Proxy URL"),
        (TranscriptsDisabled(VIDEO_ID), "Transcripts are disabled"),
        (AgeRestricted(VIDEO_ID), "age restricted"),
        (VideoUnavailable(VIDEO_ID), "is unavailable"),
        (VideoUnplayable(VIDEO_ID, "private", []), "is unavailable"),
        (InvalidVideoId(VIDEO_ID), "is unavailable"),
        (PoTokenRequired(VIDEO_ID), "additional verification"),
        (RuntimeError("connection reset"), "Could not retrieve a transcript"),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else None,
)
def test_a_refused_transcript_raises_with_its_reason(youtube_loader_module, error, fragment):
    with pytest.raises(youtube_loader_module.YoutubeTranscriptError) as raised:
        load_transcript(youtube_loader_module, error)

    assert fragment in str(raised.value)
    assert VIDEO_ID in str(raised.value)


def test_a_missing_language_names_the_languages_tried(youtube_loader_module):
    transcripts = create_autospec(TranscriptList, instance=True)
    transcripts.find_transcript.side_effect = NoTranscriptFound(VIDEO_ID, ["de"], transcripts)

    with pytest.raises(youtube_loader_module.YoutubeTranscriptError, match="de, en"):
        load_transcript(youtube_loader_module, transcripts, language=["de"])


def test_a_transcript_still_loads(youtube_loader_module):
    snippets = [
        FetchedTranscriptSnippet(text="never gonna", start=0.0, duration=1.0),
        FetchedTranscriptSnippet(text="give you up", start=1.0, duration=1.0),
    ]
    transcript = create_autospec(Transcript, instance=True)
    transcript.is_generated = False
    transcript.fetch.return_value = FetchedTranscript(
        snippets=snippets,
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=False,
    )
    transcripts = create_autospec(TranscriptList, instance=True)
    transcripts.find_transcript.return_value = transcript

    documents = load_transcript(youtube_loader_module, transcripts)

    assert [document.page_content for document in documents] == ["never gonna give you up"]
