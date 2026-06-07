"""Dependency contract: youtube-transcript-api (import ``youtube_transcript_api``).

Open WebUI's ``retrieval/loaders/youtube.py`` (``YoutubeLoader``) uses this
library to pull video captions for RAG. It is on the **v1** API where the
client is *instantiated* (older versions used classmethods). The exact
surface the loader touches:

    from youtube_transcript_api import (
        NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi,
    )
    from youtube_transcript_api.proxies import GenericProxyConfig

    api = YouTubeTranscriptApi(proxy_config=youtube_proxies)   # or None
    transcript_list = api.list(self.video_id)
    transcript = transcript_list.find_transcript([lang])
    if transcript.is_generated:
        transcript = transcript_list.find_manually_created_transcript([lang])
    pieces = transcript.fetch()                                # -> snippets
    text = ' '.join(p.text.strip(' ') for p in pieces if hasattr(p, 'text'))

It also raises ``NoTranscriptFound(video_id, languages, list(...))`` itself
when every language fails, and catches ``NoTranscriptFound`` /
``TranscriptsDisabled`` (via ``ParseError`` from the stdlib too).

This module pins that surface and — crucially — reproduces the loader's
transcript-selection logic OFFLINE by *constructing* ``Transcript`` /
``TranscriptList`` / ``FetchedTranscript`` objects directly (their public
constructors accept a ``requests.Session`` and plain data), so no request
ever reaches YouTube. ``api.list()`` / ``transcript.fetch()`` are never
called against the network; instead the same objects those would return are
built by hand and their behaviour asserted.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "youtube_transcript_api"
DIST_NAME = "youtube-transcript-api"

USED_SYMBOLS = [
    "YouTubeTranscriptApi",
    "NoTranscriptFound",
    "TranscriptsDisabled",
    "Transcript",
    "TranscriptList",
    "FetchedTranscript",
    "FetchedTranscriptSnippet",
    "CouldNotRetrieveTranscript",
    "YouTubeTranscriptApiException",
    "proxies",
    "proxies.GenericProxyConfig",
]


def _session(depcheck):
    requests = depcheck.try_load("requests")
    if requests is None:
        pytest.skip("requests not installed; needed to construct Transcript objects")
    return requests.Session()


def _make_transcript(mod, session, *, language_code="en", is_generated=False):
    """Build a Transcript object offline (no network), as api.list() would
    return inside a TranscriptList."""
    return mod.Transcript(
        http_client=session,
        video_id="vid12345678",
        url="http://example.invalid/transcript",
        language="English",
        language_code=language_code,
        is_generated=is_generated,
        translation_languages=[],
    )


def _make_transcript_list(mod, *, manual=None, generated=None):
    return mod.TranscriptList(
        video_id="vid12345678",
        manually_created_transcripts=manual or {},
        generated_transcripts=generated or {},
        translation_languages=[],
    )


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "youtube_transcript_api"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_proxies_generic_proxy_config_importable(depcheck):
    """The loader does `from youtube_transcript_api.proxies import
    GenericProxyConfig`. The submodule + class must resolve."""
    mod = depcheck.load(IMPORT_NAME)
    proxies = depcheck.resolve(mod, "proxies")
    assert hasattr(proxies, "GenericProxyConfig")


# ---------------------------------------------------------------------------
# Exception hierarchy — the loader catches NoTranscriptFound /
# TranscriptsDisabled and raises NoTranscriptFound itself.
# ---------------------------------------------------------------------------


def test_exception_hierarchy(depcheck):
    """NoTranscriptFound and TranscriptsDisabled must both subclass
    CouldNotRetrieveTranscript -> YouTubeTranscriptApiException -> Exception,
    so the loader's targeted except-clauses keep catching them."""
    mod = depcheck.load(IMPORT_NAME)
    base = mod.CouldNotRetrieveTranscript
    for name in ("NoTranscriptFound", "TranscriptsDisabled"):
        exc = getattr(mod, name)
        assert issubclass(exc, base), f"{name} no longer subclasses CouldNotRetrieveTranscript"
        assert issubclass(exc, mod.YouTubeTranscriptApiException)
        assert issubclass(exc, Exception)


# ---------------------------------------------------------------------------
# YouTubeTranscriptApi — v1 instantiation contract (offline; never .list()).
# ---------------------------------------------------------------------------


def test_api_is_instantiable_class(depcheck):
    """The loader does `api = YouTubeTranscriptApi(proxy_config=...)`. It must
    be a class (v1 API), not a namespace of classmethods (the old v0 API)."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.YouTubeTranscriptApi, type)


def test_api_constructor_accepts_proxy_config(depcheck):
    """The constructor must accept proxy_config (the loader passes it, or
    None). Construction must not perform any network I/O."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.YouTubeTranscriptApi.__init__)
    assert "proxy_config" in sig.parameters
    api = mod.YouTubeTranscriptApi(proxy_config=None)
    assert api is not None


def test_api_has_list_and_fetch_methods(depcheck):
    """The loader calls api.list(video_id); v1 also exposes api.fetch. Both
    must be callable methods on the class (we do NOT invoke them — that hits
    YouTube)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("list", "fetch"):
        assert callable(getattr(mod.YouTubeTranscriptApi, name, None)), (
            f"YouTubeTranscriptApi.{name} missing/not callable"
        )


def test_api_list_signature(depcheck):
    """api.list(video_id) — the video id must remain its argument."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.YouTubeTranscriptApi.list, ["video_id"])


# ---------------------------------------------------------------------------
# GenericProxyConfig — offline construction with http_url/https_url.
# ---------------------------------------------------------------------------


def test_generic_proxy_config_constructs(depcheck):
    """The loader builds GenericProxyConfig(http_url=, https_url=) when a proxy
    is configured. Both kwargs must be accepted and construction offline."""
    mod = depcheck.load(IMPORT_NAME)
    GenericProxyConfig = depcheck.resolve(mod, "proxies.GenericProxyConfig")
    sig = inspect.signature(GenericProxyConfig.__init__)
    assert "http_url" in sig.parameters and "https_url" in sig.parameters
    cfg = GenericProxyConfig(
        http_url="http://proxy.invalid:8080", https_url="http://proxy.invalid:8080"
    )
    assert cfg is not None


def test_api_accepts_proxy_config_instance(depcheck):
    """A GenericProxyConfig instance must be accepted as proxy_config (the
    full call shape the loader uses when self.proxy_url is set)."""
    mod = depcheck.load(IMPORT_NAME)
    GenericProxyConfig = depcheck.resolve(mod, "proxies.GenericProxyConfig")
    cfg = GenericProxyConfig(http_url="http://p.invalid:1", https_url="http://p.invalid:1")
    api = mod.YouTubeTranscriptApi(proxy_config=cfg)
    assert api is not None


# ---------------------------------------------------------------------------
# Transcript / TranscriptList — the selection logic, reproduced offline.
# ---------------------------------------------------------------------------


def test_transcript_exposes_is_generated_and_fetch(depcheck):
    """The loader reads transcript.is_generated and calls transcript.fetch().
    Build a Transcript offline and pin both members."""
    mod = depcheck.load(IMPORT_NAME)
    t = _make_transcript(mod, _session(depcheck), is_generated=True)
    assert t.is_generated is True
    assert t.language_code == "en"
    assert callable(t.fetch)


def test_transcript_fetch_signature(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Transcript.fetch, ["preserve_formatting"])


def test_transcript_list_find_transcript(depcheck):
    """find_transcript([lang]) returns the matching transcript. Build a list
    offline with an 'en' manual transcript and assert the lookup."""
    mod = depcheck.load(IMPORT_NAME)
    sess = _session(depcheck)
    en = _make_transcript(mod, sess, language_code="en", is_generated=False)
    tl = _make_transcript_list(mod, manual={"en": en})
    found = tl.find_transcript(["en"])
    assert found.language_code == "en"


def test_transcript_list_find_missing_raises(depcheck):
    """find_transcript for an unavailable language must raise NoTranscriptFound
    — the loader catches this per-language and moves on."""
    mod = depcheck.load(IMPORT_NAME)
    sess = _session(depcheck)
    tl = _make_transcript_list(mod, manual={"en": _make_transcript(mod, sess)})
    with pytest.raises(mod.NoTranscriptFound):
        tl.find_transcript(["zz"])


def test_transcript_list_prefers_manual_via_loader_logic(depcheck):
    """Reproduce the loader's manual-over-generated preference: it first does
    find_transcript([lang]); if that is .is_generated it then calls
    find_manually_created_transcript([lang]). With both a generated and a
    manual 'en' transcript present, that second call must yield the manual
    one."""
    mod = depcheck.load(IMPORT_NAME)
    sess = _session(depcheck)
    gen = _make_transcript(mod, sess, language_code="en", is_generated=True)
    man = _make_transcript(mod, sess, language_code="en", is_generated=False)
    tl = _make_transcript_list(mod, manual={"en": man}, generated={"en": gen})

    transcript = tl.find_transcript(["en"])
    if transcript.is_generated:
        transcript = tl.find_manually_created_transcript(["en"])
    assert transcript.is_generated is False  # ended on the manual transcript


def test_find_manually_created_missing_raises(depcheck):
    """When only a generated transcript exists, find_manually_created_transcript
    must raise NoTranscriptFound (the loader catches it and keeps the generated
    one)."""
    mod = depcheck.load(IMPORT_NAME)
    sess = _session(depcheck)
    gen = _make_transcript(mod, sess, language_code="en", is_generated=True)
    tl = _make_transcript_list(mod, generated={"en": gen})
    with pytest.raises(mod.NoTranscriptFound):
        tl.find_manually_created_transcript(["en"])


# ---------------------------------------------------------------------------
# FetchedTranscript / snippet — the .text extraction the loader joins.
# ---------------------------------------------------------------------------


def test_fetched_transcript_is_iterable_of_text_snippets(depcheck):
    """transcript.fetch() returns a FetchedTranscript the loader iterates,
    reading p.text on each snippet. Build one offline and reproduce the join
    `' '.join(p.text.strip(' ') for p in pieces)`."""
    mod = depcheck.load(IMPORT_NAME)
    snippets = [
        mod.FetchedTranscriptSnippet(text=" hello ", start=0.0, duration=1.0),
        mod.FetchedTranscriptSnippet(text="world ", start=1.0, duration=1.0),
    ]
    fetched = mod.FetchedTranscript(
        snippets=snippets,
        video_id="vid12345678",
        language="English",
        language_code="en",
        is_generated=False,
    )
    text = " ".join(p.text.strip(" ") for p in fetched if hasattr(p, "text"))
    assert text == "hello world"


def test_fetched_transcript_snippet_fields(depcheck):
    """A snippet must carry text/start/duration (the loader reads .text; other
    consumers read timing)."""
    mod = depcheck.load(IMPORT_NAME)
    snip = mod.FetchedTranscriptSnippet(text="hi", start=2.5, duration=0.5)
    assert snip.text == "hi"
    assert snip.start == 2.5
    assert snip.duration == 0.5


def test_no_transcript_found_constructible_by_loader(depcheck):
    """When all languages fail the loader raises
    NoTranscriptFound(video_id, languages, list(transcript_list)). The
    exception must be constructible with that 3-arg shape."""
    mod = depcheck.load(IMPORT_NAME)
    sess = _session(depcheck)
    tl = _make_transcript_list(mod, manual={"en": _make_transcript(mod, sess)})
    exc = mod.NoTranscriptFound("vid12345678", ["de", "fr"], list(tl))
    assert isinstance(exc, mod.NoTranscriptFound)
    assert isinstance(exc, Exception)
