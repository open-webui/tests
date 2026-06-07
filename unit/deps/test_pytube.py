"""Dependency contract: pytube (import name ``pytube``).

pytube is a YouTube client (metadata, streams, captions). Open WebUI pins it
in ``backend/requirements.txt`` (``pytube==15.0.0``) but does NOT import it
directly in ``open_webui/*``: the YouTube transcript ingestion in
``retrieval/loaders/youtube.py`` is built on ``youtube_transcript_api`` and a
custom ``_parse_video_id``. pytube is present as part of the YouTube/RAG
loader ecosystem (and is a transitive option for resolving video URLs /
streams); the backend's relevant config is ``YOUTUBE_LOADER_PROXY_URL``, which
maps onto pytube's ``proxies=`` constructor kwarg.

Because nothing in the backend names pytube directly, this module pins its
*core public surface* so a bump that broke it surfaces here rather than as an
opaque failure in a YouTube-loader code path. We pin: the top-level
``YouTube`` / ``Playlist`` / ``Search`` classes, the ``YouTube`` constructor
signature (notably ``proxies=`` for the proxy config), the ``extract`` URL
helpers, and the ``exceptions`` hierarchy. The behavioural contracts run fully
OFFLINE — they exercise only pure URL parsing (``extract.video_id``, which
never touches the network) and lazy ``YouTube(url)`` construction (which does
NOT fetch until a remote property like ``.streams`` is accessed, which we never
do). No video is ever downloaded and no request is made.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pytube"
DIST_NAME = "pytube"

# Top-level classes/submodules the YouTube ecosystem resolves on pytube.
TOP_LEVEL_SYMBOLS = [
    "YouTube",  # the single-video client
    "Playlist",  # playlist client
    "Search",  # search client
    "Stream",  # a downloadable stream
    "extract",  # URL parsing helpers
    "exceptions",  # error hierarchy
]

# Remote-data properties on YouTube (existence only — we never access them, as
# touching them triggers a network fetch).
YOUTUBE_PROPERTIES = ["streams", "title", "length", "captions", "description"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`pytube` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pytube"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `pytube.*` symbol the ecosystem resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_youtube_is_callable(depcheck):
    """YouTube(url) is the entry point; it must be a callable class."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "YouTube")


def test_youtube_init_accepts_url_and_proxies(depcheck):
    """YouTube(url, proxies=...) — the `url` positional and the `proxies` kwarg
    (which the YOUTUBE_LOADER_PROXY_URL config maps onto) must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.YouTube.__init__, ["url", "proxies"])


def test_youtube_properties_exist(depcheck):
    """The metadata/stream properties the loader ecosystem reads must exist on
    the class. We only check existence (accessing them would fetch over the
    network)."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.YouTube))
    missing = [p for p in YOUTUBE_PROPERTIES if p not in names]
    assert not missing, f"pytube.YouTube missing property/method(s): {missing}"


def test_extract_helpers_exist(depcheck):
    """pytube.extract holds the URL parsers; video_id is the one a URL->id
    resolution path relies on."""
    depcheck.load(IMPORT_NAME)
    extract = depcheck.load("pytube.extract")
    assert hasattr(extract, "video_id"), "pytube.extract.video_id is gone"
    assert callable(extract.video_id)


def test_exceptions_hierarchy(depcheck):
    """pytube.exceptions has a common PytubeError base; RegexMatchError (bad URL)
    and VideoUnavailable (gone/private video) must subclass it so callers can
    catch the family."""
    depcheck.load(IMPORT_NAME)
    ex = depcheck.load("pytube.exceptions")
    assert hasattr(ex, "PytubeError"), "pytube.exceptions.PytubeError is gone"
    for name in ("RegexMatchError", "VideoUnavailable"):
        assert hasattr(ex, name), f"pytube.exceptions.{name} is gone"
        assert issubclass(getattr(ex, name), ex.PytubeError), (
            f"{name} no longer subclasses PytubeError"
        )


# ---------------------------------------------------------------------------
# Behavioural: pure URL parsing (offline) + lazy construction (no fetch)
# ---------------------------------------------------------------------------


def test_behaviour_video_id_from_watch_url(depcheck):
    """extract.video_id parses the canonical watch URL form. Pure string
    parsing — no network. Pin the 11-char id is extracted."""
    depcheck.load(IMPORT_NAME)
    extract = depcheck.load("pytube.extract")
    vid = extract.video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert vid == "dQw4w9WgXcQ"


def test_behaviour_video_id_from_short_url(depcheck):
    """extract.video_id parses the youtu.be short form too (the share link
    format). Offline string parsing."""
    depcheck.load(IMPORT_NAME)
    extract = depcheck.load("pytube.extract")
    vid = extract.video_id("https://youtu.be/dQw4w9WgXcQ")
    assert vid == "dQw4w9WgXcQ"


def test_behaviour_video_id_bad_url_raises_regexmatcherror(depcheck):
    """A string with no parseable 11-char video-id segment must raise
    RegexMatchError, not return a bogus id — the signal a resolver uses to reject
    a non-video input. Pin the exact exception type (offline).

    NOTE: pytube's video_id regex is deliberately lenient (it will accept any
    trailing 11-char [A-Za-z0-9_-] run), so the negative case must be a string
    that contains NO such segment at all."""
    depcheck.load(IMPORT_NAME)
    extract = depcheck.load("pytube.extract")
    ex = depcheck.load("pytube.exceptions")
    with pytest.raises(ex.RegexMatchError):
        extract.video_id("not-a-url")


def test_behaviour_youtube_construct_is_lazy_offline(depcheck):
    """YouTube(url, proxies=...) must construct without making any request (pytube
    defers all network I/O until a remote property is accessed). Building it with
    a proxy dict and reading the locally-derived watch_url / video_id must work
    offline and never fetch."""
    mod = depcheck.load(IMPORT_NAME)
    yt = mod.YouTube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        proxies={"https": "http://127.0.0.1:1"},
    )
    # These are derived locally from the URL, no fetch:
    assert "dQw4w9WgXcQ" in yt.watch_url
    assert yt.video_id == "dQw4w9WgXcQ"
