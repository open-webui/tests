"""Dependency contract: starlette-compress (import name ``starlette_compress``).

Open WebUI installs response-compression middleware on the FastAPI app:

  - ``main.py``: ``from starlette_compress import CompressMiddleware`` and,
    gated on ``ENABLE_COMPRESSION_MIDDLEWARE`` (default True),
    ``app.add_middleware(CompressMiddleware)`` — no extra kwargs, so it
    relies entirely on the library's default behaviour (gzip/brotli/zstd
    negotiation, a ``minimum_size`` floor below which it leaves bodies
    untouched).

That is the whole integration: one imported symbol used as an ASGI
middleware class with default options. This module therefore pins:

  - the ``CompressMiddleware`` symbol exists and is an ASGI-middleware
    class (constructible with just ``app`` positionally, async ``__call__``);
  - the constructor keyword surface the library documents (``minimum_size``,
    the per-codec enable flags and quality levels) so a default change is
    noticed;
  - the real end-to-end behaviour, exercised OFFLINE against an in-process
    ASGI app with no server/network: a large body is gzip-compressed when
    the client sends ``Accept-Encoding: gzip``, the round-trip decompresses
    back to the original bytes, a body below ``minimum_size`` is passed
    through uncompressed, and a request with no ``Accept-Encoding`` is left
    untouched.

A starlette-compress bump that renamed the class, changed the constructor
contract, or broke negotiation would fail here instead of silently
shipping uncompressed (or corrupted) responses.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import gzip
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "starlette_compress"
DIST_NAME = "starlette-compress"

# The only symbol the backend imports.
USED_SYMBOLS = ["CompressMiddleware"]

# Public helpers the library documents (pinned as part of the surface even
# though main.py doesn't call them, so a removal is flagged).
PUBLIC_HELPERS = ["add_compress_type", "remove_compress_type"]

# Constructor keyword-only options main.py relies on by NOT passing (defaults).
CONSTRUCTOR_KWARGS = [
    "minimum_size",
    "zstd",
    "brotli",
    "gzip",
]


# --------------------------------------------------------------------------- #
# Minimal in-process ASGI harness (no server, no sockets)
# --------------------------------------------------------------------------- #


def _make_plain_app(body: bytes, content_type: str = "text/plain"):
    """A trivial ASGI app that returns `body` with status 200."""

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", content_type.encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


async def _drive(middleware, accept_encoding: bytes | None):
    """Run one GET request through the ASGI `middleware`, return
    (status, headers_dict, body_bytes). Pure in-memory, no network."""
    headers = []
    if accept_encoding is not None:
        headers.append((b"accept-encoding", accept_encoding))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    received = {"started": False}
    out_status = {"code": None}
    out_headers: list[tuple[bytes, bytes]] = []
    chunks: list[bytes] = []

    async def receive():
        # No request body.
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            received["started"] = True
            out_status["code"] = message["status"]
            out_headers.extend(message.get("headers", []))
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await middleware(scope, receive, send)
    hdr_dict = {k.decode().lower(): v.decode() for k, v in out_headers}
    return out_status["code"], hdr_dict, b"".join(chunks)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "starlette_compress"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_public_helpers_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, PUBLIC_HELPERS)


def test_compress_middleware_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.CompressMiddleware)


# --------------------------------------------------------------------------- #
# Constructor contract
# --------------------------------------------------------------------------- #


def test_constructor_accepts_app_positionally(depcheck):
    """main.py does add_middleware(CompressMiddleware): starlette will call
    CompressMiddleware(app). The first parameter must be the ASGI app and be
    passable positionally."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.CompressMiddleware.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params, "CompressMiddleware.__init__ takes no parameters besides self"
    first = params[0]
    assert first.name == "app"
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), "CompressMiddleware first parameter (app) is no longer positional"


def test_constructor_keyword_surface(depcheck):
    """The documented tuning knobs must still be accepted; main.py relies on
    their defaults by not passing them."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.CompressMiddleware.__init__, CONSTRUCTOR_KWARGS)


def test_constructor_defaults_unchanged(depcheck):
    """Pin the defaults the backend implicitly depends on (compression on,
    a non-zero minimum_size floor). A silent flip to gzip=False would mean
    responses stop compressing without any code change in Open WebUI."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.CompressMiddleware.__init__)
    params = sig.parameters
    gzip_p = params.get("gzip")
    if gzip_p is not None and gzip_p.default is not inspect.Parameter.empty:
        assert gzip_p.default is True, "CompressMiddleware default gzip flipped off"
    min_p = params.get("minimum_size")
    if min_p is not None and min_p.default is not inspect.Parameter.empty:
        assert isinstance(min_p.default, int) and min_p.default > 0


def test_instantiable_with_defaults(depcheck):
    """Constructing with just an app (the main.py path) must not raise."""
    mod = depcheck.load(IMPORT_NAME)

    async def app(scope, receive, send):
        return None

    inst = mod.CompressMiddleware(app)
    assert callable(inst)
    assert inspect.iscoroutinefunction(inst.__call__)


# --------------------------------------------------------------------------- #
# Behavioural: end-to-end ASGI compression (offline)
# --------------------------------------------------------------------------- #


def test_large_body_gzip_compressed_and_roundtrips(depcheck):
    """With Accept-Encoding: gzip and a body well above minimum_size, the
    middleware must set Content-Encoding: gzip and emit a body that gunzips
    back to the original bytes."""
    mod = depcheck.load(IMPORT_NAME)
    # Compressible payload comfortably above the default minimum_size (500).
    original = (b"open-webui compression contract " * 200).strip()
    assert len(original) > 1000
    mw = mod.CompressMiddleware(_make_plain_app(original))

    status, headers, body = _run(_drive(mw, accept_encoding=b"gzip"))

    assert status == 200
    assert headers.get("content-encoding") == "gzip", (
        f"expected gzip encoding, got headers={headers}"
    )
    # The wire body is the gzip stream; decompressing yields the original.
    assert gzip.decompress(body) == original
    # And it actually got smaller (the whole point).
    assert len(body) < len(original)


def test_small_body_passed_through_uncompressed(depcheck):
    """A body below minimum_size must NOT be compressed even when the client
    advertises gzip — pin the minimum-size floor behaviour."""
    mod = depcheck.load(IMPORT_NAME)
    tiny = b"hi"
    mw = mod.CompressMiddleware(_make_plain_app(tiny))

    status, headers, body = _run(_drive(mw, accept_encoding=b"gzip"))

    assert status == 200
    assert "content-encoding" not in headers, (
        "tiny body should not be compressed (below minimum_size)"
    )
    assert body == tiny


def test_no_accept_encoding_leaves_body_untouched(depcheck):
    """A client that sends no Accept-Encoding must receive the raw body with
    no Content-Encoding header, regardless of size."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"x" * 5000
    mw = mod.CompressMiddleware(_make_plain_app(original))

    status, headers, body = _run(_drive(mw, accept_encoding=None))

    assert status == 200
    assert "content-encoding" not in headers
    assert body == original


def test_identity_only_accept_encoding_not_compressed(depcheck):
    """Accept-Encoding: identity (explicitly no compression) must be honored:
    a large body comes back uncompressed."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"y" * 5000
    mw = mod.CompressMiddleware(_make_plain_app(original))

    status, headers, body = _run(_drive(mw, accept_encoding=b"identity"))

    assert status == 200
    assert headers.get("content-encoding") in (None, "identity")
    if headers.get("content-encoding") is None:
        assert body == original


def test_parse_accept_encoding_helper_if_present(depcheck):
    """The library exposes a parse_accept_encoding helper; if present it must
    recognise gzip in a typical browser Accept-Encoding header."""
    mod = depcheck.load(IMPORT_NAME)
    parse = getattr(mod, "parse_accept_encoding", None)
    if parse is None:
        pytest.skip("parse_accept_encoding not exported in this version")
    result = parse("gzip, deflate, br")
    # Return type is version-dependent (set/frozenset/etc); just assert it
    # reports gzip support in a membership-testable way.
    assert "gzip" in result
