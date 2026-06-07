"""Dependency contract: google-api-python-client (import name ``googleapiclient``).

``google-api-python-client`` is a pinned dependency
(``google-api-python-client==2.197.0`` in ``backend/requirements.txt`` and
``pyproject.toml``). The Open WebUI backend does **not** import
``googleapiclient`` anywhere in ``open_webui/`` — notably the Google
Programmable Search provider (``retrieval/web/google_pse.py``) talks to the
Custom Search REST endpoint with a plain HTTP GET via the shared session,
not through this client. It is carried as a declared dependency for Google
API integrations / transitive needs.

Because it is a version-pinned dependency with a stable, well-known public
surface, this module pins that surface and exercises the client's core
machinery OFFLINE — no network, no real Google service, no credentials:

  - the discovery entry points (``discovery.build`` /
    ``discovery.build_from_document``) and the error/http surface
    (``errors.HttpError``, ``http.HttpMock`` / ``HttpMockSequence`` /
    ``HttpRequest``);
  - ``build_from_document`` constructs a usable service object from a tiny
    in-memory discovery document, and the generated
    ``service.<resource>().<method>(...)`` chain produces an ``HttpRequest``
    with the correct method and templated URI — proving the request-builder
    contract end to end without ever issuing a request.

NOTE (version drift): the pin is ``2.197.0`` but the test venv has
``2.193.0`` installed. These tests validate against whatever is importable;
the mismatch is a packaging observation, not enforced here.

A google-api-python-client bump that moved ``build``/``HttpError`` or changed
the discovery-to-request machinery would fail here instead of surfacing only
when some Google integration is first exercised.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect
import json

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "googleapiclient"
DIST_NAME = "google-api-python-client"

# Public surface a googleapiclient consumer relies on.
USED_SYMBOLS = [
    "discovery.build",
    "discovery.build_from_document",
    "errors.HttpError",
    "errors.Error",
    "http.HttpRequest",
    "http.HttpMock",
    "http.HttpMockSequence",
]

# A minimal valid discovery document defining one resource + one GET method.
DISCOVERY_DOC = {
    "kind": "discovery#restDescription",
    "name": "demo",
    "version": "v1",
    "rootUrl": "https://demo.example/",
    "servicePath": "",
    "baseUrl": "https://demo.example/",
    "schemas": {},
    "resources": {
        "things": {
            "methods": {
                "get": {
                    "id": "demo.things.get",
                    "path": "things/{id}",
                    "httpMethod": "GET",
                    "parameters": {
                        "id": {
                            "location": "path",
                            "type": "string",
                            "required": True,
                        }
                    },
                }
            }
        }
    },
}


def _build_offline_service(depcheck):
    """Build a service from the in-memory discovery doc (no network)."""
    mod = depcheck.load(IMPORT_NAME)
    build_from_document = depcheck.resolve(mod, "discovery.build_from_document")
    return build_from_document(json.dumps(DISCOVERY_DOC), base="https://demo.example/")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "googleapiclient"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_build_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "discovery.build")
    depcheck.assert_callable(mod, "discovery.build_from_document")


def test_build_signature(depcheck):
    """discovery.build(serviceName, version, developerKey=, ...). Pin the
    leading positional params and the developerKey kwarg (the common API-key
    auth path)."""
    mod = depcheck.load(IMPORT_NAME)
    build = depcheck.resolve(mod, "discovery.build")
    sig = inspect.signature(build)
    params = list(sig.parameters.values())
    assert len(params) >= 2
    assert params[0].name == "serviceName"
    assert params[1].name == "version"
    depcheck.assert_params(build, ["developerKey"])


def test_httperror_is_exception(depcheck):
    """googleapiclient.errors.HttpError is what API calls raise on non-2xx;
    it must be a subclass of the package's base Error (and of Exception)."""
    mod = depcheck.load(IMPORT_NAME)
    http_error = depcheck.resolve(mod, "errors.HttpError")
    base_error = depcheck.resolve(mod, "errors.Error")
    assert inspect.isclass(http_error)
    assert issubclass(http_error, base_error)
    assert issubclass(http_error, Exception)


# --------------------------------------------------------------------------- #
# Behavioural: offline service construction + request building
# --------------------------------------------------------------------------- #


def test_build_from_document_returns_service(depcheck):
    """build_from_document constructs a service (Resource) object from a
    discovery doc with no network access."""
    service = _build_offline_service(depcheck)
    assert service is not None
    assert type(service).__name__ == "Resource"


def test_service_exposes_resource_accessor(depcheck):
    """The discovery doc's `things` resource becomes a callable accessor on
    the service object."""
    service = _build_offline_service(depcheck)
    assert hasattr(service, "things")
    assert callable(service.things)


def test_method_builds_http_request(depcheck):
    """service.things().get(id=...) builds an HttpRequest (it does NOT execute
    it) — the request-builder contract any googleapiclient call relies on."""
    mod = depcheck.load(IMPORT_NAME)
    http_request_cls = depcheck.resolve(mod, "http.HttpRequest")
    service = _build_offline_service(depcheck)
    request = service.things().get(id="42")
    assert isinstance(request, http_request_cls)


def test_built_request_has_correct_method_and_uri(depcheck):
    """The built request must carry the discovery-declared HTTP method and a
    URI with the path template filled in — proving parameter substitution
    works without any network round-trip."""
    service = _build_offline_service(depcheck)
    request = service.things().get(id="42")
    assert request.method == "GET"
    assert "things/42" in request.uri
    assert request.uri.startswith("https://demo.example/")


def test_httpmocksequence_is_constructible(depcheck):
    """HttpMockSequence is the documented offline test transport; pin it's
    constructible (consumers use it to stub responses without network)."""
    mod = depcheck.load(IMPORT_NAME)
    seq_cls = depcheck.resolve(mod, "http.HttpMockSequence")
    seq = seq_cls([({"status": "200"}, b"{}")])
    assert seq is not None
