"""Dependency contract: elasticsearch (the official Python client).

elasticsearch backs one of Open WebUI's pluggable vector stores,
``retrieval/vector/dbs/elasticsearch.py``:

    from elasticsearch import BadRequestError, Elasticsearch
    from elasticsearch.helpers import bulk, scan
    ...
    self.client = Elasticsearch(
        hosts=[ELASTICSEARCH_URL],
        ca_certs=ELASTICSEARCH_CA_CERTS,
        api_key=ELASTICSEARCH_API_KEY,
        cloud_id=ELASTICSEARCH_CLOUD_ID,
        basic_auth=(...),
        ssl_assert_fingerprint=SSL_ASSERT_FINGERPRINT,
    )

and then drives it with: ``client.search(index=, ...)``,
``client.count(index=, ...)``, ``client.delete_by_query(index=, ...)``, the
``client.indices.{get,exists,delete,create}(index=, ...)`` namespace, the
``bulk(client, actions)`` / ``scan(...)`` helpers, and catches
``BadRequestError``.

A real client call needs a running Elasticsearch cluster, which this offline
suite must NOT touch — but the client constructs *lazily* (no socket until the
first request), so the contract can be verified by constructing the client with
the backend's exact keyword set and asserting the method/namespace/exception
surface, all without a server. This module pins that, so an elasticsearch
client major bump (the 8 -> 9 line removes/renames public surface) that dropped
a constructor keyword, a client method, the ``indices`` namespace, the bulk/scan
helpers, or ``BadRequestError`` fails loudly here instead of at vector-search
time.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks +
offline lazy construction (NO cluster, NO network). Uses the `depcheck` fixture
from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "elasticsearch"
DIST_NAME = "elasticsearch"

# Top-level names the backend imports from `elasticsearch`.
USED_SYMBOLS = ["Elasticsearch", "BadRequestError"]

# Client methods the backend calls on the Elasticsearch instance.
CLIENT_METHODS = ["search", "count", "delete_by_query", "index", "delete"]

# The indices sub-namespace methods the backend calls.
INDICES_METHODS = ["get", "exists", "delete", "create"]

# Constructor keywords the backend passes to Elasticsearch(...).
INIT_KWARGS = [
    "hosts",
    "ca_certs",
    "api_key",
    "cloud_id",
    "basic_auth",
    "ssl_assert_fingerprint",
]


def _client(depcheck):
    """Construct an Elasticsearch client lazily (no request -> no server). This
    only builds the client object; nothing here opens a socket."""
    es = depcheck.load(IMPORT_NAME)
    return es.Elasticsearch(hosts=["http://localhost:9200"])


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "elasticsearch"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """``Elasticsearch`` and ``BadRequestError`` must resolve on the package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_elasticsearch_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.Elasticsearch)


def test_helpers_submodule_and_functions(depcheck):
    """The backend does ``from elasticsearch.helpers import bulk, scan`` — the
    helpers submodule must import and expose both as callables."""
    depcheck.load(IMPORT_NAME)
    helpers = depcheck.try_load("elasticsearch.helpers")
    assert helpers is not None, "elasticsearch.helpers no longer importable"
    assert callable(getattr(helpers, "bulk", None)), "helpers.bulk missing"
    assert callable(getattr(helpers, "scan", None)), "helpers.scan missing"


# --------------------------------------------------------------------------- #
# BadRequestError — the caught exception
# --------------------------------------------------------------------------- #
def test_bad_request_error_is_exception(depcheck):
    """The client wraps a 400 as ``BadRequestError``; the backend catches it, so
    it must subclass Exception."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.BadRequestError, Exception)


def test_bad_request_error_subclasses_api_error(depcheck):
    """BadRequestError sits under the client's ``ApiError`` base; pin that so a
    broad ``except ApiError`` (and the narrower catch the backend uses) both
    stay valid across the 8 -> 9 exception reshuffle."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "ApiError"), "elasticsearch.ApiError missing"
    assert issubclass(mod.BadRequestError, mod.ApiError), (
        "BadRequestError no longer subclasses ApiError"
    )


# --------------------------------------------------------------------------- #
# Elasticsearch constructor — lazy, accepts the backend's full kwarg set
# --------------------------------------------------------------------------- #
def test_constructor_accepts_backend_kwargs(depcheck):
    """The backend constructs with hosts/ca_certs/api_key/cloud_id/basic_auth/
    ssl_assert_fingerprint. Pin those keyword names on __init__ so the exact
    construction in ElasticsearchClient.__init__ stays valid."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Elasticsearch.__init__, INIT_KWARGS)


def test_constructs_lazily_without_connecting(depcheck):
    """``Elasticsearch(hosts=[...])`` must return a client object WITHOUT opening
    a connection (the socket is deferred to the first request). Construct with
    the backend's full kwarg set and assert we get a client — no server needed
    because no request is issued."""
    mod = depcheck.load(IMPORT_NAME)
    client = mod.Elasticsearch(
        hosts=["http://localhost:9200"],
        ca_certs=None,
        api_key=None,
        cloud_id=None,
        basic_auth=None,
        ssl_assert_fingerprint=None,
    )
    assert client is not None
    assert isinstance(client, mod.Elasticsearch)


# --------------------------------------------------------------------------- #
# Client method surface (on a lazily-constructed client — no requests)
# --------------------------------------------------------------------------- #
def test_client_methods_exist_and_callable(depcheck):
    """Every method the backend calls on the client must exist and be callable
    on a constructed instance. We never invoke them (that would hit the
    cluster)."""
    client = _client(depcheck)
    for name in CLIENT_METHODS:
        assert callable(getattr(client, name, None)), f"client.{name} missing/not callable"


def test_search_count_delete_by_query_accept_index(depcheck):
    """The backend calls ``search``/``count``/``delete_by_query`` with an
    ``index=`` keyword. Pin that ``index`` is an accepted parameter on each
    (these methods have explicit signatures, not **kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("search", "count", "delete_by_query"):
        depcheck.assert_params(getattr(mod.Elasticsearch, name), ["index"])


def test_indices_namespace_present(depcheck):
    """``client.indices`` is the index-admin namespace the backend uses. It is
    an *instance* attribute (lazily attached), so check it on a constructed
    client, and confirm it is callable-bearing (not None)."""
    client = _client(depcheck)
    assert hasattr(client, "indices"), "client.indices namespace missing"
    assert client.indices is not None


def test_indices_methods_exist_and_accept_index(depcheck):
    """``client.indices.{get,exists,delete,create}`` are all called with
    ``index=``. Pin they exist, are callable, and accept the index parameter."""
    client = _client(depcheck)
    indices = client.indices
    for name in INDICES_METHODS:
        method = getattr(indices, name, None)
        assert callable(method), f"client.indices.{name} missing/not callable"
        depcheck.assert_params(method, ["index"])


# --------------------------------------------------------------------------- #
# helpers.bulk / helpers.scan — the call shapes the backend uses
# --------------------------------------------------------------------------- #
def test_bulk_signature(depcheck):
    """The backend calls ``bulk(self.client, actions)`` — first two positionals
    are the client and the actions iterable. Pin those parameter names."""
    depcheck.load(IMPORT_NAME)
    helpers = depcheck.load("elasticsearch.helpers")
    depcheck.assert_params(helpers.bulk, ["client", "actions"])


def test_scan_signature(depcheck):
    """``scan`` is used to stream all hits; first parameter is the client, with
    a ``query`` parameter for the body. Pin those names."""
    depcheck.load(IMPORT_NAME)
    helpers = depcheck.load("elasticsearch.helpers")
    depcheck.assert_params(helpers.scan, ["client", "query"])
