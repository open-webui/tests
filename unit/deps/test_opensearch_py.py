"""Dependency contract: opensearch-py (import name ``opensearchpy``).

OpenSearch is one of Open WebUI's pluggable vector stores. ``OpenSearchClient``
in ``retrieval/vector/dbs/opensearch.py`` constructs a single
``opensearchpy.OpenSearch`` client (``hosts=``, ``use_ssl=``, ``verify_certs=``,
``http_auth=``) and drives a fixed slice of its API:

  * ``client.indices.create / exists / delete / refresh / get`` (the
    knn_vector index lifecycle — each "collection" is an index), and
  * ``client.search(index=, body=, size=)`` for both kNN script-score search
    and metadata-filter ``query``, plus ``client.delete_by_query(index=,
    body=)`` for filtered deletes, and
  * ``opensearchpy.helpers.bulk(client, actions)`` to batch index/update/delete
    operations.

This module pins exactly that surface so an opensearch-py bump (the 2.x -> 3.x
major in this env removes/renames public API) fails loudly here instead of as
a runtime ``AttributeError`` / ``TypeError`` deep in an ingest or search
request. Two layers, mirroring test_chromadb.py / test_redis.py:
symbol-existence + signature checks, plus offline construction contracts that
build a real ``OpenSearch`` client (lazy — it opens NO socket until a request
is issued) and assert the ``.indices`` namespace, the client methods, and that
the exact kwargs the backend passes (including ``size=`` on ``search``) are
accepted by the ``@query_params`` layer rather than rejected as unknown.

NO network and NO OpenSearch server are ever contacted: we never issue a
request, only inspect/construct. Where a behavioural check must prove a kwarg
is *accepted*, it confirms the call fails at the transport/connection layer
(a connection error) and NOT with a ``TypeError`` from kwarg validation.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "opensearchpy"
DIST_NAME = "opensearch-py"

# Top-level symbols opensearch.py (and its helpers import) resolves.
TOP_LEVEL_SYMBOLS = [
    "OpenSearch",  # `from opensearchpy import OpenSearch`
    "helpers",  # `from opensearchpy.helpers import bulk`
    "exceptions",  # error hierarchy
]

# Client methods opensearch.py calls on the OpenSearch instance.
CLIENT_METHODS = ["search", "delete_by_query", "indices"]

# IndicesClient methods opensearch.py calls via client.indices.*.
INDICES_METHODS = ["create", "exists", "delete", "refresh", "get"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`opensearchpy` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "opensearchpy"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `opensearchpy.*` symbol the codebase resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_opensearch_is_callable(depcheck):
    """opensearch.py: OpenSearch(hosts=[...], ...). Must be a callable factory."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "OpenSearch")


def test_helpers_bulk_exists_and_callable(depcheck):
    """opensearch.py: `from opensearchpy.helpers import bulk` then
    bulk(self.client, actions). The helper must exist and be callable."""
    depcheck.load(IMPORT_NAME)
    helpers = depcheck.load("opensearchpy.helpers")
    depcheck.assert_symbols(helpers, ["bulk"])
    depcheck.assert_callable(helpers, "bulk")


def test_bulk_signature(depcheck):
    """bulk(client, actions, ...) — the first two positionals opensearch.py
    passes (`bulk(self.client, actions)`) must remain accepted."""
    depcheck.load(IMPORT_NAME)
    helpers = depcheck.load("opensearchpy.helpers")
    depcheck.assert_params(helpers.bulk, ["client", "actions"])


def test_exception_hierarchy_has_base(depcheck):
    """opensearch.py wraps every call in broad try/except; the package's
    exceptions module must keep a common base (OpenSearchException) so a
    catch-all over the driver's errors stays meaningful."""
    depcheck.load(IMPORT_NAME)
    ex = depcheck.load("opensearchpy.exceptions")
    assert hasattr(ex, "OpenSearchException"), "opensearchpy.exceptions.OpenSearchException is gone"
    assert hasattr(ex, "NotFoundError"), "opensearchpy.exceptions.NotFoundError is gone"


# ---------------------------------------------------------------------------
# Constructor signature contract
# ---------------------------------------------------------------------------


def test_opensearch_init_accepts_hosts(depcheck):
    """opensearch.py passes hosts=[OPENSEARCH_URI] as the first named arg; the
    `hosts` parameter must remain accepted on the constructor."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.OpenSearch.__init__, ["hosts"])


def test_opensearch_init_keeps_var_kwargs(depcheck):
    """use_ssl / verify_certs / http_auth flow through **kwargs to the transport.
    Pin that OpenSearch.__init__ keeps **kwargs (so those connection kwargs are
    not rejected as unknown)."""
    mod = depcheck.load(IMPORT_NAME)
    params = inspect.signature(mod.OpenSearch.__init__).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "OpenSearch.__init__ dropped **kwargs; use_ssl/verify_certs/http_auth would be rejected."
    )


# ---------------------------------------------------------------------------
# Offline client construction — lazy, opens NO socket.
# ---------------------------------------------------------------------------


def _client(mod):
    """An OpenSearch client mirroring opensearch.py's constructor.

    opensearch-py is lazy: constructing the client builds the transport/
    connection-pool objects but does NOT open a socket until a request is
    actually issued. None of these tests issue an indexing/search request that
    would reach the (absent) server."""
    return mod.OpenSearch(
        hosts=["http://localhost:9200"],
        use_ssl=False,
        verify_certs=False,
        http_auth=("user", "pass"),
    )


def test_construct_client_offline(depcheck):
    """OpenSearchClient.__init__ builds the client with use_ssl/verify_certs/
    http_auth. Constructing with those kwargs must succeed without connecting."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    assert client is not None


def test_client_methods_exist(depcheck):
    """Every client method/namespace opensearch.py uses must exist on a real
    client (search, delete_by_query, indices)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    for name in CLIENT_METHODS:
        assert hasattr(client, name), f"OpenSearch client missing {name!r}"
    assert callable(client.search)
    assert callable(client.delete_by_query)


def test_indices_namespace_methods_exist(depcheck):
    """opensearch.py drives the index lifecycle via client.indices.create /
    exists / delete / refresh / get. All must exist and be callable."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    indices = client.indices
    for name in INDICES_METHODS:
        assert hasattr(indices, name), f"client.indices missing {name!r}"
        assert callable(getattr(indices, name)), f"client.indices.{name} not callable"


# ---------------------------------------------------------------------------
# Method-signature contracts (the kwargs opensearch.py passes)
# ---------------------------------------------------------------------------


def test_search_accepts_index_and_body(depcheck):
    """opensearch.py: client.search(index=, body=). Both kwargs must be accepted
    by the search method's signature."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    depcheck.assert_params(client.search, ["index", "body"])


def test_delete_by_query_accepts_index_and_body(depcheck):
    """opensearch.py: client.delete_by_query(index=, body=)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    depcheck.assert_params(client.delete_by_query, ["index", "body"])


def test_indices_create_accepts_index_and_body(depcheck):
    """opensearch.py: client.indices.create(index=, body=) to make the knn index."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    depcheck.assert_params(client.indices.create, ["index", "body"])


def test_indices_exists_accepts_index(depcheck):
    """has_collection: client.indices.exists(index=...)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    depcheck.assert_params(client.indices.exists, ["index"])


# ---------------------------------------------------------------------------
# Behavioural: prove `size=` on search is ACCEPTED (query() passes it).
# opensearch-py validates query-param kwargs up front via @query_params, so an
# unknown kwarg raises TypeError BEFORE any I/O. A *recognised* kwarg instead
# proceeds to the transport and fails with a connection error against the
# (absent) server. We assert the latter — kwarg accepted, no TypeError.
# ---------------------------------------------------------------------------


def test_search_accepts_size_query_param(depcheck):
    """query() calls client.search(index=, body=, size=size). `size` is a
    query-string param validated by @query_params. Passing it must NOT raise a
    TypeError (unknown-kwarg); it should instead fail later at the transport
    (connection error) because no server is running. That distinguishes
    'kwarg accepted' from 'kwarg removed'."""
    mod = depcheck.load(IMPORT_NAME)
    ex = depcheck.load("opensearchpy.exceptions")
    client = _client(mod)
    body = {"query": {"match_all": {}}, "_source": ["text", "metadata"]}
    with pytest.raises(BaseException) as excinfo:
        client.search(index="open_webui_x", body=body, size=5, request_timeout=0.05)
    # The failure must be a connection/transport failure, never a TypeError
    # about an unexpected `size` keyword.
    assert not isinstance(excinfo.value, TypeError), (
        f"client.search rejected size= as an unknown kwarg ({excinfo.value!r}); "
        "query() in opensearch.py passes size= and would break."
    )
    # Sanity: it really is a connection-class failure (server absent).
    conn_err = getattr(ex, "ConnectionError", None)
    if conn_err is not None:
        assert isinstance(excinfo.value, (conn_err, OSError, ConnectionError)) or isinstance(
            excinfo.value, getattr(ex, "OpenSearchException", Exception)
        )


def test_search_rejects_truly_unknown_kwarg(depcheck):
    """Control for the test above: a genuinely bogus query-param kwarg DOES
    raise TypeError up front (proving @query_params still validates, so the
    size= acceptance test is meaningful and not just swallowing everything)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _client(mod)
    with pytest.raises(TypeError):
        client.search(
            index="open_webui_x",
            body={"query": {"match_all": {}}},
            this_is_not_a_real_param=123,
        )
