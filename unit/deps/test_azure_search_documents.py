"""Dependency contract: azure-search-documents (import name ``azure.search.documents``).

azure-search-documents is the Azure AI Search client. Open WebUI's
``retrieval/web/azure.py`` (the "Azure AI Search" web-search provider) imports
``SearchClient`` from ``azure.search.documents`` and ``AzureKeyCredential``
from ``azure.core.credentials`` lazily (inside ``search_azure`` behind an
ImportError guard, since the package is optional), then:

    credential = AzureKeyCredential(api_key)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name,
                                 credential=credential)
    results = search_client.search(search_text=query, top=count)
    for result in results:
        result_dict = dict(result)  # rows behave like mappings

This module pins exactly that surface so a bump that renamed/changed any of it
fails loudly here instead of as a runtime ImportError/TypeError the moment an
admin runs an Azure web search. Two layers, mirroring test_chromadb.py: import
+ symbol/signature checks, plus an offline construction contract that builds a
real ``AzureKeyCredential`` and ``SearchClient`` (the client is LAZY — it opens
no socket until ``.search()`` is actually called, which we never do). NO Azure
service is contacted and NO query is issued.

Pattern mirrors test_requests.py / test_chromadb.py. Uses the ``depcheck``
fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "azure.search.documents"
DIST_NAME = "azure-search-documents"


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`azure.search.documents` must import (skip cleanly if absent — it's an
    optional dependency gated by an ImportError guard in azure.py)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "azure.search.documents"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling and
    this suite agree on what's under test. NOTE: requirements.txt pins
    `azure-search-documents==12.0.0`; the resolved version may differ (the SDK's
    GA line is 11.x), which this surfaces for the bump tooling to reconcile."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_azure_core_credentials_importable(depcheck):
    """azure.py: `from azure.core.credentials import AzureKeyCredential`. The
    credentials module (from the azure-core dependency) must be importable and
    expose AzureKeyCredential."""
    depcheck.load(IMPORT_NAME)
    cred_mod = depcheck.load("azure.core.credentials")
    assert hasattr(cred_mod, "AzureKeyCredential"), (
        "azure.core.credentials.AzureKeyCredential is gone; azure.py's API-key "
        "auth path would fail to import."
    )


# ---------------------------------------------------------------------------
# Symbol-existence + callability (API surface)
# ---------------------------------------------------------------------------


def test_search_client_exists_and_callable(depcheck):
    """azure.py: `from azure.search.documents import SearchClient`. It must exist
    and be a callable class."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "SearchClient")


def test_azure_key_credential_callable(depcheck):
    """AzureKeyCredential(api_key) is constructed before the client; pin it's a
    callable class."""
    depcheck.load(IMPORT_NAME)
    cred_mod = depcheck.load("azure.core.credentials")
    assert callable(cred_mod.AzureKeyCredential)


# ---------------------------------------------------------------------------
# Constructor + method signature contracts
# ---------------------------------------------------------------------------


def test_search_client_init_accepts_our_kwargs(depcheck):
    """azure.py: SearchClient(endpoint=, index_name=, credential=). All three
    kwargs must remain accepted on the constructor."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.SearchClient.__init__, ["endpoint", "index_name", "credential"])


def test_search_method_accepts_search_text_and_top(depcheck):
    """azure.py: search_client.search(search_text=query, top=count). Both kwargs
    must remain accepted on the search method (a rename would break the query)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.SearchClient.search, ["search_text", "top"])


# ---------------------------------------------------------------------------
# Behavioural: offline construction — lazy client, NO Azure contact.
# ---------------------------------------------------------------------------


def test_behaviour_construct_credential_and_client_offline(depcheck):
    """Mirror azure.py's setup: build an AzureKeyCredential then a SearchClient
    with endpoint/index_name/credential. The client is lazy — constructing it
    opens no socket and contacts no Azure service (a query would, but we never
    issue one). Pin that construction succeeds and exposes a callable search()."""
    mod = depcheck.load(IMPORT_NAME)
    cred_mod = depcheck.load("azure.core.credentials")

    credential = cred_mod.AzureKeyCredential("fake-query-key")
    client = mod.SearchClient(
        endpoint="https://example.search.windows.net",
        index_name="my-index",
        credential=credential,
    )
    assert client is not None
    assert hasattr(client, "search"), "SearchClient has no search() method"
    assert callable(client.search)


def test_behaviour_credential_holds_key(depcheck):
    """AzureKeyCredential stores the key it was given (azure.py passes the admin/
    query key straight in). Pin that the key round-trips via the documented
    `.key` attribute, so the auth value is actually carried."""
    depcheck.load(IMPORT_NAME)
    cred_mod = depcheck.load("azure.core.credentials")
    credential = cred_mod.AzureKeyCredential("secret-key-123")
    assert credential.key == "secret-key-123", (
        "AzureKeyCredential no longer exposes the key via .key; the API-key auth "
        "carried into SearchClient would be opaque/broken."
    )


def test_behaviour_search_client_has_close(depcheck):
    """SearchClient is a context-manager-capable resource (azure-core clients
    expose close()/__enter__/__exit__). Pin close() exists so the client can be
    cleaned up without leaking the underlying transport."""
    mod = depcheck.load(IMPORT_NAME)
    cred_mod = depcheck.load("azure.core.credentials")
    client = mod.SearchClient(
        endpoint="https://example.search.windows.net",
        index_name="my-index",
        credential=cred_mod.AzureKeyCredential("k"),
    )
    try:
        assert hasattr(client, "close") and callable(client.close)
    finally:
        # Closing an unconnected client just tears down the (idle) transport.
        try:
            client.close()
        except Exception:
            pass
