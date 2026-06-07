"""Dependency contract: azure-storage-blob (import ``azure.storage.blob``).

Open WebUI's ``storage/provider.py`` has an ``AzureStorageProvider`` that
stores uploaded files in Azure Blob Storage. It uses a precise slice of the
azure-storage-blob SDK:

    from azure.storage.blob import BlobServiceClient

    bsc = BlobServiceClient(account_url=endpoint, credential=storage_key)
    # or credential=DefaultAzureCredential() when no key is configured
    container_client = bsc.get_container_client(container_name)

    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(contents, overwrite=True)
    data = blob_client.download_blob().readall()
    blob_client.delete_blob()

    for blob in container_client.list_blobs():
        container_client.delete_blob(blob.name)

and catches ``azure.core.exceptions.ResourceNotFoundError`` (a sibling
``azure-core`` distribution) on download/delete misses.

Every client in this SDK constructs LAZILY — no network call happens until
a method that actually transfers data is invoked. This module therefore
builds the full client graph (service -> container -> blob) offline with a
throwaway account URL + fake credential, and pins the method surface the
provider drives, WITHOUT ever opening a connection. The adjacent
``azure.core.exceptions`` / ``azure.identity`` symbols the provider imports
are checked too (they ride in with this SDK), but skip cleanly if those
sibling packages are absent.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "azure.storage.blob"
DIST_NAME = "azure-storage-blob"

USED_SYMBOLS = [
    "BlobServiceClient",
    "ContainerClient",
    "BlobClient",
]

FAKE_ACCOUNT_URL = "https://unit-test-acct.blob.core.windows.net"
FAKE_KEY = "ZmFrZS1rZXktbm90LXJlYWw="  # base64 'fake-key-not-real'


def _service_client(mod):
    """Build a BlobServiceClient offline (no network on construction)."""
    return mod.BlobServiceClient(account_url=FAKE_ACCOUNT_URL, credential=FAKE_KEY)


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "azure.storage.blob"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_blobserviceclient_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.BlobServiceClient, type)


def test_blobserviceclient_ctor_accepts_account_url_and_credential(depcheck):
    """The provider calls BlobServiceClient(account_url=, credential=). Both
    parameters must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.BlobServiceClient.__init__)
    params = sig.parameters
    assert "account_url" in params
    assert "credential" in params


# ---------------------------------------------------------------------------
# Offline client graph: service -> container -> blob (NO network).
# ---------------------------------------------------------------------------


def test_service_client_constructs_offline(depcheck):
    """BlobServiceClient(account_url, credential=key) must construct without
    opening a connection — the provider builds it in __init__ before any I/O."""
    mod = depcheck.load(IMPORT_NAME)
    bsc = _service_client(mod)
    assert bsc is not None
    assert isinstance(bsc, mod.BlobServiceClient)


def test_get_container_client(depcheck):
    """bsc.get_container_client(name) returns a ContainerClient lazily."""
    mod = depcheck.load(IMPORT_NAME)
    bsc = _service_client(mod)
    cc = bsc.get_container_client("open-webui")
    assert isinstance(cc, mod.ContainerClient)


def test_container_get_blob_client(depcheck):
    """container_client.get_blob_client(filename) returns a BlobClient lazily."""
    mod = depcheck.load(IMPORT_NAME)
    cc = _service_client(mod).get_container_client("open-webui")
    bc = cc.get_blob_client("upload.bin")
    assert isinstance(bc, mod.BlobClient)


def test_blob_client_method_surface(depcheck):
    """The BlobClient must expose upload_blob / download_blob / delete_blob —
    the three operations the provider performs per file."""
    mod = depcheck.load(IMPORT_NAME)
    bc = _service_client(mod).get_container_client("c").get_blob_client("f")
    for name in ("upload_blob", "download_blob", "delete_blob"):
        assert callable(getattr(bc, name, None)), f"BlobClient.{name} missing/not callable"


def test_container_client_method_surface(depcheck):
    """ContainerClient must expose list_blobs / delete_blob / get_blob_client —
    used by delete_all_files()."""
    mod = depcheck.load(IMPORT_NAME)
    cc = _service_client(mod).get_container_client("c")
    for name in ("list_blobs", "delete_blob", "get_blob_client"):
        assert callable(getattr(cc, name, None)), f"ContainerClient.{name} missing/not callable"


def test_upload_blob_accepts_overwrite_kwarg(depcheck):
    """The provider calls upload_blob(contents, overwrite=True). upload_blob
    takes data positionally plus **kwargs (where overwrite lives); assert the
    data param and the kwargs passthrough."""
    mod = depcheck.load(IMPORT_NAME)
    bc = _service_client(mod).get_container_client("c").get_blob_client("f")
    sig = inspect.signature(bc.upload_blob)
    params = sig.parameters
    # First positional is the data payload.
    assert "data" in params or list(params)[0] != "self"
    # overwrite is passed via **kwargs; require a VAR_KEYWORD to exist.
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "upload_blob no longer accepts **kwargs (overwrite=True would break)"
    )


def test_from_connection_string_available(depcheck):
    """BlobServiceClient.from_connection_string is the common alternative
    constructor; pin it as a callable classmethod (no call -> no network)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(getattr(mod.BlobServiceClient, "from_connection_string", None))


def test_download_blob_returns_readall_capable_object(depcheck):
    """The provider does download_blob().readall(). We cannot call
    download_blob (network), but the downloader type must expose readall."""
    depcheck.load(IMPORT_NAME)  # skip cleanly if the SDK is absent
    downloader_mod = depcheck.try_load("azure.storage.blob._download")
    if downloader_mod is None:
        pytest.skip("azure.storage.blob._download not importable in this env")
    cls = getattr(downloader_mod, "StorageStreamDownloader", None)
    assert cls is not None, "StorageStreamDownloader type not found"
    assert callable(getattr(cls, "readall", None)), "download result lost .readall()"


# ---------------------------------------------------------------------------
# Adjacent imports the provider relies on (sibling azure-* distributions).
# ---------------------------------------------------------------------------


def test_resource_not_found_error_hierarchy(depcheck):
    """The provider catches azure.core.exceptions.ResourceNotFoundError on
    download/delete misses; it must remain an AzureError subclass."""
    exc_mod = depcheck.try_load("azure.core.exceptions")
    if exc_mod is None:
        pytest.skip("azure-core not installed in this env")
    rnf = getattr(exc_mod, "ResourceNotFoundError", None)
    assert rnf is not None, "ResourceNotFoundError missing from azure.core.exceptions"
    assert issubclass(rnf, getattr(exc_mod, "AzureError"))
    assert issubclass(rnf, Exception)


def test_default_azure_credential_constructs_offline(depcheck):
    """When no storage key is configured the provider passes
    DefaultAzureCredential() as the credential. It must be importable and
    construct without performing a token exchange (that is deferred to first
    use)."""
    ident = depcheck.try_load("azure.identity")
    if ident is None:
        pytest.skip("azure-identity not installed in this env")
    DefaultAzureCredential = getattr(ident, "DefaultAzureCredential", None)
    assert DefaultAzureCredential is not None
    cred = DefaultAzureCredential()
    assert cred is not None


def test_service_client_accepts_token_credential(depcheck):
    """BlobServiceClient must accept a TokenCredential (DefaultAzureCredential)
    as `credential` — the managed-identity path. Construct it offline."""
    mod = depcheck.load(IMPORT_NAME)
    ident = depcheck.try_load("azure.identity")
    if ident is None:
        pytest.skip("azure-identity not installed in this env")
    cred = ident.DefaultAzureCredential()
    bsc = mod.BlobServiceClient(account_url=FAKE_ACCOUNT_URL, credential=cred)
    assert isinstance(bsc, mod.BlobServiceClient)
