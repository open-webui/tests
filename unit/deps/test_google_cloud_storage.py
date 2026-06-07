"""Dependency contract: google-cloud-storage (import ``google.cloud.storage``).

Open WebUI's ``storage/provider.py`` has a ``GCSStorageProvider`` backing
file storage with Google Cloud Storage. It uses a precise slice of the SDK:

    from google.cloud import storage
    from google.cloud.exceptions import GoogleCloudError, NotFound

    # with an explicit service-account JSON:
    gcs_client = storage.Client.from_service_account_info(info=<dict>)
    # or Application Default Credentials:
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(GCS_BUCKET_NAME)

    blob = bucket.blob(filename)
    blob.upload_from_filename(file_path)
    blob = bucket.get_blob(filename)
    blob.download_to_filename(local_file_path)
    blob.delete()
    for blob in bucket.list_blobs():
        blob.delete()

and catches ``GoogleCloudError`` (upload) / ``NotFound`` (download, delete).

Every object in this SDK is built lazily — construction performs no network
I/O; only methods that transfer data hit GCS. This module builds the full
client graph (client -> bucket -> blob) OFFLINE using anonymous credentials
(so no Application Default Credentials lookup or token exchange happens) and
pins the exact method surface the provider drives, WITHOUT connecting. The
``google.cloud.exceptions`` hierarchy the provider catches is pinned too.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "google.cloud.storage"
DIST_NAME = "google-cloud-storage"

FAKE_BUCKET = "open-webui-unit-test-bucket"


def _anon_client(depcheck, mod):
    """Build a storage.Client offline with anonymous credentials (no ADC
    lookup, no token exchange)."""
    auth = depcheck.try_load("google.auth.credentials")
    if auth is None or not hasattr(auth, "AnonymousCredentials"):
        pytest.skip("google.auth.credentials.AnonymousCredentials unavailable")
    return mod.Client(project="unit-test-project", credentials=auth.AnonymousCredentials())


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "google.cloud.storage"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_client_symbol_exists(depcheck):
    """`from google.cloud import storage` then `storage.Client`."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "Client")
    assert isinstance(mod.Client, type)


def test_client_ctor_accepts_project_and_credentials(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.Client.__init__)
    assert "project" in sig.parameters
    assert "credentials" in sig.parameters


def test_from_service_account_info_is_callable(depcheck):
    """The provider calls storage.Client.from_service_account_info(info=<dict>)
    when GOOGLE_APPLICATION_CREDENTIALS_JSON is set. The classmethod must
    remain callable (no call here -> no credential validation/network)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(getattr(mod.Client, "from_service_account_info", None))
    sig = inspect.signature(mod.Client.from_service_account_info)
    assert list(sig.parameters)[0] == "info"


# ---------------------------------------------------------------------------
# Offline client graph: Client -> bucket -> blob (NO network).
# ---------------------------------------------------------------------------


def test_client_constructs_offline_with_anon_creds(depcheck):
    """A Client built with anonymous credentials constructs without an ADC
    lookup — proving construction is side-effect-free (the provider builds the
    client in __init__ before any transfer)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _anon_client(depcheck, mod)
    assert client is not None
    assert isinstance(client, mod.Client)


def test_client_bucket_returns_bucket(depcheck):
    """gcs_client.bucket(name) returns a Bucket lazily (no existence check)."""
    mod = depcheck.load(IMPORT_NAME)
    client = _anon_client(depcheck, mod)
    bucket = client.bucket(FAKE_BUCKET)
    assert bucket is not None
    assert bucket.name == FAKE_BUCKET


def test_bucket_blob_returns_blob(depcheck):
    """bucket.blob(filename) returns a Blob lazily (the upload handle)."""
    mod = depcheck.load(IMPORT_NAME)
    bucket = _anon_client(depcheck, mod).bucket(FAKE_BUCKET)
    blob = bucket.blob("upload.bin")
    assert blob is not None
    assert blob.name == "upload.bin"


def test_blob_method_surface(depcheck):
    """The Blob must expose the transfer/delete methods the provider calls:
    upload_from_filename / download_to_filename / delete."""
    mod = depcheck.load(IMPORT_NAME)
    blob = _anon_client(depcheck, mod).bucket(FAKE_BUCKET).blob("f")
    for name in ("upload_from_filename", "download_to_filename", "delete"):
        assert callable(getattr(blob, name, None)), f"Blob.{name} missing/not callable"


def test_bucket_method_surface(depcheck):
    """The Bucket must expose blob / get_blob / list_blobs — the provider uses
    get_blob to fetch a download handle and list_blobs to enumerate for
    deletion."""
    mod = depcheck.load(IMPORT_NAME)
    bucket = _anon_client(depcheck, mod).bucket(FAKE_BUCKET)
    for name in ("blob", "get_blob", "list_blobs"):
        assert callable(getattr(bucket, name, None)), f"Bucket.{name} missing/not callable"


def test_blob_upload_from_filename_signature(depcheck):
    """upload_from_filename(filename, ...) — the local path must remain its
    first argument (the provider passes the temp file path positionally)."""
    mod = depcheck.load(IMPORT_NAME)
    blob = _anon_client(depcheck, mod).bucket(FAKE_BUCKET).blob("f")
    sig = inspect.signature(blob.upload_from_filename)
    assert list(sig.parameters)[0] == "filename"


# ---------------------------------------------------------------------------
# google.cloud.exceptions — the hierarchy the provider catches.
# ---------------------------------------------------------------------------


def test_exceptions_importable(depcheck):
    """`from google.cloud.exceptions import GoogleCloudError, NotFound`."""
    exc_mod = depcheck.try_load("google.cloud.exceptions")
    if exc_mod is None:
        pytest.skip("google.cloud.exceptions not importable in this env")
    assert hasattr(exc_mod, "GoogleCloudError")
    assert hasattr(exc_mod, "NotFound")


def test_notfound_subclasses_googlecloud_error(depcheck):
    """The provider catches GoogleCloudError broadly (upload) and NotFound
    specifically (download/delete). NotFound must be a GoogleCloudError so the
    broad handler would also catch it, and both must be Exceptions."""
    exc_mod = depcheck.try_load("google.cloud.exceptions")
    if exc_mod is None:
        pytest.skip("google.cloud.exceptions not importable in this env")
    assert issubclass(exc_mod.NotFound, exc_mod.GoogleCloudError)
    assert issubclass(exc_mod.GoogleCloudError, Exception)
    assert issubclass(exc_mod.NotFound, Exception)


def test_googlecloud_error_is_constructible(depcheck):
    """RuntimeError-wrapping in the provider re-raises after catching these;
    the exception types must be constructible with a message."""
    exc_mod = depcheck.try_load("google.cloud.exceptions")
    if exc_mod is None:
        pytest.skip("google.cloud.exceptions not importable in this env")
    err = exc_mod.NotFound("blob missing")
    assert isinstance(err, exc_mod.GoogleCloudError)
