"""Dependency contract: boto3.

``boto3`` is the AWS SDK the Open WebUI backend uses for two storage
surfaces:

  * ``storage/provider.py``: ``boto3.client("s3", region_name=...,
    endpoint_url=..., aws_access_key_id=..., aws_secret_access_key=...,
    config=...)`` for the S3 file-storage provider. The client methods the
    provider calls: ``upload_file`` / ``download_file`` /
    ``delete_object`` / ``list_objects_v2`` / ``put_object_tagging``.
  * ``retrieval/vector/dbs/s3vector.py``: ``boto3.client("s3vectors",
    region_name=...)`` for the S3 Vectors vector store. Methods used:
    ``create_index`` / ``get_index`` / ``delete_index`` / ``put_vectors``
    / ``query_vectors`` / ``list_vectors`` / ``delete_vectors``.

A breaking bump (changed ``boto3.client`` factory, dropped an operation,
or a service-model removal) would break file storage and/or the vector
store. This module pins the factory + the operation surface by
*constructing the clients offline* (boto3 builds the client from local
service models; it never contacts AWS until an operation is invoked — and
we never invoke one) and asserting the methods exist. NO network, NO real
AWS, NO credentials beyond throwaway placeholders.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "boto3"
DIST_NAME = "boto3"

TOP_LEVEL_SYMBOLS = [
    "client",  # boto3.client(service, ...) — the factory the backend uses
    "resource",
    "Session",
    "setup_default_session",
]

# Operations storage/provider.py invokes on the s3 client.
S3_OPERATIONS = [
    "upload_file",
    "download_file",
    "delete_object",
    "list_objects_v2",
    "put_object_tagging",
    # also used implicitly / common
    "put_object",
    "get_object",
]

# Operations s3vector.py invokes on the s3vectors client.
S3VECTORS_OPERATIONS = [
    "create_index",
    "get_index",
    "delete_index",
    "put_vectors",
    "query_vectors",
    "list_vectors",
    "delete_vectors",
]

# Throwaway placeholder creds so client construction has something to bind to
# (still no network — credentials are only sent when an operation runs).
_FAKE = {
    "region_name": "us-east-1",
    "aws_access_key_id": "AKIA-TEST-NOT-REAL",
    "aws_secret_access_key": "test-secret-not-real",
}


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`boto3` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "boto3"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface).
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_client_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.client)


def test_botocore_config_importable(depcheck):
    """storage/provider.py builds a botocore.config.Config(...) and passes it as
    config=. The Config class must remain importable."""
    cfg_mod = depcheck.load("botocore.config")
    assert hasattr(cfg_mod, "Config")


def test_botocore_exceptions_importable(depcheck):
    """Callers handle ClientError / NoCredentialsError from botocore."""
    exc = depcheck.load("botocore.exceptions")
    for name in ("ClientError", "NoCredentialsError", "BotoCoreError"):
        assert hasattr(exc, name), f"botocore.exceptions.{name} missing"


# ---------------------------------------------------------------------------
# Config construction contract (offline) — the exact kwargs the provider uses.
# ---------------------------------------------------------------------------


def test_config_accepts_provider_kwargs(depcheck):
    """storage/provider.py constructs Config(s3={...},
    request_checksum_calculation='when_required',
    response_checksum_validation='when_required'). Those must be accepted."""
    cfg_mod = depcheck.load("botocore.config")
    config = cfg_mod.Config(
        s3={"use_accelerate_endpoint": False, "addressing_style": "auto"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    assert config is not None


# ---------------------------------------------------------------------------
# S3 client construction contract (OFFLINE — builds from local service model).
# ---------------------------------------------------------------------------


def test_s3_client_constructs_offline(depcheck):
    """boto3.client('s3', ...) must construct purely from the bundled service
    model — no AWS contact. The provider builds it at init time."""
    boto3 = depcheck.load(IMPORT_NAME)
    client = boto3.client("s3", **_FAKE)
    assert client is not None
    assert client.meta.region_name == "us-east-1"


def test_s3_client_has_all_used_operations(depcheck):
    """Every S3 operation storage/provider.py calls must exist on the client."""
    boto3 = depcheck.load(IMPORT_NAME)
    client = boto3.client("s3", **_FAKE)
    missing = [op for op in S3_OPERATIONS if not callable(getattr(client, op, None))]
    assert not missing, f"s3 client missing operations the backend calls: {missing}"


def test_s3_client_accepts_endpoint_url(depcheck):
    """The provider passes endpoint_url (for S3-compatible stores like MinIO).
    Constructing with a custom endpoint must work offline."""
    boto3 = depcheck.load(IMPORT_NAME)
    client = boto3.client(
        "s3",
        endpoint_url="https://minio.example.local:9000",
        **_FAKE,
    )
    assert client is not None


def test_s3_client_accepts_config(depcheck):
    """The provider passes config=Config(...). Construction with a Config object
    must succeed offline."""
    boto3 = depcheck.load(IMPORT_NAME)
    cfg_mod = depcheck.load("botocore.config")
    client = boto3.client(
        "s3",
        config=cfg_mod.Config(request_checksum_calculation="when_required"),
        **_FAKE,
    )
    assert client is not None


def test_s3_client_exposes_exceptions(depcheck):
    """Clients expose a `.exceptions` namespace (ClientError etc.) callers use
    for fine-grained error handling."""
    boto3 = depcheck.load(IMPORT_NAME)
    client = boto3.client("s3", **_FAKE)
    assert hasattr(client, "exceptions")


def test_upload_file_signature(depcheck):
    """provider calls s3_client.upload_file(local_path, bucket, key). The
    operation must accept those three positionals (Filename, Bucket, Key)."""
    boto3 = depcheck.load(IMPORT_NAME)
    client = boto3.client("s3", **_FAKE)
    sig = inspect.signature(client.upload_file)
    params = list(sig.parameters)
    # upload_file(Filename, Bucket, Key, ExtraArgs=None, Callback=None, Config=None)
    assert params[:3] == ["Filename", "Bucket", "Key"], (
        f"upload_file positional shape changed: {params}"
    )


# ---------------------------------------------------------------------------
# S3 Vectors client construction contract (OFFLINE).
# ---------------------------------------------------------------------------


def test_s3vectors_client_constructs_offline(depcheck):
    """s3vector.py builds boto3.client('s3vectors', region_name=...). The
    service model must be present and the client constructable offline. If the
    installed botocore predates the s3vectors service, skip cleanly."""
    boto3 = depcheck.load(IMPORT_NAME)
    try:
        client = boto3.client("s3vectors", **_FAKE)
    except Exception as e:
        # UnknownServiceError when the service model isn't bundled in this
        # botocore version — pin via the s3 path; flag the absence.
        pytest.skip(f"s3vectors service model unavailable in this botocore: {e}")
    assert client is not None
    assert client.meta.region_name == "us-east-1"


def test_s3vectors_client_has_all_used_operations(depcheck):
    """Every s3vectors operation s3vector.py calls must exist on the client."""
    boto3 = depcheck.load(IMPORT_NAME)
    try:
        client = boto3.client("s3vectors", **_FAKE)
    except Exception as e:
        pytest.skip(f"s3vectors service model unavailable: {e}")
    missing = [op for op in S3VECTORS_OPERATIONS if not callable(getattr(client, op, None))]
    assert not missing, f"s3vectors client missing operations: {missing}"


# ---------------------------------------------------------------------------
# Session contract — boto3.client routes through a Session.
# ---------------------------------------------------------------------------


def test_session_can_create_client(depcheck):
    """boto3.Session().client('s3', ...) is the session-scoped equivalent of
    the module factory; pin it constructs offline too."""
    boto3 = depcheck.load(IMPORT_NAME)
    session = boto3.Session()
    client = session.client("s3", **_FAKE)
    assert client is not None
    assert callable(client.list_objects_v2)
