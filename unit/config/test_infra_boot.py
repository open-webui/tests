"""Boot-time and deployment-shape regressions fixed between v0.11.0 and v0.11.1.

Six independent startup failures, grouped because they all decide whether the process comes up:

* 14 (PR28242, c5ec01b1f, env.py, issues #28013/#28215): aiohttp defaulted to the c-ares async
  resolver, so name lookups failed intermittently and surfaced as a misleading model-not-found.
  env.py pins aiohttp's default resolver to `ThreadedResolver` at import unless
  `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` is set.
* 51 (PR27838, 3dbb4078b, retrieval/vector/dbs/opengauss.py): `SRC_LOG_LEVELS['RAG']` was read
  at import from what is now an empty legacy dict, so any openGauss deployment hit a KeyError.
* 52 (same commit, retrieval/models/colbert.py): the model name was passed as a logging argument
  to a message with no placeholder, so the startup record could not be formatted.
* 55 (PR28061, 2207876ae, start_windows.bat, issue #28060): key generation read from a file named
  by `%RANDOM%` and `%KEY_FILE%` was unquoted. These run only where cmd.exe exists.
* 120 (0480ca9653 + 4d5084025 / PR28866, Dockerfile, issue #27651): the model caches baked into
  the image were root-only, so runAsNonRoot deployments could not start.
* 167 (PR27754, baeb2dfb8, internal/db.py + retrieval/vector/dbs/pgvector.py, issue #27752): the
  pgvector engine never got the RDS IAM token, so startup died with "no password supplied"; the
  token must also stay off engines for another host, port or user.

The IAM tests let SQLAlchemy connect for real up to the driver: `psycopg2.connect` is the stand-in
that records the password it was handed. The Dockerfile is parsed the way docker reads it.

Discriminates: passes on bbfa876af; unpinning the resolver, indexing `SRC_LOG_LEVELS` in
opengauss.py, dropping the ColBERT record's placeholder, dropping the cache chmod or the data
chown from the Dockerfile, dropping `enable_iam_token_auth` from `PgvectorClient` and dropping
its identity check each fail their tests. The start_windows.bat tests need cmd.exe and did not run
on the Linux runner this was proven on.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import logging
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from unittest.mock import create_autospec

import pytest

from unit.config.container_files import read_dockerfile, run_commands

pytestmark = pytest.mark.regression

IAM_DATABASE_URL = "postgresql://owui:from-url@main.example.com:5432/openwebui"
IAM_ENGINE_URL = "postgresql+psycopg2://owui:from-url@main.example.com:5432/openwebui"
IAM_TOKEN = "iam-token-for-owui"


@pytest.fixture(scope="session")
def env_module(owui_module):
    """`open_webui.env`, imported once; entry 14's fix runs at import time."""
    return owui_module("open_webui.env")


@pytest.fixture(scope="session")
def db_module(owui_module):
    return owui_module("open_webui.internal.db")


@pytest.fixture(scope="session")
def repo_root(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent


# ─────────────────────────────────────────────────────────────────────────────
# 14 — aiohttp DNS resolver
# ─────────────────────────────────────────────────────────────────────────────


def test_a_new_connector_resolves_names_with_the_threaded_resolver(env_module) -> None:
    import aiohttp

    async def resolver_of_a_new_connector():
        connector = aiohttp.TCPConnector()
        try:
            return type(connector._resolver)
        finally:
            await connector.close()

    assert asyncio.run(resolver_of_a_new_connector()) is aiohttp.resolver.ThreadedResolver, (
        "importing open_webui.env no longer pins aiohttp to the threaded resolver (#28013); "
        "retarget at the module that now pins it"
    )


def test_the_async_resolver_is_opt_in(env_module) -> None:
    assert env_module.AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER is False


# ─────────────────────────────────────────────────────────────────────────────
# 51 — openGauss import-time KeyError
# ─────────────────────────────────────────────────────────────────────────────


def test_the_opengauss_client_imports(open_webui_backend) -> None:
    opengauss = importlib.import_module("open_webui.retrieval.vector.dbs.opengauss")
    assert opengauss.log.name.endswith("opengauss")


def test_no_backend_module_indexes_src_log_levels(open_webui_backend: Path, env_module) -> None:
    """SRC_LOG_LEVELS is an empty legacy dict, so any subscript of it is a KeyError."""
    assert env_module.SRC_LOG_LEVELS == {}

    offenders = []
    for path in (open_webui_backend / "open_webui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            indexed = node.value if isinstance(node, ast.Subscript) else None
            name = getattr(indexed, "id", None) or getattr(indexed, "attr", None)
            if name == "SRC_LOG_LEVELS":
                offenders.append(f"{path.relative_to(open_webui_backend)}:{node.lineno}")
    assert offenders == []


# ─────────────────────────────────────────────────────────────────────────────
# 52 — ColBERT startup log record
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def colbert_module(owui_module, monkeypatch):
    colbert = owui_module("open_webui.retrieval.models.colbert")
    # the checkpoint loader is the model download this stands in for
    monkeypatch.setattr(colbert, "Checkpoint", create_autospec(colbert.Checkpoint))
    return colbert


def test_the_colbert_startup_record_names_the_model(colbert_module, caplog) -> None:
    name = "colbert-ir/colbertv2.0"
    with caplog.at_level(logging.INFO, logger=colbert_module.log.name):
        colbert_module.ColBERT(name)

    messages = [record.getMessage() for record in caplog.records if "ColBERT" in str(record.msg)]
    assert messages, "ColBERT.__init__ logged no startup record"
    assert name in messages[0]


def test_colbert_similarity_scores_still_work(colbert_module) -> None:
    import numpy as np
    import torch

    reranker = colbert_module.ColBERT("colbert-ir/colbertv2.0")
    scores = reranker.calculate_similarity_scores(torch.ones(1, 4, 8), torch.ones(3, 5, 8))
    assert scores.shape == (3,)
    assert scores.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# 55 — start_windows.bat secret key generation
# ─────────────────────────────────────────────────────────────────────────────

_KEY_ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_MISSING_FILE_ERROR = "The system cannot find the file"
_STRIPPED_KEY_ENV = (
    "WEBUI_SECRET_KEY",
    "WEBUI_JWT_SECRET_KEY",
    "WEB_LOADER_ENGINE",
    "WEBUI_SECRET_KEY_FILE",
)


def _run_start_windows_bat(open_webui_backend: Path, workdir: Path, key_file: Path | None):
    """Run the script up to the uvicorn launch, in its own scratch directory."""
    if os.name != "nt":
        pytest.skip("start_windows.bat needs cmd.exe")
    cmd = shutil.which("cmd.exe")
    if cmd is None:
        pytest.skip("cmd.exe not found")

    source_path = open_webui_backend / "start_windows.bat"
    if not source_path.is_file():
        pytest.skip(f"start_windows.bat not found at {source_path}")

    source = source_path.read_text(encoding="utf-8")
    launch_marker = ":: Execute uvicorn"
    if launch_marker not in source:
        pytest.skip("uvicorn launch marker gone from start_windows.bat; update this test")

    script = workdir / "start_windows.bat"
    script.write_text(source[: source.index(launch_marker)], encoding="utf-8", newline="")

    env = {key: value for key, value in os.environ.items() if key not in _STRIPPED_KEY_ENV}
    if key_file is not None:
        env["WEBUI_SECRET_KEY_FILE"] = str(key_file)

    return subprocess.run(
        [cmd, "/c", str(script)],
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def test_start_windows_bat_generates_a_usable_secret_key(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Narrow: pre-fix this read from a file named by %RANDOM% and wrote nothing."""
    result = _run_start_windows_bat(open_webui_backend, tmp_path, None)

    key_file = tmp_path / ".webui_secret_key"
    assert _MISSING_FILE_ERROR not in result.stderr, result.stderr.strip()[:400]
    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.is_file(), "no secret key file was written"

    key = key_file.read_text(encoding="utf-8").strip()
    assert len(key) == 24
    assert set(key) <= _KEY_ALPHABET


def test_start_windows_bat_handles_a_key_path_with_spaces(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Narrow: %KEY_FILE% was unquoted, so any install path with a space broke."""
    key_dir = tmp_path / "Open WebUI"
    key_dir.mkdir()
    key_file = key_dir / "secret key"

    result = _run_start_windows_bat(open_webui_backend, tmp_path, key_file)

    assert _MISSING_FILE_ERROR not in result.stderr, result.stderr.strip()[:400]
    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.is_file(), "no secret key file was written to the spaced path"
    assert set(key_file.read_text(encoding="utf-8").strip()) <= _KEY_ALPHABET


def test_start_windows_bat_reuses_an_existing_key(open_webui_backend: Path, tmp_path: Path) -> None:
    """Nearby: an existing key file is loaded, never regenerated."""
    key_file = tmp_path / ".webui_secret_key"
    key_file.write_text("preexisting-key", encoding="utf-8")

    result = _run_start_windows_bat(open_webui_backend, tmp_path, None)

    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.read_text(encoding="utf-8") == "preexisting-key"
    assert "Generating WEBUI_SECRET_KEY" not in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# 120 — non-root container startup
# ─────────────────────────────────────────────────────────────────────────────


def _grants_others_read(mode: str) -> bool:
    if mode.isdigit():
        return int(mode[-1]) & 4 == 4
    return any(
        clause[0] in "ao" and "+" in clause and "r" in clause.split("+", 1)[1]
        for clause in mode.split(",")
    )


def _covers(target: str, path: str) -> bool:
    target_path, covered_path = PurePosixPath(target.rstrip("/")), PurePosixPath(path)
    return covered_path == target_path or target_path in covered_path.parents


def test_every_baked_model_cache_is_readable_by_any_user(repo_root: Path) -> None:
    """Narrow: the caches were root-only, so a runAsNonRoot pod could not load its models."""
    runtime = read_dockerfile(repo_root).runtime
    caches = sorted(value for value in runtime.env().values() if "/cache/" in value)
    assert caches, "the runtime stage sets no model cache directory; retarget this guard"

    readable_trees = [
        target
        for command in run_commands(runtime)
        if command[0] == "chmod"
        and "-R" in command
        and _grants_others_read(next(word for word in command[1:] if not word.startswith("-")))
        for target in command[2:]
        if target.startswith("/")
    ]
    unreadable = [
        cache for cache in caches if not any(_covers(tree, cache) for tree in readable_trees)
    ]
    assert not unreadable, (
        f"model caches only root can read, so runAsNonRoot deployments cannot start (#27651): "
        f"{unreadable}"
    )


def test_the_data_directory_is_still_handed_to_the_runtime_user(repo_root: Path) -> None:
    runtime = read_dockerfile(repo_root).runtime
    chowned = [
        target.rstrip("/")
        for command in run_commands(runtime)
        if command[0] == "chown" and "-R" in command
        for target in command[3:]
    ]
    assert "/app/backend/data" in chowned, chowned


# ─────────────────────────────────────────────────────────────────────────────
# 167 — RDS IAM token auth and the pgvector engine
# ─────────────────────────────────────────────────────────────────────────────


class ConnectRefused(Exception):
    """Raised by the stand-in driver once it has recorded what it was handed."""


@pytest.fixture
def sent_passwords(monkeypatch) -> list[str]:
    """Every password SQLAlchemy hands the psycopg2 driver; each connect then fails."""
    import psycopg2

    passwords: list[str] = []

    def connect(*args, **params):
        passwords.append(params.get("password"))
        raise ConnectRefused

    monkeypatch.setattr(psycopg2, "connect", connect)
    return passwords


@pytest.fixture(scope="module")
def rds_client():
    """A real RDS client to spec the AWS stand-in from; building one takes seconds."""
    import boto3

    return boto3.client("rds", region_name="us-east-1")


@pytest.fixture
def iam_token_auth(db_module, rds_client, monkeypatch):
    """IAM token auth configured for main.example.com, the AWS API answering with `IAM_TOKEN`."""
    import boto3

    rds = create_autospec(rds_client, instance=True)
    rds.generate_db_auth_token.return_value = IAM_TOKEN
    monkeypatch.setattr(boto3, "client", lambda service, **options: rds)
    auth = db_module.RDSIAMTokenAuth(IAM_DATABASE_URL)
    monkeypatch.setattr(db_module, "_rds_iam_token_auth", auth)


def _password_sent_for(db_module, url: str, sent_passwords: list[str]) -> str:
    from sqlalchemy import create_engine

    engine = create_engine(url)
    db_module.enable_iam_token_auth(engine)
    with pytest.raises(Exception):
        engine.connect()
    return sent_passwords[-1]


def test_the_pgvector_engine_connects_with_the_iam_token(
    owui_module, iam_token_auth, sent_passwords, monkeypatch
) -> None:
    """Narrow: PgvectorClient built its own engine and never attached the token."""
    pgvector = owui_module("open_webui.retrieval.vector.dbs.pgvector")
    monkeypatch.setattr(pgvector, "PGVECTOR_DB_URL", IAM_ENGINE_URL)

    with pytest.raises(Exception):
        pgvector.PgvectorClient()

    assert sent_passwords, "PgvectorClient never connected to its own database"
    assert sent_passwords[-1] == IAM_TOKEN, (
        "the pgvector connection went out with the URL's password instead of the IAM token, "
        "so startup died with no password supplied (#27752)"
    )


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg2://owui:from-url@vectors.example.com:5432/vectors",
        "postgresql+psycopg2://owui:from-url@main.example.com:6432/openwebui",
        "postgresql+psycopg2://other:from-url@main.example.com:5432/openwebui",
    ],
    ids=["other-host", "other-port", "other-user"],
)
def test_the_token_stays_off_an_engine_for_another_database(
    db_module, iam_token_auth, sent_passwords, caplog, url
) -> None:
    """Narrow and broad: host, port and user each have to match the token's."""
    with caplog.at_level(logging.WARNING, logger=db_module.log.name):
        sent = _password_sent_for(db_module, url, sent_passwords)

    assert sent == "from-url", "the main database's IAM token replaced another engine's password"
    assert any("IAM token auth not applied" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "url",
    [IAM_ENGINE_URL, "postgresql+psycopg2://owui:other-pw@main.example.com/openwebui"],
    ids=["explicit-port", "default-port"],
)
def test_the_token_reaches_the_matching_engine(db_module, iam_token_auth, sent_passwords, url):
    assert _password_sent_for(db_module, url, sent_passwords) == IAM_TOKEN


def test_nothing_changes_with_iam_auth_off(db_module, sent_passwords, monkeypatch) -> None:
    monkeypatch.setattr(db_module, "_rds_iam_token_auth", None)

    assert _password_sent_for(db_module, IAM_ENGINE_URL, sent_passwords) == "from-url"
