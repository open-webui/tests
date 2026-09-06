"""Boot-time and deployment-shape regressions fixed between v0.11.0 and v0.11.1.

Six independent startup failures, grouped because they all decide whether the
process comes up at all:

* 14 (PR28242, c5ec01b1f, env.py, issues #28013/#28215) — aiohttp defaulted to the
  c-ares async resolver, so name lookups failed intermittently and surfaced as a
  misleading model-not-found. env.py now pins aiohttp's `DefaultResolver` to
  `ThreadedResolver` at import time unless `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` is set.
* 51 (PR27838, 3dbb4078b, retrieval/vector/dbs/opengauss.py) — `log.setLevel(SRC_LOG_LEVELS['RAG'])`
  ran at import against a `SRC_LOG_LEVELS` that is now an empty legacy dict, so any
  openGauss deployment hit a KeyError the moment it touched the vector store.
* 52 (same commit, retrieval/models/colbert.py) — the model name was passed as a
  logging format arg to a message with no placeholder, so logging raised internally
  and printed its own traceback instead of the name.
* 55 (PR28061, 2207876ae, start_windows.bat, issue #28060) — key generation read from
  a file named by `%RANDOM%`, printing a run of file-not-found errors and leaving no
  key file; `%KEY_FILE%` was also unquoted, breaking install paths with spaces.
* 120 (0480ca9653 + 4d5084025 / PR28866, Dockerfile, issue #27651) — the bundled
  faster-whisper cache was root-only, so runAsNonRoot deployments would not start.
  The nltk download_dir half of that fix is gone with nltk itself (ca9ec06c7).
* 167 (PR27754, baeb2dfb8, internal/db.py + retrieval/vector/dbs/pgvector.py, issue
  #27752) — the pgvector engine never got the RDS IAM `do_connect` listener, so
  startup died with `fe_sendauth: no password supplied`; the listener now also
  refuses to attach to an engine pointing at a different host/port/user than the
  token was issued for.

Discriminates: passes on v0.11.1, fails on v0.11.0 (aiohttp still on AsyncResolver,
openGauss import raises KeyError('RAG'), the ColBERT log record cannot be formatted,
start_windows.bat writes no key file, the Dockerfile lacks the cache chmod, and the
pgvector engine carries no IAM listener while
enable_iam_token_auth attaches to any engine handed to it).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

_IAM_DATABASE_URL = 'postgresql://owui:from-url@main.example.com:5432/openwebui'
_IAM_SQLALCHEMY_URL = 'postgresql+psycopg2://owui:from-url@main.example.com:5432/openwebui'
_OTHER_SQLALCHEMY_URL = 'postgresql+psycopg2://owui:from-url@vectors.example.com:5432/vectors'


@pytest.fixture(scope='session')
def env_module(owui_module):
    """`open_webui.env` — imported once; entry 14's fix runs at import time."""
    return owui_module('open_webui.env')


@pytest.fixture(scope='session')
def db_module(owui_module):
    """`open_webui.internal.db` (enable_iam_token_auth)."""
    return owui_module('open_webui.internal.db')


@pytest.fixture(scope='session')
def pgvector_module(owui_module):
    """`open_webui.retrieval.vector.dbs.pgvector` (PgvectorClient)."""
    return owui_module('open_webui.retrieval.vector.dbs.pgvector')


@pytest.fixture(scope='session')
def repo_root(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent


# ─────────────────────────────────────────────────────────────────────────────
# 14 — aiohttp DNS resolver
# ─────────────────────────────────────────────────────────────────────────────


def test_env_import_pins_aiohttp_to_the_threaded_resolver(env_module) -> None:
    """Narrow: importing open_webui.env must replace the c-ares default resolver."""
    import aiohttp
    import aiohttp.connector
    import aiohttp.resolver

    threaded = aiohttp.resolver.ThreadedResolver
    assert aiohttp.DefaultResolver is threaded
    assert aiohttp.resolver.DefaultResolver is threaded
    assert aiohttp.connector.DefaultResolver is threaded


@pytest.mark.asyncio
async def test_new_tcp_connector_uses_the_threaded_resolver(env_module) -> None:
    """Narrow: a freshly built connector, the object that actually resolves names."""
    import aiohttp
    import aiohttp.resolver

    connector = aiohttp.TCPConnector()
    try:
        assert isinstance(connector._resolver, aiohttp.resolver.ThreadedResolver)
    finally:
        await connector.close()


def test_async_dns_resolver_is_opt_in(env_module) -> None:
    """Narrow: the escape hatch exists and defaults to off."""
    assert env_module.AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER is False


def test_aiohttp_still_offers_both_resolvers(env_module) -> None:
    """Nearby: the fix swaps the default, it does not remove the async resolver."""
    import aiohttp.resolver

    assert issubclass(aiohttp.resolver.AsyncResolver, aiohttp.abc.AbstractResolver)
    assert issubclass(aiohttp.resolver.ThreadedResolver, aiohttp.abc.AbstractResolver)


# ─────────────────────────────────────────────────────────────────────────────
# 51 — openGauss import-time KeyError
# ─────────────────────────────────────────────────────────────────────────────


def test_opengauss_module_imports(owui_module) -> None:
    """Narrow: the module raised KeyError('RAG') at import."""
    opengauss = owui_module('open_webui.retrieval.vector.dbs.opengauss')
    assert opengauss.log.name.endswith('opengauss')


def test_no_backend_module_subscripts_src_log_levels(open_webui_backend: Path) -> None:
    """Broad: SRC_LOG_LEVELS is an empty legacy dict, so any subscript is a KeyError."""
    offenders = [
        str(path.relative_to(open_webui_backend))
        for path in (open_webui_backend / 'open_webui').rglob('*.py')
        if 'SRC_LOG_LEVELS[' in path.read_text(encoding='utf-8', errors='replace')
    ]
    assert offenders == []


def test_src_log_levels_is_empty(env_module) -> None:
    """Nearby: pins why the subscript is fatal rather than merely redundant."""
    assert env_module.SRC_LOG_LEVELS == {}


# ─────────────────────────────────────────────────────────────────────────────
# 52 — ColBERT startup log record
# ─────────────────────────────────────────────────────────────────────────────


class _FakeCheckpoint:
    """Stands in for the model loader; the only I/O in ColBERT.__init__."""

    def __init__(self, name, colbert_config=None) -> None:
        self.name = name

    def to(self, device):
        return self


def test_colbert_startup_log_carries_the_model_name(owui_module, monkeypatch, caplog) -> None:
    """Narrow: the name was a format arg to a message with no placeholder."""
    colbert = owui_module('open_webui.retrieval.models.colbert')
    monkeypatch.setattr(colbert, 'Checkpoint', _FakeCheckpoint)

    name = 'colbert-ir/colbertv2.0'
    with caplog.at_level(logging.INFO, logger=colbert.log.name):
        colbert.ColBERT(name)

    loading = [record for record in caplog.records if 'ColBERT' in str(record.msg)]
    assert loading, 'ColBERT.__init__ logged no startup record'
    assert name in loading[0].getMessage()


def test_colbert_similarity_scores_still_work(owui_module, monkeypatch) -> None:
    """Nearby: the reranker maths around the logging fix."""
    import numpy as np
    import torch

    colbert = owui_module('open_webui.retrieval.models.colbert')
    monkeypatch.setattr(colbert, 'Checkpoint', _FakeCheckpoint)

    reranker = colbert.ColBERT('colbert-ir/colbertv2.0')
    scores = reranker.calculate_similarity_scores(torch.ones(1, 4, 8), torch.ones(3, 5, 8))
    assert scores.shape == (3,)
    assert scores.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# 55 — start_windows.bat secret key generation
# ─────────────────────────────────────────────────────────────────────────────

_KEY_ALPHABET = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
_MISSING_FILE_ERROR = 'The system cannot find the file'
_STRIPPED_KEY_ENV = (
    'WEBUI_SECRET_KEY',
    'WEBUI_JWT_SECRET_KEY',
    'WEB_LOADER_ENGINE',
    'WEBUI_SECRET_KEY_FILE',
)


def _run_start_windows_bat(open_webui_backend: Path, workdir: Path, key_file: Path | None):
    """Run the script up to the uvicorn launch, in its own scratch directory."""
    if os.name != 'nt':
        pytest.skip('start_windows.bat needs cmd.exe')
    cmd = shutil.which('cmd.exe')
    if cmd is None:
        pytest.skip('cmd.exe not found')

    source_path = open_webui_backend / 'start_windows.bat'
    if not source_path.is_file():
        pytest.skip(f'start_windows.bat not found at {source_path}')

    source = source_path.read_text(encoding='utf-8')
    launch_marker = ':: Execute uvicorn'
    if launch_marker not in source:
        pytest.skip('uvicorn launch marker gone from start_windows.bat; update this test')

    script = workdir / 'start_windows.bat'
    script.write_text(source[: source.index(launch_marker)], encoding='utf-8', newline='')

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _STRIPPED_KEY_ENV
    }
    if key_file is not None:
        env['WEBUI_SECRET_KEY_FILE'] = str(key_file)

    return subprocess.run(
        [cmd, '/c', str(script)],
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

    key_file = tmp_path / '.webui_secret_key'
    assert _MISSING_FILE_ERROR not in result.stderr, result.stderr.strip()[:400]
    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.is_file(), 'no secret key file was written'

    key = key_file.read_text(encoding='utf-8').strip()
    assert len(key) == 24
    assert set(key) <= _KEY_ALPHABET


def test_start_windows_bat_handles_a_key_path_with_spaces(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Narrow: %KEY_FILE% was unquoted, so any install path with a space broke."""
    key_dir = tmp_path / 'Open WebUI'
    key_dir.mkdir()
    key_file = key_dir / 'secret key'

    result = _run_start_windows_bat(open_webui_backend, tmp_path, key_file)

    assert _MISSING_FILE_ERROR not in result.stderr, result.stderr.strip()[:400]
    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.is_file(), 'no secret key file was written to the spaced path'
    assert set(key_file.read_text(encoding='utf-8').strip()) <= _KEY_ALPHABET


def test_start_windows_bat_reuses_an_existing_key(open_webui_backend: Path, tmp_path: Path) -> None:
    """Nearby: an existing key file is loaded, never regenerated."""
    key_file = tmp_path / '.webui_secret_key'
    key_file.write_text('preexisting-key', encoding='utf-8')

    result = _run_start_windows_bat(open_webui_backend, tmp_path, None)

    assert result.returncode == 0, result.stderr.strip()[:400]
    assert key_file.read_text(encoding='utf-8') == 'preexisting-key'
    assert 'Generating WEBUI_SECRET_KEY' not in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# 120 — non-root container startup
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope='session')
def dockerfile_text(repo_root: Path) -> str:
    dockerfile = repo_root / 'Dockerfile'
    if not dockerfile.is_file():
        pytest.skip(f'Dockerfile not found at {dockerfile}')
    return dockerfile.read_text(encoding='utf-8')


def test_dockerfile_makes_the_model_cache_world_readable(dockerfile_text: str) -> None:
    """Narrow: the bundled faster-whisper cache was root-only, blocking runAsNonRoot."""
    assert re.search(
        r'if \[ -d /app/backend/data/cache \]; then\s+chmod -R a\+rX /app/backend/data/cache; fi',
        dockerfile_text,
    ), 'Dockerfile does not relax permissions on /app/backend/data/cache'


def test_dockerfile_still_chowns_the_data_dir(dockerfile_text: str) -> None:
    """Nearby: the pre-existing ownership fix-up is untouched."""
    assert 'chown -R $UID:$GID /app/backend/data/' in dockerfile_text


# ─────────────────────────────────────────────────────────────────────────────
# 167 — RDS IAM token auth and the pgvector engine
# ─────────────────────────────────────────────────────────────────────────────


class _InitAborted(Exception):
    """Raised by the stub session to stop PgvectorClient before it hits the network."""


class _StubSession:
    def execute(self, *args, **kwargs):
        raise _InitAborted

    def rollback(self) -> None:
        pass


@pytest.fixture
def iam_token_auth(db_module, monkeypatch):
    """A real RDSIAMTokenAuth for main.example.com; its constructor does no I/O."""
    auth = db_module.RDSIAMTokenAuth(_IAM_DATABASE_URL)
    monkeypatch.setattr(db_module, '_rds_iam_token_auth', auth)
    return auth


def _has_iam_listener(db_module, engine) -> bool:
    from sqlalchemy import event

    return event.contains(engine, 'do_connect', db_module._set_iam_token_password)


def test_pgvector_engine_gets_the_iam_token_listener(
    db_module, pgvector_module, iam_token_auth, monkeypatch
) -> None:
    """Narrow: PgvectorClient built its own engine and never instrumented it."""
    captured = {}

    def fake_sessionmaker(*args, **kwargs):
        captured['engine'] = kwargs['bind']
        return lambda: _StubSession()

    monkeypatch.setattr(pgvector_module, 'PGVECTOR_DB_URL', _IAM_SQLALCHEMY_URL)
    monkeypatch.setattr(pgvector_module, 'sessionmaker', fake_sessionmaker)
    monkeypatch.setattr(pgvector_module, 'scoped_session', lambda factory: factory())

    with pytest.raises(_InitAborted):
        pgvector_module.PgvectorClient()

    assert 'engine' in captured, 'PgvectorClient did not build its own engine'
    assert _has_iam_listener(db_module, captured['engine']), (
        'the pgvector engine carries no RDS IAM do_connect listener, so its connections '
        'go out with no password'
    )


def test_iam_listener_skips_an_engine_for_another_database(
    db_module, iam_token_auth, caplog
) -> None:
    """Narrow: the token authenticates one host/port/user, not every engine."""
    from sqlalchemy import create_engine

    engine = create_engine(_OTHER_SQLALCHEMY_URL)
    with caplog.at_level(logging.WARNING, logger=db_module.log.name):
        db_module.enable_iam_token_auth(engine)

    assert not _has_iam_listener(db_module, engine), (
        'the RDS IAM token for the main database was attached to an engine pointing at '
        'a different host, overwriting that connection its own password'
    )
    assert any('IAM token auth not applied' in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    'url',
    [
        'postgresql+psycopg2://owui:from-url@main.example.com:6432/openwebui',
        'postgresql+psycopg2://other:from-url@main.example.com:5432/openwebui',
        'postgresql+psycopg2://owui:from-url@replica.example.com:5432/openwebui',
    ],
    ids=['other-port', 'other-user', 'other-host'],
)
def test_iam_listener_skips_every_mismatching_identity(db_module, iam_token_auth, url: str) -> None:
    """Broad: host, port and user each have to match, not just the host."""
    from sqlalchemy import create_engine

    engine = create_engine(url)
    db_module.enable_iam_token_auth(engine)
    assert not _has_iam_listener(db_module, engine)


def test_iam_listener_attaches_to_the_matching_engine(db_module, iam_token_auth) -> None:
    """Nearby: the positive path, including the implicit default port."""
    from sqlalchemy import create_engine

    for url in (_IAM_SQLALCHEMY_URL, 'postgresql+psycopg2://owui:other-pw@main.example.com/openwebui'):
        engine = create_engine(url)
        db_module.enable_iam_token_auth(engine)
        assert _has_iam_listener(db_module, engine), url


def test_iam_listener_is_not_attached_twice(db_module, iam_token_auth) -> None:
    """Nearby: enable_iam_token_auth stays idempotent."""
    from sqlalchemy import create_engine, event

    engine = create_engine(_IAM_SQLALCHEMY_URL)
    db_module.enable_iam_token_auth(engine)
    db_module.enable_iam_token_auth(engine)

    listeners = [
        listener
        for listener in event.registry._key_to_collection
        if listener[1] == 'do_connect' and listener[0] == id(engine)
    ]
    assert len(listeners) == 1


def test_iam_listener_is_a_no_op_when_the_feature_is_off(db_module, monkeypatch) -> None:
    """Nearby: with DATABASE_ENABLE_IAM_TOKEN_AUTH unset nothing is instrumented."""
    from sqlalchemy import create_engine

    monkeypatch.setattr(db_module, '_rds_iam_token_auth', None)
    engine = create_engine(_IAM_SQLALCHEMY_URL)
    db_module.enable_iam_token_auth(engine)
    assert not _has_iam_listener(db_module, engine)
