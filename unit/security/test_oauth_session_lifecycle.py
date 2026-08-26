"""Regression tests for the 0.11.0 OAuth sign-in / session-lifecycle fixes.

Seven defects that all sat on the sign-in path:

- Profile-picture SSRF (#26699, commit 5dcca59ae): `OAuthManager._process_picture_url`
  fetched the provider-supplied picture with a plain `aiohttp.ClientSession(trust_env=True)`
  after `validate_url()` had vetted only the initial URL, so a host that changed address
  between check and fetch reached an internal service with the sign-in token attached.
  Fixed by switching to `get_ssrf_safe_session()`, which pins the connect-time IP.
- MCP OAuth discovery (#26654, issue #26647, commit 3fff80ad2): authlib raises
  `RuntimeError('Missing "authorize_url" value')` when the authorization endpoint could
  not be resolved; `OAuthClientManager.handle_authorize` let it escape as a 500. Fixed by
  re-raising a 400 that says to re-register the MCP server.
- Duplicate account on first trusted-header sign-in (commits b190dcf + 50e050e, #27571,
  issue #27117): `AuthsTable.insert_new_auth` committed with no integrity handling, so two
  concurrent first-time requests created two accounts. Fixed with try/except IntegrityError
  + rollback, plus alembic revision f0bd01a18a3d adding a unique index over lower(email).
- SSO settings from env vars (#26928, commit e398ba350): `Config.seed_defaults` inserted a
  row for every default key including the non-persistent `oauth.*` ones, and the stale row
  then shadowed the environment variable forever. Fixed by skipping keys the DB is not
  authoritative for. Same cluster: `ENABLE_OAUTH` joined `OAUTH_RUNTIME_CONFIG` and both
  `handle_login`/`handle_callback` now 404 when it is off.
- Expired identity tokens sent to tools (#27520, issue #27066, commit c055203f2) and
  never-expiring sign-in tokens (commit 98656b7, #26802, issue #26141): both in
  `_normalize_token_expiry`. It ignored the id_token's own `exp` (so an expired identity
  JWT kept being forwarded to tools and pipes) and fabricated a one-hour lifetime even for
  providers that returned no refresh_token (so the session died unrecoverably after an hour).
- Single sign-on after a key rotation (#27310, issue #26407, commit 4f823774a): the JWKS
  eviction/retry in `OAuthManager.handle_callback` caught `authlib.jose.errors.BadSignatureError`,
  but authlib 1.7 delegates JWS verification to joserfc, which raises the unrelated
  `joserfc.errors.BadSignatureError`. The except never matched, so the retry never ran.

v0.11.1 (commit c2107e5bb) then reworked the same MCP authorize leg: it takes the initiating
`user_id`, stamps it into the authlib state and requires a state to exist, and `handle_callback`
stopped taking a `user_id` argument and reads it back out of the state instead. Before that the
callback bound the token to whoever's cookie was on the returning request. Those tests are
guarded by a signature check and skip on a checkout that predates the change.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (v0.10.2 uses a plain aiohttp
session for the picture, lets the authorize RuntimeError escape, never rolls back and has no
lower(email) index, seeds oauth.* config rows, has no ENABLE_OAUTH gate, ignores the id_token exp
and invents a 3600s expiry without a refresh path, and catches the wrong BadSignatureError class).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import jwt as pyjwt
import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from joserfc.errors import BadSignatureError as JoseRFCBadSignatureError
from sqlalchemy.exc import IntegrityError
from starlette.responses import RedirectResponse

pytestmark = pytest.mark.regression

PNG_BYTES = b'\x89PNG\r\n\x1a\n' + b'0' * 32
CALL_BUDGET = 5
MCP_AUTHORIZE_URL = 'https://mcp.example/authorize'
MCP_STATE = 'state-token-1'
INITIATOR = 'alice'


class _Sentinel(BaseException):
    """Subclasses BaseException so production `except Exception` cannot swallow it."""


@pytest.fixture(scope='session')
def oauth_module(owui_module):
    return owui_module('open_webui.utils.oauth')


@pytest.fixture(scope='session')
def auths_model_module(owui_module):
    return owui_module('open_webui.models.auths')


# ── fakes (I/O boundary only) ───────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, body: bytes = PNG_BYTES, content_type: str = 'image/png'):
        self.ok = True
        self.status = 200
        self.headers = {'Content-Type': content_type}
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeHttpSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.get_calls: list[tuple[str, dict]] = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDbSession:
    """Records the calls insert_new_auth / seed_defaults make on a session."""

    def __init__(self, commit_error: Exception | None = None):
        self.commit_error = commit_error
        self.added: list = []
        self.commits = 0
        self.rollbacks = 0
        self.refreshes = 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        if self.commit_error is not None:
            raise self.commit_error

    async def rollback(self):
        self.rollbacks += 1

    async def refresh(self, obj):
        self.refreshes += 1

    async def execute(self, statement):
        return SimpleNamespace(all=lambda: [], scalars=lambda: SimpleNamespace(all=lambda: []))


def _fake_db_context(session: _FakeDbSession):
    @contextlib.asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield session

    return _ctx


def _auth_config(**overrides) -> SimpleNamespace:
    values = {
        'ENABLE_OAUTH': True,
        'OAUTH_AUDIENCE': None,
        'OAUTH_EMAIL_CLAIM': 'email',
        'OAUTH_USERNAME_CLAIM': 'name',
        'OAUTH_SUB_CLAIM': None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_runtime_config(monkeypatch, oauth_module, **overrides) -> None:
    config = _auth_config(**overrides)

    async def _get():
        return config

    monkeypatch.setattr(oauth_module, 'get_oauth_runtime_config', _get)


def _make_oauth_manager(monkeypatch, oauth_module, provider: str, client):
    """OAuthManager with one registered provider, no real IdP wiring."""
    monkeypatch.setattr(oauth_module, 'OAUTH_PROVIDERS', {}, raising=False)
    manager = oauth_module.OAuthManager(app=SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(oauth_module, 'OAUTH_PROVIDERS', {provider: {}}, raising=False)
    manager._clients[provider] = client
    return manager


def _id_token(exp: int | None) -> str:
    claims = {'sub': 'u1'}
    if exp is not None:
        claims['exp'] = exp
    return pyjwt.encode(claims, 'unit-test-key', algorithm='HS256')


# ── 1. profile-picture SSRF (#26699) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_profile_picture_fetch_uses_ssrf_safe_session(monkeypatch, oauth_module):
    """NARROW: the picture fetch must go through the IP-pinning session, not a plain one."""
    monkeypatch.setattr(oauth_module, 'validate_url', lambda url: None)

    fake_session = _FakeHttpSession(_FakeResponse())
    safe_calls = []
    plain_calls = []

    def _safe_session(*args, **kwargs):
        safe_calls.append(kwargs)
        return fake_session

    def _plain_session(*args, **kwargs):
        plain_calls.append(kwargs)
        return fake_session

    monkeypatch.setattr(oauth_module, 'get_ssrf_safe_session', _safe_session, raising=False)
    monkeypatch.setattr(oauth_module, 'aiohttp', SimpleNamespace(ClientSession=_plain_session))

    manager = oauth_module.OAuthManager(app=SimpleNamespace(state=SimpleNamespace()))
    result = await asyncio.wait_for(
        manager._process_picture_url('https://idp.example/avatar.png', 'secret-access-token'),
        timeout=15,
    )

    expected = 'data:image/png;base64,' + base64.b64encode(PNG_BYTES).decode()
    assert result == expected
    assert len(safe_calls) == 1, 'picture fetch did not use get_ssrf_safe_session'
    assert plain_calls == [], 'picture fetch used a plain aiohttp.ClientSession'


@pytest.mark.asyncio
async def test_profile_picture_fetch_still_refuses_redirects(monkeypatch, oauth_module):
    """BROAD: IP pinning does not replace the redirect ban; both must hold together."""
    monkeypatch.setattr(oauth_module, 'validate_url', lambda url: None)
    fake_session = _FakeHttpSession(_FakeResponse())
    monkeypatch.setattr(
        oauth_module, 'get_ssrf_safe_session', lambda *a, **k: fake_session, raising=False
    )
    monkeypatch.setattr(
        oauth_module, 'aiohttp', SimpleNamespace(ClientSession=lambda *a, **k: fake_session)
    )

    manager = oauth_module.OAuthManager(app=SimpleNamespace(state=SimpleNamespace()))
    await asyncio.wait_for(
        manager._process_picture_url('https://idp.example/avatar.png', 'secret-access-token'),
        timeout=15,
    )

    assert len(fake_session.get_calls) == 1
    url, kwargs = fake_session.get_calls[0]
    assert url == 'https://idp.example/avatar.png'
    assert kwargs['allow_redirects'] is False
    assert kwargs['headers'] == {'Authorization': 'Bearer secret-access-token'}


@pytest.mark.asyncio
async def test_profile_picture_empty_url_returns_default(oauth_module):
    """NEARBY: the no-picture path never fetches anything, on either ref."""
    manager = oauth_module.OAuthManager(app=SimpleNamespace(state=SimpleNamespace()))
    assert await asyncio.wait_for(manager._process_picture_url(''), timeout=15) == '/user.png'


# ── 2. MCP OAuth authorize with unresolved endpoint (#26654) ────────────────


def _mcp_client_info():
    return SimpleNamespace(
        redirect_uris=['https://owui.example/oauth/callback'],
        scope=None,
        resource=None,
        oauth_resource_parameter=None,
    )


def _make_client_manager(oauth_module, client_id: str, client):
    manager = oauth_module.OAuthClientManager(app=SimpleNamespace(state=SimpleNamespace()))
    manager.clients[client_id] = {'client': client, 'client_info': _mcp_client_info()}
    return manager


class _FakeMcpOAuthClient:
    """Mirrors authlib's Starlette app, whose authorize_redirect is exactly
    create_authorization_url + save_authorize_data, so the 0.11.0 and 0.11.1 call shapes both
    land on the same recorded boundary."""

    def __init__(self, state: str | None = MCP_STATE, create_error: Exception | None = None):
        self.state = state
        self.create_error = create_error
        self.create_calls: list[tuple] = []
        self.saved_state_data: list[tuple[str, dict]] = []

    async def create_authorization_url(self, redirect_uri=None, **kwargs):
        if self.create_error is not None:
            raise self.create_error
        self.create_calls.append((redirect_uri, kwargs))
        data = {'url': MCP_AUTHORIZE_URL}
        if self.state is not None:
            data['state'] = self.state
        return data

    async def save_authorize_data(self, request, **kwargs):
        state = kwargs.pop('state', None)
        if not state:
            raise RuntimeError('Missing state value')
        self.saved_state_data.append((state, kwargs))

    async def authorize_redirect(self, request, redirect_uri=None, **kwargs):
        data = await self.create_authorization_url(redirect_uri, **kwargs)
        await self.save_authorize_data(request, redirect_uri=redirect_uri, **data)
        return RedirectResponse(data['url'], status_code=302)


def _authorize_binds_user(oauth_module) -> bool:
    parameters = inspect.signature(oauth_module.OAuthClientManager.handle_authorize).parameters
    return 'user_id' in parameters


def _authorize_kwargs(oauth_module, user_id: str = INITIATOR) -> dict:
    """0.11.1 requires the initiating user; earlier refs have no such parameter."""
    return {'user_id': user_id} if _authorize_binds_user(oauth_module) else {}


def _require_authorize_user_binding(oauth_module) -> None:
    if not _authorize_binds_user(oauth_module):
        pytest.skip('checkout predates c2107e5bb: handle_authorize takes no user_id')


@pytest.mark.asyncio
async def test_mcp_authorize_unresolved_endpoint_is_400_with_guidance(oauth_module):
    """NARROW: authlib's Missing "authorize_url" RuntimeError must surface as a 400."""
    client = _FakeMcpOAuthClient(create_error=RuntimeError('Missing "authorize_url" value'))
    manager = _make_client_manager(oauth_module, 'mcp1', client)

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(
            manager.handle_authorize(SimpleNamespace(), 'mcp1', **_authorize_kwargs(oauth_module)),
            timeout=15,
        )

    assert excinfo.value.status_code == 400
    assert 'Re-register the MCP server' in excinfo.value.detail


@pytest.mark.asyncio
async def test_mcp_authorize_success_path_unchanged(oauth_module):
    """NEARBY: a resolvable client is still redirected to the provider's authorize URL."""
    client = _FakeMcpOAuthClient()
    manager = _make_client_manager(oauth_module, 'mcp1', client)

    result = await asyncio.wait_for(
        manager.handle_authorize(SimpleNamespace(), 'mcp1', **_authorize_kwargs(oauth_module)),
        timeout=15,
    )

    assert result.status_code == 302
    assert result.headers['location'] == MCP_AUTHORIZE_URL
    assert client.create_calls[0][0] == 'https://owui.example/oauth/callback'


@pytest.mark.asyncio
async def test_mcp_authorize_always_persists_the_state_it_redirects_with(oauth_module):
    """BROAD: the callback can only match a flow that was recorded, so state must be saved."""
    client = _FakeMcpOAuthClient()
    manager = _make_client_manager(oauth_module, 'mcp1', client)

    await asyncio.wait_for(
        manager.handle_authorize(SimpleNamespace(), 'mcp1', **_authorize_kwargs(oauth_module)),
        timeout=15,
    )

    assert [state for state, _ in client.saved_state_data] == [MCP_STATE]
    assert client.saved_state_data[0][1]['redirect_uri'] == 'https://owui.example/oauth/callback'


@pytest.mark.asyncio
async def test_mcp_authorize_binds_the_initiating_user_to_the_oauth_state(oauth_module):
    """NARROW (0.11.1): the flow is tied to its initiator; the callback trusts only that id,
    where it previously took the user off whatever session returned to the callback."""
    _require_authorize_user_binding(oauth_module)
    client = _FakeMcpOAuthClient()
    manager = _make_client_manager(oauth_module, 'mcp1', client)

    await asyncio.wait_for(
        manager.handle_authorize(SimpleNamespace(), 'mcp1', user_id=INITIATOR), timeout=15
    )

    assert client.saved_state_data[0][1]['user_id'] == INITIATOR


@pytest.mark.asyncio
async def test_mcp_authorize_refuses_to_redirect_without_a_state(oauth_module):
    """NARROW (0.11.1): no state means nothing to bind the user to, so no redirect may go out."""
    _require_authorize_user_binding(oauth_module)
    client = _FakeMcpOAuthClient(state=None)
    manager = _make_client_manager(oauth_module, 'mcp1', client)

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(
            manager.handle_authorize(SimpleNamespace(), 'mcp1', user_id=INITIATOR), timeout=15
        )

    assert excinfo.value.status_code == 500
    assert client.saved_state_data == [], 'an unbindable flow was persisted anyway'


@pytest.mark.asyncio
async def test_mcp_authorize_unknown_client_is_404(oauth_module):
    """NEARBY: the pre-existing unknown-client 404 is not swallowed by the new handler."""
    manager = oauth_module.OAuthClientManager(app=SimpleNamespace(state=SimpleNamespace()))

    async def _ensure(client_id):
        return None

    manager.ensure_client_from_config = _ensure

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(
            manager.handle_authorize(SimpleNamespace(), 'nope', **_authorize_kwargs(oauth_module)),
            timeout=15,
        )
    assert excinfo.value.status_code == 404


# ── 2b. MCP OAuth callback binds the token to the state's user (c2107e5bb) ──


class _FakeCallbackFramework:
    def __init__(self, state_data: dict | None):
        self.state_data = state_data
        self.cleared: list[str] = []

    async def get_state_data(self, session, state):
        return self.state_data

    async def clear_state_data(self, session, state):
        self.cleared.append(state)


class _FakeCallbackClient:
    def __init__(self, state_data: dict | None):
        self.framework = _FakeCallbackFramework(state_data)
        self.token_calls = 0

    async def authorize_access_token(self, request, **kwargs):
        self.token_calls += 1
        return {'access_token': 'at', 'refresh_token': 'rt', 'expires_in': 3600}


def _callback_reads_user_from_state(oauth_module) -> bool:
    parameters = inspect.signature(oauth_module.OAuthClientManager.handle_callback).parameters
    return 'user_id' not in parameters


def _require_callback_user_from_state(oauth_module) -> None:
    if not _callback_reads_user_from_state(oauth_module):
        pytest.skip('checkout predates c2107e5bb: handle_callback is handed a user_id')


def _patch_callback_boundaries(monkeypatch, oauth_module, session_user_id: str | None = None):
    """Stub the callback's I/O boundary; returns the list of sessions it created."""
    created: list[dict] = []

    async def _verified_by_id(user_id):
        return SimpleNamespace(id=user_id)

    async def _user_from_request(request):
        return SimpleNamespace(id=session_user_id) if session_user_id else None

    async def _get_sessions(user_id):
        return []

    async def _create_session(user_id, provider, token):
        created.append({'user_id': user_id, 'provider': provider})
        return SimpleNamespace(id='oauth-session-1')

    async def _config_get(key, default=None):
        return 'https://owui.example'

    monkeypatch.setattr(oauth_module, 'get_verified_user_by_id', _verified_by_id)
    monkeypatch.setattr(oauth_module, 'get_optional_verified_user_from_request', _user_from_request)
    monkeypatch.setattr(
        oauth_module,
        'OAuthSessions',
        SimpleNamespace(get_sessions_by_user_id=_get_sessions, create_session=_create_session),
    )
    monkeypatch.setattr(oauth_module, 'Config', SimpleNamespace(get=_config_get))
    return created


def _callback_request(state: str | None = MCP_STATE):
    return SimpleNamespace(
        query_params={'state': state} if state else {},
        session={},
        base_url='https://owui.example/',
    )


async def _run_callback(manager, request):
    return await asyncio.wait_for(
        manager.handle_callback(request, 'mcp1', SimpleNamespace(headers={})), timeout=15
    )


@pytest.mark.asyncio
async def test_mcp_callback_stores_the_token_for_the_state_user(monkeypatch, oauth_module):
    """NARROW (0.11.1): the token is filed against the id stamped into the state, and the
    returning request carries no session at all."""
    _require_callback_user_from_state(oauth_module)
    client = _FakeCallbackClient({'user_id': INITIATOR})
    manager = _make_client_manager(oauth_module, 'mcp1', client)
    created = _patch_callback_boundaries(monkeypatch, oauth_module)

    result = await _run_callback(manager, _callback_request())

    assert 'error=' not in result.headers['location']
    assert created == [{'user_id': INITIATOR, 'provider': 'mcp1'}]


@pytest.mark.asyncio
async def test_mcp_callback_refuses_a_foreign_session_returning_to_the_flow(
    monkeypatch, oauth_module
):
    """NARROW (0.11.1): whoever's cookie comes back must be the initiator, or nothing is stored."""
    _require_callback_user_from_state(oauth_module)
    client = _FakeCallbackClient({'user_id': INITIATOR})
    manager = _make_client_manager(oauth_module, 'mcp1', client)
    created = _patch_callback_boundaries(monkeypatch, oauth_module, session_user_id='mallory')

    result = await _run_callback(manager, _callback_request())

    assert 'error=' in result.headers['location']
    assert created == [], 'the token was bound to the session that returned to the callback'
    assert client.token_calls == 0, 'the code was exchanged for a mismatched session'


@pytest.mark.asyncio
async def test_mcp_callback_refuses_a_state_carrying_no_user(monkeypatch, oauth_module):
    """NARROW (0.11.1): an unbound state has nobody to file the token against."""
    _require_callback_user_from_state(oauth_module)
    client = _FakeCallbackClient({})
    manager = _make_client_manager(oauth_module, 'mcp1', client)
    created = _patch_callback_boundaries(monkeypatch, oauth_module, session_user_id='mallory')

    result = await _run_callback(manager, _callback_request())

    assert 'error=' in result.headers['location']
    assert created == []
    assert client.token_calls == 0, 'an unbound state still reached the token endpoint'


@pytest.mark.asyncio
async def test_mcp_callback_clears_the_state_it_consumed(monkeypatch, oauth_module):
    """BROAD: a consumed state must not stay replayable, on the success or the refusal path."""
    _require_callback_user_from_state(oauth_module)
    client = _FakeCallbackClient({'user_id': INITIATOR})
    manager = _make_client_manager(oauth_module, 'mcp1', client)
    _patch_callback_boundaries(monkeypatch, oauth_module)

    await _run_callback(manager, _callback_request())

    assert client.framework.cleared == [MCP_STATE]


# ── 3a. duplicate account: insert_new_auth integrity handling (#27571) ──────


def _stub_users(monkeypatch, auths_model_module, user):
    async def _insert_new_user(*args, **kwargs):
        return user

    monkeypatch.setattr(
        auths_model_module, 'Users', SimpleNamespace(insert_new_user=_insert_new_user)
    )


@pytest.mark.asyncio
async def test_insert_new_auth_rolls_back_on_integrity_error(monkeypatch, auths_model_module):
    """NARROW: a racing duplicate must roll the half-written transaction back, then re-raise."""
    session = _FakeDbSession(commit_error=IntegrityError('INSERT', {}, Exception('duplicate')))
    monkeypatch.setattr(auths_model_module, 'get_async_db_context', _fake_db_context(session))
    _stub_users(monkeypatch, auths_model_module, SimpleNamespace(id='u1'))

    with pytest.raises(IntegrityError):
        await asyncio.wait_for(
            auths_model_module.Auths.insert_new_auth('A@x.com', 'hash', 'A'), timeout=15
        )

    assert session.rollbacks == 1, 'insert_new_auth left the failed transaction un-rolled-back'


@pytest.mark.asyncio
async def test_insert_new_auth_happy_path_commits_once(monkeypatch, auths_model_module):
    """NEARBY: the non-racing path still adds the credential and commits, on either ref."""
    session = _FakeDbSession()
    monkeypatch.setattr(auths_model_module, 'get_async_db_context', _fake_db_context(session))
    user = SimpleNamespace(id='u1')
    _stub_users(monkeypatch, auths_model_module, user)

    result = await asyncio.wait_for(
        auths_model_module.Auths.insert_new_auth('a@x.com', 'hash', 'A'), timeout=15
    )

    assert result is user
    assert session.commits == 1
    assert session.rollbacks == 0
    assert [obj.email for obj in session.added] == ['a@x.com']


# ── 3b. duplicate account: the unique lower(email) index (f0bd01a18a3d) ─────


def _alembic_upgrade_head(backend: Path, database_url: str, data_dir: Path) -> tuple[int, str, str]:
    """Run the migration chain out of process; this process already holds a different DB."""
    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ['DATABASE_URL'] = {database_url!r}
        os.environ['DATA_DIR'] = {str(data_dir)!r}
        sys.path.insert(0, {str(backend)!r})

        from alembic import command
        from alembic.config import Config as AlembicConfig
        from open_webui.env import OPEN_WEBUI_DIR

        cfg = AlembicConfig(OPEN_WEBUI_DIR / 'alembic.ini')
        cfg.set_main_option('script_location', str(OPEN_WEBUI_DIR / 'migrations'))
        command.upgrade(cfg, 'head')
        print('MIGRATED')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, 'PYTHONUNBUFFERED': '1', 'WEBUI_SECRET_KEY': 'test'},
    )
    return result.returncode, result.stdout, result.stderr


def _insert_user(engine, user_id: str, email: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text('INSERT INTO "user" (id, name, email, role) VALUES (:id, :n, :e, :r)'),
            {'id': user_id, 'n': email, 'e': email, 'r': 'user'},
        )


@pytest.fixture
def migrated_sqlite(open_webui_backend, tmp_path):
    """Fresh migrated SQLite DB in tmp_path; never the suite's shared store."""
    db_path = (tmp_path / 'webui.db').resolve()
    db_url = f'sqlite:///{db_path.as_posix()}'
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    rc, stdout, stderr = _alembic_upgrade_head(open_webui_backend, db_url, data_dir)
    assert rc == 0, f'alembic upgrade head failed\n{stderr[-3000:]}'
    assert 'MIGRATED' in stdout

    engine = sa.create_engine(db_url)
    yield engine
    engine.dispose()


def test_unique_normalized_email_index_exists(migrated_sqlite):
    """NARROW: revision f0bd01a18a3d must leave a unique index over lower(email)."""
    with migrated_sqlite.begin() as conn:
        rows = conn.execute(
            sa.text("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='user'")
        ).fetchall()

    matching = [row for row in rows if row.name == 'uq_user_email_lower']
    assert matching, f'uq_user_email_lower missing; indexes present: {[r.name for r in rows]}'
    assert 'UNIQUE' in matching[0].sql.upper()
    assert 'lower(email)' in matching[0].sql.lower()


def test_second_user_with_differently_cased_email_is_rejected(migrated_sqlite):
    """NARROW: the same address in different capitalisation must not create a second account."""
    _insert_user(migrated_sqlite, 'u1', 'A@x.com')

    with pytest.raises(IntegrityError):
        _insert_user(migrated_sqlite, 'u2', 'a@x.com')


def test_distinct_emails_and_null_emails_still_allowed(migrated_sqlite):
    """NEARBY: the index must not block genuinely different users, on either ref."""
    _insert_user(migrated_sqlite, 'u1', 'a@x.com')
    _insert_user(migrated_sqlite, 'u2', 'b@x.com')
    with migrated_sqlite.begin() as conn:
        conn.execute(
            sa.text('INSERT INTO "user" (id, name, email, role) VALUES (:id, :n, NULL, :r)'),
            {'id': 'u3', 'n': 'no-email-1', 'r': 'user'},
        )
        conn.execute(
            sa.text('INSERT INTO "user" (id, name, email, role) VALUES (:id, :n, NULL, :r)'),
            {'id': 'u4', 'n': 'no-email-2', 'r': 'user'},
        )
        count = conn.execute(sa.text('SELECT count(*) FROM "user"')).scalar()
    assert count == 4


# ── 4. SSO settings from env vars (#26928) ─────────────────────────────────


@pytest.fixture
def seedable_config(monkeypatch, config_model_module):
    """Config class wired to a fake DB session, with oauth persistence off."""
    config_class = config_model_module.Config
    session = _FakeDbSession()
    monkeypatch.setattr(config_model_module, 'get_async_db', _fake_db_context(session))
    monkeypatch.setattr(config_class, 'PERSISTENT_ENABLED', True)
    monkeypatch.setattr(config_class, 'OAUTH_PERSISTENT_ENABLED', False)
    return config_class, session


@pytest.mark.parametrize(
    'oauth_key',
    ['oauth.enable', 'oauth.enable_signup', 'oauth.allowed_domains', 'oauth.admin_roles'],
)
@pytest.mark.asyncio
async def test_seed_defaults_skips_non_persistent_oauth_keys(seedable_config, oauth_key):
    """NARROW+BROAD: a seeded oauth.* row shadows the env var forever, so none may be seeded."""
    config_class, session = seedable_config

    await asyncio.wait_for(
        config_class.seed_defaults(
            {oauth_key: 'from-first-boot', 'ui.default_user_role': 'pending'}
        ),
        timeout=15,
    )

    seeded_keys = {row.key for row in session.added}
    assert oauth_key not in seeded_keys
    assert seeded_keys == {'ui.default_user_role'}


@pytest.mark.asyncio
async def test_seed_defaults_still_seeds_persistent_keys(seedable_config):
    """NEARBY: ordinary keys are still seeded, on either ref."""
    config_class, session = seedable_config

    await asyncio.wait_for(
        config_class.seed_defaults({'ui.default_user_role': 'pending', 'webui.url': 'https://x'}),
        timeout=15,
    )

    assert {row.key for row in session.added} == {'ui.default_user_role', 'webui.url'}
    assert session.commits == 1


@pytest.mark.asyncio
async def test_seed_defaults_seeds_oauth_keys_when_persistence_enabled(
    monkeypatch, seedable_config
):
    """NEARBY: with the persistence flag on the DB is authoritative again, on either ref."""
    config_class, session = seedable_config
    monkeypatch.setattr(config_class, 'OAUTH_PERSISTENT_ENABLED', True)

    await asyncio.wait_for(config_class.seed_defaults({'oauth.enable_signup': True}), timeout=15)

    assert {row.key for row in session.added} == {'oauth.enable_signup'}


@pytest.mark.asyncio
async def test_handle_login_404s_when_oauth_disabled(monkeypatch, oauth_module):
    """NARROW: with oauth.enable off, sign-in must not reach the provider at all."""
    redirects = []

    async def _authorize_redirect(request, redirect_uri, **kwargs):
        redirects.append(redirect_uri)
        return object()

    client = SimpleNamespace(
        authorize_redirect=_authorize_redirect,
        server_metadata={'redirect_uri': 'https://owui.example/cb'},
    )
    manager = _make_oauth_manager(monkeypatch, oauth_module, 'testidp', client)
    _patch_runtime_config(monkeypatch, oauth_module, ENABLE_OAUTH=False)

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(manager.handle_login(SimpleNamespace(), 'testidp'), timeout=15)

    assert excinfo.value.status_code == 404
    assert redirects == [], 'sign-in redirect issued while OAuth is disabled'


@pytest.mark.asyncio
async def test_handle_callback_404s_when_oauth_disabled(monkeypatch, oauth_module):
    """NARROW: the callback must be closed too, not just the login redirect."""
    token_calls = []

    async def _authorize_access_token(request, **kwargs):
        token_calls.append(kwargs)
        raise _Sentinel('callback reached the token exchange')

    client = SimpleNamespace(
        authorize_access_token=_authorize_access_token,
        server_metadata={},
    )
    manager = _make_oauth_manager(monkeypatch, oauth_module, 'testidp', client)
    _patch_runtime_config(monkeypatch, oauth_module, ENABLE_OAUTH=False)

    request = SimpleNamespace(
        base_url='https://owui.example/', app=SimpleNamespace(state=SimpleNamespace(redis=None))
    )
    response = SimpleNamespace(headers={})

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(manager.handle_callback(request, 'testidp', response), timeout=15)

    assert excinfo.value.status_code == 404
    assert token_calls == [], 'callback exchanged a token while OAuth is disabled'


@pytest.mark.asyncio
async def test_handle_login_404s_for_unknown_provider(monkeypatch, oauth_module):
    """NEARBY: the pre-existing unknown-provider 404 survives, on either ref."""
    manager = _make_oauth_manager(monkeypatch, oauth_module, 'testidp', SimpleNamespace())
    _patch_runtime_config(monkeypatch, oauth_module, ENABLE_OAUTH=True)

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(manager.handle_login(SimpleNamespace(), 'other'), timeout=15)
    assert excinfo.value.status_code == 404


# ── 5+6. token expiry normalisation (#27520, #26802) ───────────────────────


def test_expiry_capped_at_id_token_exp(oauth_module):
    """NARROW: an access token outliving its id_token must not keep forwarding a dead JWT."""
    now = int(time.time())
    token = {'expires_in': 7200, 'id_token': _id_token(now + 600)}

    result = oauth_module._normalize_token_expiry(token)

    assert abs(result['expires_at'] - (now + 600)) <= 5, (
        'expires_at was not capped at the id_token exp'
    )
    assert result['issued_at'] == pytest.approx(now, abs=5)


def test_no_expiry_and_no_refresh_token_is_treated_as_non_expiring(oauth_module):
    """NARROW: without a refresh path an invented 3600s lifetime kills the session for good."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry({'access_token': 'abc'})

    assert result['expires_at'] > now + 365 * 86400, (
        'a refresh-less token was given a short fabricated expiry'
    )
    assert result['expires_at'] == oauth_module.NON_EXPIRING_TOKEN_EXPIRES_AT
    assert result['issued_at'] > 0


def test_no_expiry_with_refresh_token_keeps_the_conservative_default(oauth_module):
    """NEARBY: the 3600s fallback is still right when the session can be refreshed."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry({'access_token': 'abc', 'refresh_token': 'r'})

    assert abs(result['expires_at'] - (now + 3600)) <= 5
    assert result['issued_at'] == pytest.approx(now, abs=5)


def test_malformed_id_token_skips_the_cap_without_raising(oauth_module):
    """NEARBY: an unparseable id_token must not break sign-in, on either ref."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry({'expires_in': 7200, 'id_token': 'not-a-jwt'})

    assert abs(result['expires_at'] - (now + 7200)) <= 5
    assert result['issued_at'] == pytest.approx(now, abs=5)


def test_id_token_without_exp_skips_the_cap(oauth_module):
    """NEARBY: an exp-less id_token leaves the access-token expiry alone, on either ref."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry(
        {'expires_in': 7200, 'id_token': _id_token(None)}
    )

    assert abs(result['expires_at'] - (now + 7200)) <= 5


def test_id_token_outliving_access_token_does_not_extend_it(oauth_module):
    """NEARBY: the cap is a min(), never a bump, on either ref."""
    now = int(time.time())
    result = oauth_module._normalize_token_expiry(
        {'expires_in': 600, 'id_token': _id_token(now + 7200)}
    )

    assert abs(result['expires_at'] - (now + 600)) <= 5


@pytest.mark.parametrize(
    'token',
    [
        {'expires_at': 1900000000},
        {'expires_in': 300},
        {'access_token': 'abc'},
        {'access_token': 'abc', 'refresh_token': 'r'},
        {'expires_in': 300, 'id_token': 'not-a-jwt'},
    ],
)
def test_every_branch_stamps_an_int_expiry_and_an_issued_at(oauth_module, token):
    """BROAD: every resolution path must leave a usable numeric expires_at and issued_at."""
    result = oauth_module._normalize_token_expiry(dict(token))

    assert isinstance(result['expires_at'], int)
    assert result['expires_at'] > 0
    assert result['issued_at'] == pytest.approx(time.time(), abs=5)


# ── 7. JWKS eviction after an IdP key rotation (#27310) ────────────────────


class _RotatingKeyClient:
    """First exchange fails the way authlib 1.7 fails; the retry is bounded by a sentinel."""

    def __init__(self):
        self.calls = 0
        self.server_metadata = {'jwks': {'keys': ['stale']}, 'issuer': 'https://idp.example'}

    async def authorize_access_token(self, request, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise JoseRFCBadSignatureError()
        if self.calls == 2:
            raise _Sentinel('retry reached')
        raise RuntimeError(f'authorize_access_token called {self.calls} times')


@pytest.mark.asyncio
async def test_bad_signature_evicts_jwks_and_retries(monkeypatch, oauth_module):
    """NARROW: joserfc's BadSignatureError must trigger the JWKS eviction and one retry."""
    client = _RotatingKeyClient()
    manager = _make_oauth_manager(monkeypatch, oauth_module, 'testidp', client)
    _patch_runtime_config(monkeypatch, oauth_module, ENABLE_OAUTH=True)

    request = SimpleNamespace(
        base_url='https://owui.example/', app=SimpleNamespace(state=SimpleNamespace(redis=None))
    )
    response = SimpleNamespace(headers={})

    with pytest.raises(_Sentinel):
        await asyncio.wait_for(
            manager.handle_callback(request, 'testidp', response), timeout=15
        )

    assert client.calls == 2, f'token exchange was not retried (calls={client.calls})'
    assert 'jwks' not in client.server_metadata, 'stale JWKS was not evicted'
    assert client.server_metadata['issuer'] == 'https://idp.example', 'eviction was too broad'


@pytest.mark.asyncio
async def test_bad_signature_class_is_the_one_authlib_actually_raises(oauth_module):
    """BROAD: the caught class must be joserfc's, which authlib 1.7 delegates JWS checks to."""
    caught = oauth_module.BadSignatureError
    assert issubclass(JoseRFCBadSignatureError, caught)


@pytest.mark.asyncio
async def test_non_signature_token_error_is_not_retried(monkeypatch, oauth_module):
    """NEARBY: an ordinary exchange failure still fails fast with no retry, on either ref."""
    calls = []

    async def _authorize_access_token(request, **kwargs):
        calls.append(1)
        if len(calls) > CALL_BUDGET:
            raise _Sentinel('retry loop is unbounded')
        raise ValueError('invalid_grant')

    client = SimpleNamespace(
        authorize_access_token=_authorize_access_token,
        server_metadata={'jwks': {'keys': ['fresh']}},
    )
    manager = _make_oauth_manager(monkeypatch, oauth_module, 'testidp', client)
    _patch_runtime_config(monkeypatch, oauth_module, ENABLE_OAUTH=True)

    request = SimpleNamespace(
        base_url='https://owui.example/', app=SimpleNamespace(state=SimpleNamespace(redis=None))
    )
    response = SimpleNamespace(headers={})

    await asyncio.wait_for(manager.handle_callback(request, 'testidp', response), timeout=15)

    assert len(calls) == 1
    assert 'jwks' in client.server_metadata, 'a non-signature failure evicted the JWKS cache'
