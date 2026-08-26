"""Regression: ten 0.11.0 fixes across the ASGI stack, logging, audit and startup.

* Transcription chunk order (#27417, issue #27143, `routers/audio.py`): `transcribe`
  collected chunk results with `asyncio.as_completed`, so a long recording came back
  with its sections in completion order. Fixed by `asyncio.gather`.
* Streamed responses cutting out (#26924, issue #26922, `utils/security_headers.py`):
  `SecurityHeadersMiddleware` was a `BaseHTTPMiddleware`, which re-buffers every
  response through an anyio memory stream. Rewritten as pure ASGI.
* Blocked webhook targets (commit 0671b7a, issue #26975, `utils/webhook.py`): the
  `validate_url` call shared a try block with the POST, so a non-resolvable target
  logged a full traceback. Fixed by giving it its own try that warns and returns False.
* Values in error logs (commit 6aebfd8, #26814, `env.py` + `utils/logger.py`): loguru's
  `diagnose` was left at its default, printing nearby variables (API keys, message
  content) next to every traceback. Fixed by `diagnose=LOGURU_DIAGNOSE`, default off.
* Empty audit exclusion list (#27370, commit 2ef6c76, `utils/audit.py`): an emptied
  `AUDIT_EXCLUDED_PATHS` produced `['']`, whose alternation matched every path and
  silently switched auditing off. Fixed by normalizing and pre-compiling the patterns.
* Readable audit bodies (#27369, `main.py`): `AuditLoggingMiddleware` was added after
  `CompressMiddleware`, so it sat outside compression and recorded gzip bytes. Fixed by
  registering audit first.
* Feedback rating in events (commit 300302d, `routers/evaluations.py`): the event read
  `feedback.rating`, an attribute the model does not have, so every event said None.
* `X-Process-Time` (#27368, `utils/asgi_middleware.py`): the elapsed time was truncated
  with `int()`, so every sub-second response reported `0`.
* Provider rejection logging (#27238, issues #27237/#26253, `events.py`): the upstream
  rejection reason never reached the server log.
* Licensed startup (commits 8f77533, 0c7ddbd): the lifespan fetched the license
  synchronously, blocking readiness on the license server, and an unreachable license
  host propagated out of `handler`.

0.11.1 `b96d2b12d` collapsed the five pure-ASGI HTTP middlewares into one
`utils.asgi_middleware.AppHTTPMiddleware` and deleted `SecurityHeadersMiddleware` from
`utils/security_headers.py` entirely. The merged class keeps both behaviours under test
(pure ASGI, and one `X-Process-Time` plus the configured security headers stamped on
`http.response.start`), so the middleware class is resolved at runtime and the
assertions are unchanged.

Discriminates: passes on v0.11.0 and v0.11.1, fails on v0.10.2 (chunks joined in completion order,
`SecurityHeadersMiddleware` still a `BaseHTTPMiddleware`, blocked webhooks logged via
`log.exception`, no `diagnose` kwarg and no `LOGURU_DIAGNOSE`, `['']` disables auditing,
audit registered after compression, the rating reported as None, `X-Process-Time` an
integer string, no provider rejection log line, license fetched inline and the second
license host never tried).
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def audio_module(owui_module):
    return owui_module('open_webui.routers.audio')


@pytest.fixture(scope='module')
def security_headers_module(owui_module):
    return owui_module('open_webui.utils.security_headers')


@pytest.fixture(scope='module')
def webhook_module(owui_module):
    return owui_module('open_webui.utils.webhook')


@pytest.fixture(scope='module')
def env_module(owui_module):
    return owui_module('open_webui.env')


@pytest.fixture(scope='module')
def logger_module(owui_module):
    return owui_module('open_webui.utils.logger')


@pytest.fixture(scope='module')
def audit_module(owui_module):
    return owui_module('open_webui.utils.audit')


@pytest.fixture(scope='module')
def evaluations_module(owui_module):
    return owui_module('open_webui.routers.evaluations')


@pytest.fixture(scope='module')
def asgi_middleware_module(owui_module):
    return owui_module('open_webui.utils.asgi_middleware')


@pytest.fixture(scope='module')
def events_module(owui_module):
    return owui_module('open_webui.events')


@pytest.fixture(scope='module')
def auth_module(owui_module):
    return owui_module('open_webui.utils.auth')


@pytest.fixture(scope='module')
def main_source(open_webui_backend: Path) -> ast.Module:
    """`main.py` parsed, not imported: importing it builds the whole app."""
    return ast.parse(
        (Path(open_webui_backend) / 'open_webui' / 'main.py').read_text(encoding='utf-8')
    )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _http_scope(path: str = '/api/v1/chats/abc', method: str = 'POST', headers=None) -> dict:
    return {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': method,
        'scheme': 'http',
        'server': ('testserver', 80),
        'root_path': '',
        'path': path,
        'raw_path': path.encode(),
        'query_string': b'',
        'headers': headers or [],
        'client': ('127.0.0.1', 1234),
    }


async def _drive(app, scope, body_chunks=(b'ok',)):
    """Run an ASGI app once; returns (start_message, [body chunks])."""
    incoming = [
        {'type': 'http.request', 'body': b'', 'more_body': False},
        {'type': 'http.disconnect'},
    ]

    async def receive():
        return incoming.pop(0) if incoming else {'type': 'http.disconnect'}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def inner(scope, receive, send):
        await send(
            {
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'content-type', b'text/plain')],
            }
        )
        for index, chunk in enumerate(body_chunks):
            await send(
                {
                    'type': 'http.response.body',
                    'body': chunk,
                    'more_body': index < len(body_chunks) - 1,
                }
            )

    await asyncio.wait_for(app(inner, scope, receive, send), timeout=10)
    start = next(message for message in sent if message['type'] == 'http.response.start')
    body = [message.get('body', b'') for message in sent if message['type'] == 'http.response.body']
    return start, body


def _header_values(start_message: dict, name: bytes) -> list[bytes]:
    return [value for key, value in start_message['headers'] if key.lower() == name]


def _http_middleware(security_headers_module, asgi_middleware_module):
    """The response-stamping HTTP middleware: (class, extra constructor kwargs).

    0.11.1 merged `SecurityHeadersMiddleware` and `AuthTokenMiddleware` into
    `AppHTTPMiddleware`, which drops the `fastapi_app` keyword.
    """
    merged = getattr(asgi_middleware_module, 'AppHTTPMiddleware', None)
    if merged is not None:
        return merged, {}

    split = getattr(security_headers_module, 'SecurityHeadersMiddleware', None)
    if split is not None:
        return split, {}

    raise AssertionError('neither AppHTTPMiddleware nor SecurityHeadersMiddleware exists')


def _timing_middleware(security_headers_module, asgi_middleware_module):
    merged = getattr(asgi_middleware_module, 'AppHTTPMiddleware', None)
    if merged is not None:
        return merged, {}

    split = getattr(asgi_middleware_module, 'AuthTokenMiddleware', None)
    if split is not None:
        return split, {'fastapi_app': SimpleNamespace()}

    raise AssertionError('neither AppHTTPMiddleware nor AuthTokenMiddleware exists')


# --------------------------------------------------------------------------------------
# 1. transcription chunk order (#27417, issue #27143)
# --------------------------------------------------------------------------------------


def _stub_chunking(monkeypatch, audio_module, chunk_paths):
    monkeypatch.setattr(audio_module, 'BYPASS_PYDUB_PREPROCESSING', False)
    monkeypatch.setattr(audio_module, 'is_audio_conversion_required', lambda path: False)
    monkeypatch.setattr(audio_module, 'compress_audio', lambda path: path)
    monkeypatch.setattr(audio_module, 'split_audio', lambda path, max_bytes: list(chunk_paths))


@pytest.mark.asyncio
async def test_transcribe_joins_chunks_in_spoken_order(monkeypatch, audio_module):
    """The last chunk finishes first; the transcript must still read front to back."""
    chunk_paths = [
        '/nonexistent/chunk_0.mp3',
        '/nonexistent/chunk_1.mp3',
        '/nonexistent/chunk_2.mp3',
    ]
    _stub_chunking(monkeypatch, audio_module, chunk_paths)

    last_chunk_done = asyncio.Event()

    async def fake_handler(request, chunk_path, metadata, user):
        index = chunk_paths.index(chunk_path)
        if index == len(chunk_paths) - 1:
            last_chunk_done.set()
        else:
            await asyncio.wait_for(last_chunk_done.wait(), timeout=5)
        return {'text': f'part{index}'}

    monkeypatch.setattr(audio_module, 'transcription_handler', fake_handler)

    result = await asyncio.wait_for(
        audio_module.transcribe(SimpleNamespace(), '/nonexistent/recording.mp3'),
        timeout=15,
    )
    assert result['text'] == 'part0 part1 part2'


@pytest.mark.asyncio
async def test_transcribe_single_chunk_is_unchanged(monkeypatch, audio_module):
    chunk_paths = ['/nonexistent/only.mp3']
    _stub_chunking(monkeypatch, audio_module, chunk_paths)

    async def fake_handler(request, chunk_path, metadata, user):
        return {'text': 'hello'}

    monkeypatch.setattr(audio_module, 'transcription_handler', fake_handler)

    result = await asyncio.wait_for(
        audio_module.transcribe(SimpleNamespace(), '/nonexistent/recording.mp3'),
        timeout=15,
    )
    assert result['text'] == 'hello'


@pytest.mark.asyncio
async def test_transcribe_propagates_chunk_failure(monkeypatch, audio_module):
    from fastapi import HTTPException

    chunk_paths = ['/nonexistent/chunk_0.mp3', '/nonexistent/chunk_1.mp3']
    _stub_chunking(monkeypatch, audio_module, chunk_paths)

    async def fake_handler(request, chunk_path, metadata, user):
        raise RuntimeError('provider down')

    monkeypatch.setattr(audio_module, 'transcription_handler', fake_handler)

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(
            audio_module.transcribe(SimpleNamespace(), '/nonexistent/recording.mp3'),
            timeout=15,
        )
    assert excinfo.value.status_code == 500


# --------------------------------------------------------------------------------------
# 2. streamed responses cutting out (#26924, issue #26922)
# --------------------------------------------------------------------------------------


def test_security_headers_middleware_is_pure_asgi(security_headers_module, asgi_middleware_module):
    """BaseHTTPMiddleware is the re-buffering that truncated streamed audio."""
    from starlette.middleware.base import BaseHTTPMiddleware

    cls, _ = _http_middleware(security_headers_module, asgi_middleware_module)
    assert not issubclass(cls, BaseHTTPMiddleware)
    assert not hasattr(cls, 'dispatch')


@pytest.mark.asyncio
async def test_security_headers_streams_every_chunk_and_stamps_once(
    monkeypatch, security_headers_module, asgi_middleware_module
):
    monkeypatch.setenv('XFRAME_OPTIONS', 'DENY')
    cls, kwargs = _http_middleware(security_headers_module, asgi_middleware_module)
    chunks = [b'chunk-a', b'chunk-b', b'chunk-c']

    async def app(inner, scope, receive, send):
        await cls(inner, **kwargs)(scope, receive, send)

    start, body = await _drive(app, _http_scope(method='GET'), body_chunks=chunks)

    assert b''.join(body) == b''.join(chunks)
    assert _header_values(start, b'x-frame-options') == [b'DENY']


@pytest.mark.asyncio
async def test_security_headers_passes_non_http_scopes_through(
    monkeypatch, security_headers_module, asgi_middleware_module
):
    monkeypatch.setenv('XFRAME_OPTIONS', 'DENY')
    seen: list[str] = []

    async def inner(scope, receive, send):
        seen.append(scope['type'])

    async def receive():
        return {'type': 'lifespan.startup'}

    async def send(message):
        return None

    cls, kwargs = _http_middleware(security_headers_module, asgi_middleware_module)
    middleware = cls(inner, **kwargs)
    await asyncio.wait_for(middleware({'type': 'lifespan'}, receive, send), timeout=5)
    assert seen == ['lifespan']


# --------------------------------------------------------------------------------------
# 3. blocked webhook targets (commit 0671b7a, issue #26975)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_webhook_url_warns_without_traceback(monkeypatch, caplog, webhook_module):
    def blocked(url):
        raise ValueError('URL is not publicly resolvable')

    sessions: list[str] = []

    def record_session():
        sessions.append('opened')
        raise AssertionError('no request must be made for a blocked webhook target')

    monkeypatch.setattr(webhook_module, 'validate_url', blocked)
    monkeypatch.setattr(webhook_module, 'get_ssrf_safe_session', record_session)

    with caplog.at_level(logging.DEBUG, logger='open_webui.utils.webhook'):
        result = await asyncio.wait_for(
            webhook_module.post_webhook(
                'Open WebUI', 'http://169.254.169.254/latest', 'hi', {'event': 'x'}
            ),
            timeout=10,
        )

    assert result is False
    assert sessions == []
    relevant = [record for record in caplog.records if record.name == 'open_webui.utils.webhook']
    assert [record.levelno for record in relevant if record.levelno >= logging.WARNING] == [
        logging.WARNING
    ]
    assert all(record.exc_info is None for record in relevant)


@pytest.mark.asyncio
async def test_webhook_payload_failure_still_logs_the_exception(
    monkeypatch, caplog, webhook_module
):
    """The second try block keeps its traceback; only the URL check was split out."""
    monkeypatch.setattr(webhook_module, 'validate_url', lambda url: None)

    def boom():
        raise RuntimeError('connection reset')

    monkeypatch.setattr(webhook_module, 'get_ssrf_safe_session', boom)

    with caplog.at_level(logging.DEBUG, logger='open_webui.utils.webhook'):
        result = await asyncio.wait_for(
            webhook_module.post_webhook(
                'Open WebUI', 'https://example.com/hook', 'hi', {'event': 'x'}
            ),
            timeout=10,
        )

    assert result is False
    assert any(
        record.levelno >= logging.ERROR and record.exc_info is not None
        for record in caplog.records
        if record.name == 'open_webui.utils.webhook'
    )


# --------------------------------------------------------------------------------------
# 4. values in error logs (commit 6aebfd8, #26814)
# --------------------------------------------------------------------------------------


def test_loguru_diagnose_defaults_to_off(env_module):
    assert hasattr(env_module, 'LOGURU_DIAGNOSE'), 'LOGURU_DIAGNOSE is missing from env.py'
    assert env_module.LOGURU_DIAGNOSE is False


def test_start_logger_passes_diagnose_to_every_sink(monkeypatch, logger_module):
    calls: list[dict] = []

    def fake_add(sink, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(logger_module.logger, 'add', fake_add)
    monkeypatch.setattr(logger_module.logger, 'remove', lambda *args, **kwargs: None)
    monkeypatch.setattr(logger_module.logging, 'basicConfig', lambda *args, **kwargs: None)

    logger_module.start_logger()

    assert calls, 'start_logger registered no sinks'
    assert all('diagnose' in kwargs for kwargs in calls)
    assert all(kwargs['diagnose'] is False for kwargs in calls)


# --------------------------------------------------------------------------------------
# 5. empty audit exclusion list (#27370, commit 2ef6c76)
# --------------------------------------------------------------------------------------


def _audit_middleware(audit_module, **kwargs):
    async def app(scope, receive, send):
        return None

    return audit_module.AuditLoggingMiddleware(
        app, audit_level=audit_module.AuditLevel.REQUEST, **kwargs
    )


def _request(audit_module, path: str, method: str = 'POST'):
    # Unauthenticated requests are skipped before any path matching runs.
    return audit_module.Request(
        _http_scope(path=path, method=method, headers=[(b'authorization', b'Bearer token')])
    )


@pytest.mark.parametrize('excluded', [[''], ['', '  '], ['', '  ', '/chats']])
def test_empty_excluded_entry_does_not_disable_auditing(monkeypatch, audit_module, excluded):
    monkeypatch.setattr(audit_module, 'AUDIT_LOG_LEVEL', 'REQUEST')
    middleware = _audit_middleware(audit_module, excluded_paths=excluded)
    assert middleware._should_skip_auditing(_request(audit_module, '/api/v1/users/update')) is False


def test_real_excluded_path_is_still_excluded(monkeypatch, audit_module):
    monkeypatch.setattr(audit_module, 'AUDIT_LOG_LEVEL', 'REQUEST')
    middleware = _audit_middleware(audit_module, excluded_paths=['', '  ', '/chats'])
    assert middleware._should_skip_auditing(_request(audit_module, '/api/v1/chats/abc')) is True


def test_included_paths_whitelist_still_applies(monkeypatch, audit_module):
    monkeypatch.setattr(audit_module, 'AUDIT_LOG_LEVEL', 'REQUEST')
    middleware = _audit_middleware(audit_module, included_paths=['chats'])
    assert middleware._should_skip_auditing(_request(audit_module, '/api/v1/chats/abc')) is False
    assert middleware._should_skip_auditing(_request(audit_module, '/api/v1/users/update')) is True


def test_audit_skips_unaudited_methods_and_none_level(monkeypatch, audit_module):
    monkeypatch.setattr(audit_module, 'AUDIT_LOG_LEVEL', 'REQUEST')
    middleware = _audit_middleware(audit_module)
    assert (
        middleware._should_skip_auditing(_request(audit_module, '/api/v1/chats/abc', method='GET'))
        is True
    )

    monkeypatch.setattr(audit_module, 'AUDIT_LOG_LEVEL', 'NONE')
    assert middleware._should_skip_auditing(_request(audit_module, '/api/v1/chats/abc')) is True


def test_always_log_endpoints_is_a_class_attribute(monkeypatch, audit_module):
    endpoints = getattr(audit_module.AuditLoggingMiddleware, 'ALWAYS_LOG_ENDPOINTS', None)
    assert endpoints is not None, 'ALWAYS_LOG_ENDPOINTS is not a class attribute'
    assert set(endpoints) == {
        '/api/v1/auths/signin',
        '/api/v1/auths/signout',
        '/api/v1/auths/signup',
    }

    monkeypatch.setattr(audit_module, 'AUDIT_LOG_LEVEL', 'REQUEST')
    middleware = _audit_middleware(audit_module, excluded_paths=['auths'])
    for endpoint in endpoints:
        assert middleware._should_skip_auditing(_request(audit_module, endpoint)) is False


# --------------------------------------------------------------------------------------
# 6. readable audit bodies (#27369)
# --------------------------------------------------------------------------------------


def _add_middleware_order(module: ast.Module) -> dict[str, int]:
    """Line number of each `app.add_middleware(X)` call in main.py, keyed by X."""
    order: dict[str, int] = {}
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_middleware'
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            order.setdefault(node.args[0].id, node.lineno)
    return order


def test_audit_middleware_is_registered_inside_compression(main_source):
    """Starlette runs the last-added middleware outermost, so audit must be added first."""
    order = _add_middleware_order(main_source)
    assert 'AuditLoggingMiddleware' in order
    assert 'CompressMiddleware' in order
    assert order['AuditLoggingMiddleware'] < order['CompressMiddleware']


# --------------------------------------------------------------------------------------
# 7. feedback rating in events (commit 300302d)
# --------------------------------------------------------------------------------------


@pytest.fixture
def captured_events(monkeypatch, evaluations_module):
    events: list[dict] = []

    async def fake_publish_event(request_or_app, event, **kwargs):
        events.append({'event': event, **kwargs})

    monkeypatch.setattr(evaluations_module, 'publish_event', fake_publish_event)
    return events


def _feedback_form(evaluations_module, rating):
    return evaluations_module.FeedbackForm(type='rating', data={'rating': rating})


@pytest.mark.asyncio
@pytest.mark.parametrize('rating', [1, -1])
async def test_created_feedback_event_carries_the_rating(
    monkeypatch, evaluations_module, captured_events, rating
):
    feedback = SimpleNamespace(id='fb-1', data={'rating': rating})

    async def fake_insert(user_id, form_data, db=None):
        return feedback

    monkeypatch.setattr(evaluations_module.Feedbacks, 'insert_new_feedback', fake_insert)

    result = await evaluations_module.create_feedback(
        request=SimpleNamespace(),
        form_data=_feedback_form(evaluations_module, rating),
        user=SimpleNamespace(id='u-1', role='user'),
        db=None,
    )

    assert result is feedback
    assert [event['data'] for event in captured_events] == [{'rating': rating}]


@pytest.mark.asyncio
async def test_updated_feedback_event_carries_the_rating(
    monkeypatch, evaluations_module, captured_events
):
    feedback = SimpleNamespace(id='fb-2', data={'rating': 1})

    async def fake_update(id, form_data, db=None):
        return feedback

    monkeypatch.setattr(evaluations_module.Feedbacks, 'update_feedback_by_id', fake_update)

    await evaluations_module.update_feedback_by_id(
        request=SimpleNamespace(),
        id='fb-2',
        form_data=_feedback_form(evaluations_module, 1),
        user=SimpleNamespace(id='admin-1', role='admin'),
        db=None,
    )

    assert [event['data'] for event in captured_events] == [{'rating': 1}]


@pytest.mark.asyncio
async def test_feedback_event_rating_is_none_when_data_is_missing(
    monkeypatch, evaluations_module, captured_events
):
    feedback = SimpleNamespace(id='fb-3', data=None)

    async def fake_insert(user_id, form_data, db=None):
        return feedback

    monkeypatch.setattr(evaluations_module.Feedbacks, 'insert_new_feedback', fake_insert)

    await evaluations_module.create_feedback(
        request=SimpleNamespace(),
        form_data=_feedback_form(evaluations_module, 1),
        user=SimpleNamespace(id='u-1', role='user'),
        db=None,
    )

    assert [event['data'] for event in captured_events] == [{'rating': None}]


# --------------------------------------------------------------------------------------
# 8. X-Process-Time (#27368)
# --------------------------------------------------------------------------------------


class _FakeConfig:
    @staticmethod
    async def get(key, default=None):
        return True


def _patch_auth_middleware_io(monkeypatch, asgi_middleware_module, elapsed):
    """Fake clock plus the runtime-config read the pre-fix middleware performed."""
    ticks = iter([0.0, elapsed])
    monkeypatch.setattr(
        asgi_middleware_module, 'time', SimpleNamespace(monotonic=lambda: next(ticks))
    )
    monkeypatch.setattr(asgi_middleware_module, 'Config', _FakeConfig, raising=False)


@pytest.mark.asyncio
async def test_process_time_header_keeps_sub_second_precision(
    monkeypatch, security_headers_module, asgi_middleware_module
):
    _patch_auth_middleware_io(monkeypatch, asgi_middleware_module, 0.25)
    cls, kwargs = _timing_middleware(security_headers_module, asgi_middleware_module)

    async def app(inner, scope, receive, send):
        await cls(inner, **kwargs)(scope, receive, send)

    start, _ = await _drive(app, _http_scope(method='GET'))

    values = _header_values(start, b'x-process-time')
    assert values == [b'0.250000']
    assert float(values[0]) == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_auth_token_middleware_stashes_the_bearer_token(
    monkeypatch, security_headers_module, asgi_middleware_module
):
    _patch_auth_middleware_io(monkeypatch, asgi_middleware_module, 0.5)
    scope = _http_scope(method='GET', headers=[(b'authorization', b'Bearer abc123')])
    cls, kwargs = _timing_middleware(security_headers_module, asgi_middleware_module)

    async def app(inner, scope, receive, send):
        await cls(inner, **kwargs)(scope, receive, send)

    await _drive(app, scope)

    assert scope['state']['token'].credentials == 'abc123'


# --------------------------------------------------------------------------------------
# 9. provider rejection logging (#27238, issues #27237/#26253)
# --------------------------------------------------------------------------------------


@pytest.fixture
def silenced_event_sinks(monkeypatch, events_module):
    async def fake_publish_event(request_or_app, event, **kwargs):
        return None

    monkeypatch.setattr(events_module, 'publish_event', fake_publish_event)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'status_code,expected_level',
    [(400, logging.WARNING), (429, logging.WARNING), (503, logging.ERROR)],
)
async def test_provider_rejection_reason_reaches_the_server_log(
    caplog, events_module, silenced_event_sinks, status_code, expected_level
):
    with caplog.at_level(logging.DEBUG, logger='open_webui.events'):
        await asyncio.wait_for(
            events_module.publish_model_provider_request_failed(
                SimpleNamespace(),
                actor=None,
                provider='anthropic',
                base_url='https://api.anthropic.com/v1',
                status=status_code,
                requested_model='claude-sonnet-4-5',
                upstream_error={
                    'error': {'type': 'invalid_request_error', 'message': 'max_tokens is too large'}
                },
            ),
            timeout=10,
        )

    records = [record for record in caplog.records if record.name == 'open_webui.events']
    assert records, 'the upstream rejection was never logged'
    record = records[-1]
    assert record.levelno == expected_level
    message = record.getMessage()
    assert 'max_tokens is too large' in message
    assert 'claude-sonnet-4-5' in message
    assert str(status_code) in message


# --------------------------------------------------------------------------------------
# 10. licensed startup (commits 8f77533, 0c7ddbd)
# --------------------------------------------------------------------------------------


def _lifespan_function(module: ast.Module) -> ast.AsyncFunctionDef:
    for node in module.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'lifespan':
            return node
    raise AssertionError('main.py has no lifespan function')


def test_lifespan_does_not_block_startup_on_the_license_server(main_source):
    lifespan = _lifespan_function(main_source)
    source = ast.dump(lifespan)

    bare_calls = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'get_license_data'
    ]
    assert not bare_calls, 'lifespan still calls get_license_data inline'
    assert "attr='create_task'" in source
    assert "attr='shield'" in source
    assert "attr='wait_for'" in source

    handlers = [node for node in ast.walk(lifespan) if isinstance(node, ast.ExceptHandler)]
    assert any(
        isinstance(handler.type, ast.Attribute) and handler.type.attr == 'TimeoutError'
        for handler in handlers
    ), 'lifespan does not handle the license timeout'


def test_unreachable_license_host_falls_through_to_the_next_one(monkeypatch, auth_module):
    attempted: list[str] = []

    def unreachable(url, **kwargs):
        attempted.append(url)
        raise OSError('name or service not known')

    monkeypatch.setattr(auth_module.requests, 'post', unreachable)
    app = SimpleNamespace(state=SimpleNamespace())

    assert auth_module.get_license_data(app, 'test-license-key') is False
    assert len(attempted) == 2, 'the second license host was never tried'
    assert attempted[0] != attempted[1]
