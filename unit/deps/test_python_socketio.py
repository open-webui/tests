"""Dependency contract: python-socketio (import name ``socketio``).

python-socketio is Open WebUI's entire realtime layer. ``socket/main.py``
constructs a single ``socketio.AsyncServer`` (``async_mode='asgi'``) with a
fixed set of constructor kwargs (cors_allowed_origins, transports,
allow_upgrades, always_connect, client_manager, logger, ping_interval,
ping_timeout, engineio_logger), wraps it in a ``socketio.ASGIApp`` mounted
at ``/ws/socket.io``, and, when ``WEBSOCKET_MANAGER == 'redis'``, swaps the
in-process client manager for a ``socketio.AsyncRedisManager`` (multi-worker
fan-out). Every realtime feature is a handler registered with the
``@sio.on('<event>')`` / ``@sio.event`` decorators (usage, connect,
user-join, heartbeat, join-channels, the whole ``ydoc:*`` collaborative-doc
protocol, disconnect). The server fans messages out with
``sio.emit(event, data, room=/to=, skip_sid=)``, does request/response RPC
with ``sio.call(event, data, to=, timeout=)``, manages room membership with
``sio.enter_room`` / ``sio.leave_room`` / ``sio.disconnect``, and reads the
live participant set through ``sio.manager.get_participants(namespace=,
room=)``.

This module pins exactly that slice so a python-socketio bump that
removed/renamed any of it fails loudly here instead of as a runtime
``AttributeError`` / ``TypeError`` deep in the websocket path (which would
silently break every realtime feature). Two layers, mirroring
test_requests.py / test_redis.py: symbol-existence + signature checks for the
API surface, plus offline BEHAVIOURAL contracts that construct a real
``AsyncServer`` / ``ASGIApp`` / ``AsyncRedisManager`` in-process (no ASGI
server, no socket, no redis) and assert handler registration, the manager
participant API, and the decorator/emit/call shapes the backend relies on.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "socketio"
DIST_NAME = "python-socketio"


# ---------------------------------------------------------------------------
# Symbol inventory — every dotted name socket/main.py resolves on `socketio`.
# ---------------------------------------------------------------------------

TOP_LEVEL_SYMBOLS = [
    "AsyncServer",  # the realtime server (socket/main.py)
    "ASGIApp",  # the ASGI wrapper mounted at /ws/socket.io
    "AsyncRedisManager",  # multi-worker client manager (redis fan-out)
    "AsyncManager",  # the default in-process async manager base
]

# AsyncServer methods socket/main.py calls on the `sio` instance.
SERVER_METHODS = [
    "on",  # @sio.on('<event>')
    "event",  # @sio.event
    "emit",  # await sio.emit(...)
    "call",  # await sio.call(...)
    "enter_room",  # await sio.enter_room(sid, room)
    "leave_room",  # await sio.leave_room(sid, room)
    "disconnect",  # await sio.disconnect(sid)
]

# Constructor kwargs socket/main.py passes to AsyncServer (beyond the explicit
# `client_manager` parameter, these flow through **kwargs to the engine/server).
SERVER_INIT_KWARGS_FLOW_THROUGH = [
    "cors_allowed_origins",
    "async_mode",
    "transports",
    "allow_upgrades",
    "always_connect",
    "ping_interval",
    "ping_timeout",
    "engineio_logger",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`socketio` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "socketio"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test. NOTE: the import package
    `socketio` does not expose `__version__`; resolve via the python-socketio
    distribution metadata instead."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_is_v5(depcheck):
    """socket/main.py is written against python-socketio 5.x (the AsyncServer
    asgi mode, AsyncRedisManager, sio.manager.get_participants API). Guard the
    major version so a 5->6 reshuffle is caught here."""
    major = depcheck.dist_version(DIST_NAME).split(".")[0]
    assert major == "5", (
        f"Expected python-socketio 5.x, got {depcheck.dist_version(DIST_NAME)}. "
        "socket/main.py relies on the 5.x AsyncServer/ASGIApp/manager API."
    )


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `socketio.*` class socket/main.py imports must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_async_server_and_asgi_app_callable(depcheck):
    """socket/main.py: socketio.AsyncServer(...) and socketio.ASGIApp(...).
    Both must be callable class factories."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "AsyncServer")
    depcheck.assert_callable(mod, "ASGIApp")
    depcheck.assert_callable(mod, "AsyncRedisManager")


def test_async_redis_manager_is_async_manager(depcheck):
    """socket/main.py passes the AsyncRedisManager as `client_manager=` to the
    AsyncServer; it must remain an AsyncManager subclass so the server treats it
    as a drop-in for the default in-process manager (same fan-out contract)."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.AsyncRedisManager, mod.AsyncManager), (
        "AsyncRedisManager no longer subclasses AsyncManager; the redis "
        "client_manager swap in socket/main.py would break."
    )


# ---------------------------------------------------------------------------
# AsyncServer constructor signature contract
# ---------------------------------------------------------------------------


def test_async_server_init_accepts_client_manager_and_logger(depcheck):
    """socket/main.py passes client_manager=redis_manager and logger=... as
    explicit named args. Both must remain accepted parameters."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncServer.__init__, ["client_manager", "logger"])


def test_async_server_init_accepts_our_kwargs(depcheck):
    """The remaining AsyncServer kwargs socket/main.py passes (cors_allowed_origins,
    async_mode, transports, allow_upgrades, always_connect, ping_interval,
    ping_timeout, engineio_logger) flow through **kwargs to the Server/engine.
    Constructing with all of them must not raise — that is the real contract,
    since they aren't named parameters on AsyncServer.__init__."""
    mod = depcheck.load(IMPORT_NAME)
    sio = mod.AsyncServer(
        cors_allowed_origins="*",
        async_mode="asgi",
        transports=["websocket"],
        allow_upgrades=True,
        always_connect=True,
        logger=False,
        ping_interval=25,
        ping_timeout=20,
        engineio_logger=False,
    )
    assert sio is not None
    # The kwargs we care about must be representable; AsyncServer keeps **kwargs.
    init_params = inspect.signature(mod.AsyncServer.__init__).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in init_params.values()), (
        "AsyncServer.__init__ dropped **kwargs; the engine/server kwargs would be rejected"
    )
    # These pass-through kwargs are precisely the ones NOT declared as named
    # AsyncServer parameters (so they must ride **kwargs to the engine/server).
    named = {p for p in init_params if init_params[p].kind is not inspect.Parameter.VAR_KEYWORD}
    riding_kwargs = [k for k in SERVER_INIT_KWARGS_FLOW_THROUGH if k not in named]
    promoted = set(SERVER_INIT_KWARGS_FLOW_THROUGH) - set(riding_kwargs)
    assert riding_kwargs == SERVER_INIT_KWARGS_FLOW_THROUGH, (
        "Some pass-through kwargs became named params unexpectedly; if **kwargs "
        f"is later dropped these would break: {promoted}"
    )


def test_asgi_app_init_accepts_server_and_path(depcheck):
    """socket/main.py: ASGIApp(sio, socketio_path='/ws/socket.io'). The first
    positional (the server) and the socketio_path kwarg must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ASGIApp.__init__, ["socketio_server", "socketio_path"])


# ---------------------------------------------------------------------------
# AsyncServer instance method surface
# ---------------------------------------------------------------------------


def test_server_methods_exist(depcheck):
    """Every AsyncServer method socket/main.py calls must exist on the class."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.AsyncServer))
    missing = [m for m in SERVER_METHODS if m not in names]
    assert not missing, f"socketio.AsyncServer missing method(s) socket/main.py calls: {missing}"
    for m in SERVER_METHODS:
        assert callable(getattr(mod.AsyncServer, m)), f"AsyncServer.{m} is not callable"


def test_emit_signature(depcheck):
    """socket/main.py emits with keyword targeting: sio.emit(event, data,
    room=, skip_sid=) and sio.emit(event, data, to=). Pin those kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncServer.emit, ["event", "data", "room", "to", "skip_sid"])


def test_call_signature(depcheck):
    """get_event_call: sio.call('events', payload, to=session_id,
    timeout=WEBSOCKET_EVENT_CALLER_TIMEOUT). The to= and timeout= kwargs must
    remain accepted (the RPC contract for the event caller)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncServer.call, ["event", "data", "to", "timeout"])


def test_enter_leave_room_signature(depcheck):
    """sio.enter_room(sid, room) / sio.leave_room(sid, f'doc_{...}'). Both take
    the sid + room positionals."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AsyncServer.enter_room, ["sid", "room"])
    depcheck.assert_params(mod.AsyncServer.leave_room, ["sid", "room"])


def test_emit_and_call_are_coroutine_functions(depcheck):
    """socket/main.py `await`s sio.emit(...) and sio.call(...); both must be
    async (coroutine) functions on the AsyncServer."""
    mod = depcheck.load(IMPORT_NAME)
    assert asyncio.iscoroutinefunction(mod.AsyncServer.emit), "AsyncServer.emit is no longer async"
    assert asyncio.iscoroutinefunction(mod.AsyncServer.call), "AsyncServer.call is no longer async"
    assert asyncio.iscoroutinefunction(mod.AsyncServer.enter_room), (
        "AsyncServer.enter_room is no longer async"
    )


# ---------------------------------------------------------------------------
# Behavioural: construct a real AsyncServer offline (no ASGI server, no socket)
# ---------------------------------------------------------------------------


def _server(mod):
    """An offline AsyncServer mirroring socket/main.py's non-redis branch.

    async_mode='asgi' constructs without binding a socket or starting a server;
    nothing is awaited, so no event loop / network is needed for construction."""
    return mod.AsyncServer(
        cors_allowed_origins="*",
        async_mode="asgi",
        transports=["websocket"],
        allow_upgrades=True,
        always_connect=True,
        logger=False,
        engineio_logger=False,
    )


def test_behaviour_construct_asgi_server(depcheck):
    """socket/main.py builds the AsyncServer with async_mode='asgi'. Construction
    must succeed offline and expose the manager + handler machinery."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)
    assert sio is not None
    assert hasattr(sio, "manager"), "AsyncServer instance has no .manager"
    assert hasattr(sio, "handlers"), "AsyncServer instance has no .handlers registry"


def test_behaviour_asgi_app_wraps_server(depcheck):
    """app = socketio.ASGIApp(sio, socketio_path='/ws/socket.io'). The wrapper
    must construct and be an ASGI callable (the object FastAPI mounts)."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)
    app = mod.ASGIApp(sio, socketio_path="/ws/socket.io")
    assert app is not None
    assert callable(app), "ASGIApp instance is not callable (not an ASGI app)"


def test_behaviour_on_decorator_registers_named_event(depcheck):
    """Every realtime feature is `@sio.on('<event>')`. Registering a handler must
    place it in the server's handler registry for the default namespace, keyed by
    the event name (socket/main.py: 'usage', 'user-join', 'heartbeat',
    'ydoc:document:join', ...)."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)

    @sio.on("ydoc:document:join")
    async def _join(sid, data):  # pragma: no cover - never invoked here
        return {"ok": True}

    ns = sio.handlers.get("/", {})
    assert "ydoc:document:join" in ns, (
        "@sio.on('<event>') no longer registers under handlers['/']; the "
        "realtime event routing in socket/main.py would silently break."
    )
    assert callable(ns["ydoc:document:join"])


def test_behaviour_event_decorator_registers_by_function_name(depcheck):
    """socket/main.py uses bare `@sio.event` for connect/disconnect, which
    registers the handler under its function name. Pin that connect/disconnect
    land in the registry under those exact names."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)

    @sio.event
    async def connect(sid, environ, auth):  # pragma: no cover
        return True

    @sio.event
    async def disconnect(sid, reason=None):  # pragma: no cover
        return None

    ns = sio.handlers.get("/", {})
    assert "connect" in ns, "@sio.event did not register `connect` by name"
    assert "disconnect" in ns, "@sio.event did not register `disconnect` by name"


def test_behaviour_multiple_handlers_coexist(depcheck):
    """socket/main.py registers ~15 handlers on one server; registering several
    must not clobber earlier ones (each event key is independent)."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)
    events = ["usage", "user-join", "heartbeat", "join-channels", "events:channel"]
    for ev in events:

        @sio.on(ev)
        async def _h(sid, data):  # pragma: no cover
            return None

    ns = sio.handlers.get("/", {})
    missing = [ev for ev in events if ev not in ns]
    assert not missing, f"handler registration dropped event(s): {missing}"


def test_behaviour_manager_get_participants_exists(depcheck):
    """get_session_ids_from_room / channel_events call
    sio.manager.get_participants(namespace='/', room=room) and iterate
    (sid, eio_sid) tuples. Pin that the live server's manager exposes that
    method (the room->participant readback the backend depends on)."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)
    assert hasattr(sio.manager, "get_participants"), "manager.get_participants is gone"
    assert callable(sio.manager.get_participants)
    depcheck.assert_params(sio.manager.get_participants, ["namespace", "room"])


def test_behaviour_redis_manager_constructs_offline(depcheck):
    """socket/main.py: socketio.AsyncRedisManager(ws_redis_url,
    redis_options=WEBSOCKET_REDIS_OPTIONS). The manager must construct lazily
    (no connect) and expose the same get_participants room API as the default
    manager, since it's swapped in as client_manager=."""
    mod = depcheck.load(IMPORT_NAME)
    mgr = mod.AsyncRedisManager("redis://localhost:6379/0")
    assert mgr is not None
    assert hasattr(mgr, "get_participants"), "AsyncRedisManager lacks get_participants"
    assert callable(mgr.get_participants)


def test_behaviour_redis_manager_accepts_redis_options(depcheck):
    """socket/main.py passes redis_options=WEBSOCKET_REDIS_OPTIONS. Constructing
    with that kwarg must not raise (it remains an accepted constructor kwarg)."""
    mod = depcheck.load(IMPORT_NAME)
    mgr = mod.AsyncRedisManager(
        "redis://localhost:6379/0",
        redis_options={"socket_connect_timeout": 5},
    )
    assert mgr is not None


def test_behaviour_redis_manager_attaches_as_client_manager(depcheck):
    """The redis branch builds the server as AsyncServer(client_manager=mgr,...).
    The server must accept the redis manager and wire it up as its .manager,
    proving the multi-worker fan-out path constructs offline."""
    mod = depcheck.load(IMPORT_NAME)
    mgr = mod.AsyncRedisManager("redis://localhost:6379/0")
    sio = mod.AsyncServer(
        client_manager=mgr,
        cors_allowed_origins="*",
        async_mode="asgi",
        logger=False,
        engineio_logger=False,
    )
    assert sio.manager is mgr, (
        "AsyncServer no longer adopts the passed client_manager as .manager; "
        "the redis fan-out wiring in socket/main.py would be a no-op."
    )


# ---------------------------------------------------------------------------
# Behavioural: handler invocation shape (the (sid, data) calling convention)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_behaviour_registered_handler_is_dispatchable(depcheck):
    """socket/main.py handlers take (sid, data) and may return a value (e.g.
    user_join returns {'id', 'name'}). The server's internal dispatch
    (_trigger_event) must invoke the registered coroutine with the sid +
    arguments and return its value — pin that calling convention offline."""
    mod = depcheck.load(IMPORT_NAME)
    sio = _server(mod)
    seen = {}

    @sio.on("user-join")
    async def _join(sid, data):
        seen["sid"] = sid
        seen["data"] = data
        return {"id": "u1", "name": "n"}

    # _trigger_event is the documented dispatch entry the server uses to fan an
    # incoming event to its handler; exercising it needs no transport/socket.
    result = await sio._trigger_event("user-join", "/", "SID123", {"auth": {"token": "t"}})
    assert seen["sid"] == "SID123"
    assert seen["data"] == {"auth": {"token": "t"}}
    assert result == {"id": "u1", "name": "n"}, (
        "handler return value is no longer propagated by the server dispatch; "
        "user_join's ack ({'id','name'}) would be lost."
    )
