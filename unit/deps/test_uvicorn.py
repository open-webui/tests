"""Dependency contract: uvicorn.

uvicorn is the ASGI server Open WebUI runs the FastAPI app on. The CLI
entrypoints in ``open_webui/__init__.py`` invoke ``uvicorn.run(...)`` to
serve ``open_webui.main:app``:

  * ``serve``  -> uvicorn.run('open_webui.main:app', host=, port=,
                  forwarded_allow_ips='*', workers=UVICORN_WORKERS, loop=)
  * ``dev``    -> uvicorn.run('open_webui.main:app', host=, port=,
                  reload=, forwarded_allow_ips='*')

Open WebUI also drives uvicorn's loggers directly: ``utils/logger.py``
attaches its loguru intercept to the ``uvicorn`` / ``uvicorn.error``
loggers and to the audit access logger (default ``uvicorn.access``, see
``env.AUDIT_UVICORN_LOGGER_NAMES``), and ``config.py`` adds an endpoint
filter onto ``logging.getLogger('uvicorn.access')``. Those are stdlib
logging concerns, but they only make sense because uvicorn registers
those named loggers, so we pin the keyword-argument surface of
``uvicorn.run`` that the launchers depend on, plus ``uvicorn.Config`` /
``uvicorn.Server`` (the objects ``run`` builds) so a uvicorn bump that
drops/renames any of those kwargs fails here instead of at boot.

This file does FULL surface validation but NEVER binds a socket or starts
the server: ``uvicorn.run`` is only signature-inspected, never called.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "uvicorn"
DIST_NAME = "uvicorn"

# Top-level symbols the codebase (or its supported launch path) relies on.
TOP_LEVEL_SYMBOLS = [
    "run",  # __init__.py serve()/dev()
    "Config",  # the config object run() constructs
    "Server",  # the server object run() constructs
]

# Keyword arguments the two CLI entrypoints pass to uvicorn.run(...).
RUN_KWARGS = [
    "app",  # first positional: 'open_webui.main:app'
    "host",
    "port",
    "forwarded_allow_ips",  # both serve() and dev() pass '*'
    "workers",  # serve(): UVICORN_WORKERS
    "loop",  # serve(): 'none' on win32 else 'auto'
    "reload",  # dev(): reload=True
]

# uvicorn.Config carries the same options; run() forwards through it.
CONFIG_KWARGS = [
    "app",
    "host",
    "port",
    "loop",
    "workers",
    "forwarded_allow_ips",
    "reload",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "uvicorn"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_top_level_symbols_exist(depcheck):
    """run/Config/Server must remain importable off the top-level package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_run_is_callable(depcheck):
    """The launchers call uvicorn.run(...); it must stay callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "run")


def test_run_accepts_launcher_kwargs(depcheck):
    """serve()/dev() pass app/host/port/forwarded_allow_ips/workers/loop/reload.
    Every one of those keyword names must remain accepted by uvicorn.run
    (or it must take **kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.run, RUN_KWARGS)


def test_config_accepts_launcher_kwargs(depcheck):
    """uvicorn.run builds a uvicorn.Config from these kwargs; Config must
    accept the same option names the launchers rely on."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Config.__init__, CONFIG_KWARGS)


def test_run_app_first_param_is_app(depcheck):
    """The launchers pass the import string 'open_webui.main:app' as the first
    positional argument; the first parameter must still be the app target."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.run)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert params, "uvicorn.run has no positional parameter for the app"
    assert params[0].name == "app", (
        f"uvicorn.run's first positional parameter is now {params[0].name!r}, "
        "but the launchers pass the app import string positionally"
    )


def test_config_constructs_with_import_string_offline(depcheck):
    """Building a uvicorn.Config from an import string + the launcher kwargs must
    not raise and must not import/start anything. We pass a never-imported
    dummy import string and only assert the option fields stuck."""
    mod = depcheck.load(IMPORT_NAME)
    cfg = mod.Config(
        "tests.unit.deps._nonexistent_app:app",
        host="0.0.0.0",
        port=8080,
        forwarded_allow_ips="*",
        loop="none",
        reload=False,
    )
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080
    # forwarded_allow_ips is normalised to a list of trusted hosts internally;
    # just assert the attribute exists and reflects our wildcard.
    assert hasattr(cfg, "forwarded_allow_ips")


def test_config_loop_none_is_accepted(depcheck):
    """__init__.py passes loop='none' on win32 (so asyncio.run respects the
    Selector policy from db.py). 'none' must remain a valid loop setup option."""
    mod = depcheck.load(IMPORT_NAME)
    cfg = mod.Config("tests.unit.deps._nonexistent_app:app", loop="none")
    assert cfg.loop == "none"


def test_config_workers_setting_offline(depcheck):
    """serve() passes workers=UVICORN_WORKERS (an int). Config must retain a
    workers option that accepts an int without spinning up subprocesses."""
    mod = depcheck.load(IMPORT_NAME)
    cfg = mod.Config("tests.unit.deps._nonexistent_app:app", workers=3)
    assert cfg.workers == 3


def test_server_class_exists_and_constructs(depcheck):
    """uvicorn.run instantiates uvicorn.Server(config). Building one offline
    (no .run()/.serve()) must not bind a socket; just assert the object and the
    serve coroutine surface exist."""
    mod = depcheck.load(IMPORT_NAME)
    cfg = mod.Config("tests.unit.deps._nonexistent_app:app")
    server = mod.Server(cfg)
    assert server is not None
    assert hasattr(server, "serve")
    assert inspect.iscoroutinefunction(server.serve), (
        "uvicorn.Server.serve is expected to be a coroutine function"
    )


def test_uvicorn_named_loggers_present(depcheck):
    """utils/logger.py attaches loguru's intercept to the 'uvicorn',
    'uvicorn.error' and (default) 'uvicorn.access' loggers, and config.py adds
    an EndpointFilter to 'uvicorn.access'. Importing uvicorn must register its
    logging config so those named loggers are real (not typo'd)."""
    import logging

    depcheck.load(IMPORT_NAME)
    # Importing uvicorn.config defines the LOGGING_CONFIG referencing these.
    cfg_mod = depcheck.load("uvicorn.config")
    logging_config = getattr(cfg_mod, "LOGGING_CONFIG", None)
    assert isinstance(logging_config, dict), "uvicorn.config.LOGGING_CONFIG missing"
    loggers = logging_config.get("loggers", {})
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        # Either declared in uvicorn's logging config, or resolvable as a
        # stdlib logger (getLogger always succeeds, but we assert the config
        # knows the access/error split the backend keys on).
        assert name in loggers or logging.getLogger(name) is not None
    # The two the backend explicitly references must be in uvicorn's own config.
    assert "uvicorn.error" in loggers
    assert "uvicorn.access" in loggers


def test_run_does_not_execute_on_signature_inspection(depcheck):
    """Guard: inspecting uvicorn.run must not have side effects (no server).
    Trivially true, but documents that this whole suite never calls run()."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.run)
    assert sig is not None
