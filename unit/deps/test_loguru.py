"""Dependency contract: loguru (import name ``loguru``).

loguru is the Open WebUI backend's logging backbone. ``utils/logger.py``
reconfigures it as the single sink for *all* logging (stdlib logging is
intercepted and re-emitted through loguru), and ``utils/audit.py`` layers a
bound audit logger on top. The backend leans on a large, specific slice of
the API:

  - ``from loguru import logger`` — the singleton; ``Message`` / ``Record``
    / ``Logger`` are imported only under ``TYPE_CHECKING`` (type hints), so
    they are NOT runtime attributes of the package.
  - ``logger.remove()`` then ``logger.add(sink, level=, format=, filter=)``
    to install stdout/json/file sinks; the file sink also passes
    ``rotation=`` and ``compression="zip"``.
  - custom *sink callables* and *format callables* that receive a record and
    read a fixed set of keys: ``record["time"]`` (a datetime with
    ``.isoformat`` / ``.timestamp``), ``record["level"].name``,
    ``record["name"]`` / ``["function"]`` / ``["line"]``,
    ``record["message"]``, ``record["extra"]`` (a dict), and
    ``record["exception"]`` (a tuple-like with ``.type`` / ``.value`` /
    ``.traceback``).
  - ``logger.level(name)`` to resolve a level object by name (with a
    ``ValueError`` fallback for unknown names in the stdlib intercept).
  - ``logger.opt(depth=, exception=).bind(**extras).log(level, msg)`` — the
    exact chain ``InterceptHandler.emit`` uses.
  - ``logger.bind(auditable=True)`` to tag audit records for a filter.

This module pins that surface and, more importantly, the *behavioural
contract* of the record dict and the add/remove/bind/opt/level chain —
exercised OFFLINE against an in-memory sink (no files, no stdout capture,
no network). A loguru bump that renamed a record key, changed ``add``'s
keyword surface, or altered the exception-record shape would fail here
instead of silently breaking structured logging or the audit trail.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "loguru"
DIST_NAME = "loguru"

# Methods the backend calls on the `logger` singleton.
USED_LOGGER_METHODS = [
    "remove",
    "add",
    "level",
    "opt",
    "bind",
    "log",
    "info",
    "error",
    "debug",
    "warning",
]

# add() keyword arguments the backend passes.
ADD_KWARGS = ["level", "format", "filter", "rotation", "compression"]

# opt() keyword arguments the InterceptHandler uses.
OPT_KWARGS = ["depth", "exception"]


def _fresh_logger(depcheck):
    """Return the loguru logger with all handlers removed, so a test can add
    its own in-memory sink without interference. Tests must restore by
    calling logger.remove() on their own sink id.
    """
    mod = depcheck.load(IMPORT_NAME)
    logger = mod.logger
    logger.remove()
    return logger


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "loguru"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_logger_singleton_exists(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "logger")
    assert type(mod.logger).__name__ == "Logger"


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_logger_methods_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    logger = mod.logger
    names = set(dir(logger))
    missing = [m for m in USED_LOGGER_METHODS if m not in names]
    assert not missing, f"loguru.logger missing method(s): {missing}"
    for m in USED_LOGGER_METHODS:
        assert callable(getattr(logger, m)), f"logger.{m} not callable"


def test_add_keyword_surface(depcheck):
    """logger.add(sink, level=, format=, filter=, rotation=, compression=).
    Pin those keyword names remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.logger.add)
    params = sig.parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    # rotation/compression go through **kwargs for file sinks; level/format/
    # filter are explicit. Assert the explicit ones are real params.
    for name in ("level", "format", "filter"):
        assert name in params, f"logger.add lost keyword {name!r}"
    if not has_var_kw:
        for name in ("rotation", "compression"):
            assert name in params, f"logger.add lost keyword {name!r}"


def test_opt_keyword_surface(depcheck):
    """InterceptHandler uses logger.opt(depth=..., exception=...)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.logger.opt, OPT_KWARGS)


def test_level_signature(depcheck):
    """logger.level(name) resolves a level by name."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.logger.level)
    params = list(sig.parameters.values())
    assert params and params[0].name == "name"


# --------------------------------------------------------------------------- #
# Behavioural: add/remove a sink + the record dict shape
# --------------------------------------------------------------------------- #


def test_add_returns_sink_id_and_remove_accepts_it(depcheck):
    """logger.add returns an int handler id; logger.remove(id) detaches it.
    start_logger relies on remove() to reset all handlers first."""
    logger = _fresh_logger(depcheck)
    captured = []
    sink_id = logger.add(lambda m: captured.append(m), level="DEBUG")
    try:
        assert isinstance(sink_id, int)
    finally:
        logger.remove(sink_id)
    # After removal, a log emits nothing to that sink.
    logger.info("after removal")
    assert captured == []


def test_record_dict_has_expected_keys(depcheck):
    """A sink callable receives a Message whose .record exposes every key the
    backend's stdout_format / _json_sink / file_format read."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.info("hello world")
    finally:
        logger.remove(sink_id)

    assert records, "sink never received a record"
    rec = records[0]
    for key in ("time", "level", "name", "function", "line", "message", "extra"):
        assert key in rec, f"loguru record missing key {key!r}"
    # exception key is present (None when no exception).
    assert "exception" in rec
    assert rec["message"] == "hello world"
    assert isinstance(rec["extra"], dict)
    assert isinstance(rec["line"], int)


def test_record_time_is_datetime_with_isoformat_and_timestamp(depcheck):
    """_json_sink calls record['time'].isoformat(timespec='milliseconds') and
    file_format calls record['time'].timestamp(). Pin that 'time' is a
    datetime supporting both."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.info("t")
    finally:
        logger.remove(sink_id)
    t = records[0]["time"]
    assert isinstance(t, datetime.datetime)
    assert isinstance(t.isoformat(timespec="milliseconds"), str)
    assert isinstance(t.timestamp(), float)


def test_record_level_has_name(depcheck):
    """_json_sink reads record['level'].name; stdout_format relies on the
    level object too."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.warning("w")
    finally:
        logger.remove(sink_id)
    lvl = records[0]["level"]
    assert lvl.name == "WARNING"


# --------------------------------------------------------------------------- #
# Behavioural: level filtering
# --------------------------------------------------------------------------- #


def test_level_threshold_filters_below(depcheck):
    """A sink added at level=WARNING must drop INFO/DEBUG records — this is how
    GLOBAL_LOG_LEVEL gates output."""
    logger = _fresh_logger(depcheck)
    msgs = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    try:
        logger.debug("dropped-debug")
        logger.info("dropped-info")
        logger.warning("kept-warning")
        logger.error("kept-error")
    finally:
        logger.remove(sink_id)
    assert "dropped-debug" not in msgs
    assert "dropped-info" not in msgs
    assert "kept-warning" in msgs
    assert "kept-error" in msgs


def test_level_lookup_by_name(depcheck):
    """logger.level('INFO') returns a level object whose .name round-trips;
    InterceptHandler does logger.level(record.levelname).name."""
    mod = depcheck.load(IMPORT_NAME)
    info = mod.logger.level("INFO")
    assert info.name == "INFO"
    assert isinstance(info.no, int)


def test_level_lookup_unknown_raises_value_error(depcheck):
    """InterceptHandler wraps logger.level(...) in `except ValueError`. An
    unknown level name must raise ValueError so that fallback path stays live."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(ValueError):
        mod.logger.level("NOPE_NOT_A_LEVEL")


# --------------------------------------------------------------------------- #
# Behavioural: bind / extra (the audit filter mechanism)
# --------------------------------------------------------------------------- #


def test_bind_attaches_extra_to_record(depcheck):
    """AuditLogger does logger.bind(auditable=True); the audit file filter is
    `record['extra'].get('auditable') is True`. Prove bound kwargs land in
    record['extra']."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.bind(auditable=True, id="abc").info("audit-line")
    finally:
        logger.remove(sink_id)
    extra = records[0]["extra"]
    assert extra.get("auditable") is True
    assert extra.get("id") == "abc"


def test_filter_on_extra_excludes_unbound_records(depcheck):
    """start_logger's audit_filter is essentially
    `lambda r: 'auditable' not in r['extra']`. Replicate it: a plain log
    passes, a bound-auditable log is filtered out."""
    logger = _fresh_logger(depcheck)
    seen = []
    sink_id = logger.add(
        lambda m: seen.append(m.record["message"]),
        level="DEBUG",
        filter=lambda r: "auditable" not in r["extra"],
    )
    try:
        logger.info("normal")
        logger.bind(auditable=True).info("audit")
    finally:
        logger.remove(sink_id)
    assert "normal" in seen
    assert "audit" not in seen


def test_bind_is_isolated_per_call(depcheck):
    """bind returns a new contextualized logger; the base logger's records
    must not carry a previous bind's extras."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.bind(auditable=True).info("bound")
        logger.info("plain")
    finally:
        logger.remove(sink_id)
    assert records[0]["extra"].get("auditable") is True
    assert "auditable" not in records[1]["extra"]


# --------------------------------------------------------------------------- #
# Behavioural: opt(...).log(...) chain + level=string in log()
# --------------------------------------------------------------------------- #


def test_log_with_string_level(depcheck):
    """AuditLogger.write calls self.logger.log(log_level, '') with a string
    level like 'INFO'. logger.log must accept a level name string."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.log("INFO", "string-level-message")
    finally:
        logger.remove(sink_id)
    assert records[0]["level"].name == "INFO"
    assert records[0]["message"] == "string-level-message"


def test_opt_bind_log_chain(depcheck):
    """InterceptHandler.emit does
    logger.opt(depth=..., exception=...).bind(**extras).log(level, msg).
    Prove the whole chain delivers a record with the message and extras."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.opt(depth=0, exception=None).bind(trace_id="t1").log("INFO", "chained")
    finally:
        logger.remove(sink_id)
    assert records[0]["message"] == "chained"
    assert records[0]["extra"].get("trace_id") == "t1"


# --------------------------------------------------------------------------- #
# Behavioural: exception record shape (_json_sink reads .type/.value/.traceback)
# --------------------------------------------------------------------------- #


def test_exception_record_shape(depcheck):
    """_json_sink reads record['exception'].type/.value/.traceback. When an
    exception is logged, that field must be a tuple-like exposing those
    three attributes; otherwise it's None."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            logger.opt(exception=True).error("caught")
    finally:
        logger.remove(sink_id)

    exc = records[0]["exception"]
    assert exc is not None
    assert exc.type is ValueError
    assert isinstance(exc.value, ValueError)
    assert str(exc.value) == "boom"
    assert exc.traceback is not None


def test_no_exception_record_is_none(depcheck):
    """A plain log has record['exception'] is None — the `if exc is not None`
    branch in _json_sink must not fire for ordinary lines."""
    logger = _fresh_logger(depcheck)
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        logger.info("no-exc")
    finally:
        logger.remove(sink_id)
    assert records[0]["exception"] is None


# --------------------------------------------------------------------------- #
# Behavioural: custom format callable (stdout_format / file_format)
# --------------------------------------------------------------------------- #


def test_format_callable_receives_record(depcheck):
    """logger.add(sink, format=callable) is used (stdout_format/file_format are
    functions). The format callable must be invoked with the record and its
    return value used as the template."""
    logger = _fresh_logger(depcheck)
    seen_records = []

    def fmt(record):
        seen_records.append(record)
        return "{message}\n"

    written = []
    sink_id = logger.add(lambda m: written.append(str(m)), format=fmt, level="DEBUG")
    try:
        logger.info("formatted")
    finally:
        logger.remove(sink_id)
    assert seen_records, "format callable was never called"
    assert seen_records[0]["message"] == "formatted"


# --------------------------------------------------------------------------- #
# Type-only imports documentation
# --------------------------------------------------------------------------- #


def test_message_record_logger_are_type_only(depcheck):
    """Message / Record / Logger are imported by the backend only under
    TYPE_CHECKING. They may or may not be runtime attributes depending on
    loguru version; this test just documents that the *runtime* dependency is
    only `logger`, and that type names (if present) are usable as annotations.
    """
    mod = depcheck.load(IMPORT_NAME)
    # The one guaranteed runtime export:
    assert hasattr(mod, "logger")
    # Logger is the type of the singleton regardless of export.
    assert type(mod.logger).__name__ == "Logger"
