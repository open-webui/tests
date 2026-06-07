"""Dependency contract: black.

``black`` is the Python code formatter declared as a dependency of the
Open WebUI backend (``black==26.3.1``). It is used to format model/tool
code and (historically) Python served through the UI; it is invoked
through its public API (``black.format_str`` and friends) rather than only
the CLI. A breaking bump (renamed ``format_str``, changed the keyword-only
``mode`` parameter, moved ``Mode``/``FileMode``, or altered the
``NothingChanged``/``InvalidInput`` control-flow exceptions) would break
any in-process formatting call.

This module pins the public formatting surface and its keyword-only call
shape, then exercises real formatting offline: reformat messy source,
confirm idempotence, confirm the line-length ``Mode`` knob takes effect,
and confirm the two control-flow exceptions fire on the inputs that
trigger them. Pure in-memory string work — no files, no network.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "black"
DIST_NAME = "black"

TOP_LEVEL_SYMBOLS = [
    "format_str",  # format a source string
    "format_file_contents",  # format file text (raises NothingChanged)
    "format_file_in_place",  # in-place file formatting
    "Mode",  # formatting options object
    "FileMode",  # legacy alias of Mode
    "InvalidInput",  # raised on unparseable source
    "NothingChanged",  # raised when input is already formatted
    "TargetVersion",  # target python-version enum
    "WriteBack",  # write-back strategy enum
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`black` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "black"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature checks.
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_format_str_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.format_str)


def test_format_str_signature(depcheck):
    """format_str(src_contents, *, mode) — `mode` is keyword-only. The src
    positional and the mode keyword must both remain."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.format_str)
    params = sig.parameters
    assert "mode" in params, "black.format_str dropped the `mode` parameter"
    assert params["mode"].kind is inspect.Parameter.KEYWORD_ONLY, (
        "black.format_str `mode` is no longer keyword-only"
    )
    # First positional is the source contents.
    positional = [
        n
        for n, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, "black.format_str lost its source positional parameter"


def test_format_file_contents_signature(depcheck):
    """format_file_contents(src, *, fast, mode) — pin the fast + mode kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.format_file_contents, ["src_contents", "fast", "mode"])


def test_mode_aliases_filemode(depcheck):
    """`FileMode` has historically been an alias of `Mode`; callers use either."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.FileMode is mod.Mode


def test_control_flow_exceptions_are_exceptions(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.InvalidInput, Exception)
    assert issubclass(mod.NothingChanged, Exception)


def test_mode_line_length_knob(depcheck):
    """Mode(line_length=N) is the primary formatting knob; pin that it is
    accepted and stored (default is 88)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.Mode().line_length == 88
    assert mod.Mode(line_length=40).line_length == 40


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — real formatting semantics.
# ---------------------------------------------------------------------------


def test_behaviour_format_str_reformats_messy_code(depcheck):
    """The core contract: format_str normalises spacing/quotes/layout. Feed
    deliberately messy source and assert it is cleaned up."""
    mod = depcheck.load(IMPORT_NAME)
    messy = "x = {  1:2,3:4  }\ndef f( a,b ):\n  return  a+b\n"
    out = mod.format_str(messy, mode=mod.Mode())
    assert "{1: 2, 3: 4}" in out
    assert "def f(a, b):" in out
    assert "return a + b" in out
    # Output ends with a single trailing newline (black invariant).
    assert out.endswith("\n")


def test_behaviour_format_is_idempotent(depcheck):
    """Formatting already-formatted code must be a no-op — re-running black on
    its own output yields identical text (the stability guarantee)."""
    mod = depcheck.load(IMPORT_NAME)
    src = "a = [1, 2, 3]\n"
    once = mod.format_str(src, mode=mod.Mode())
    twice = mod.format_str(once, mode=mod.Mode())
    assert once == twice


def test_behaviour_line_length_affects_wrapping(depcheck):
    """A small line_length must force black to wrap a call that fits at the
    default width — proving the Mode knob actually changes output."""
    mod = depcheck.load(IMPORT_NAME)
    src = "result = some_function(argument_one, argument_two, argument_three, four)\n"
    wide = mod.format_str(src, mode=mod.Mode(line_length=200))
    narrow = mod.format_str(src, mode=mod.Mode(line_length=20))
    # At a wide width it stays on one line; at width 20 it must wrap.
    assert "\n" in narrow.rstrip("\n"), "narrow line_length did not wrap"
    assert narrow != wide


def test_behaviour_nothing_changed_on_formatted_input(depcheck):
    """format_file_contents raises NothingChanged when the input already matches
    black's output — callers use this exception as control flow."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.NothingChanged):
        mod.format_file_contents("x = 1\n", fast=True, mode=mod.Mode())


def test_behaviour_invalid_input_raises(depcheck):
    """Unparseable source must raise InvalidInput, not return garbage or crash —
    so a formatter caller can surface a clean error to the user."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.InvalidInput):
        mod.format_str("def (:\n", mode=mod.Mode())


def test_behaviour_normalises_quotes(depcheck):
    """black normalises single quotes to double quotes by default — a visible,
    stable transformation we can pin."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.format_str("s = 'hello'\n", mode=mod.Mode())
    assert out == 's = "hello"\n'
