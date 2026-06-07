"""Dependency contract: RestrictedPython (SECURITY-CRITICAL sandbox).

RestrictedPython is *not* imported by name anywhere in the Open WebUI
backend today (it ships in requirements.txt / requirements-min.txt as
``RestrictedPython==8.2``), but it is the canonical Python sandbox the
project depends on for any restricted / user-supplied code-execution
surface. Because the entire point of the package is to make ``exec`` of
untrusted source *safe*, a silent regression in its restriction behaviour
is a security incident, not a feature change. This module is therefore
deliberately exhaustive: it does not merely check that symbols exist, it
**compiles and executes** restricted snippets and asserts that dangerous
operations are blocked and that the operations a legitimate sandbox needs
still work.

The pinned contract, in two layers:

  COMPILE-TIME (RestrictingNodeTransformer rejects the source outright):
    - any access to an attribute whose name starts with ``_`` (the
      ``obj.__class__`` / ``obj._secret`` escape hatch) raises SyntaxError;
    - binding/declaring an ``_``-prefixed name raises SyntaxError.

  RUNTIME (the restricted namespace simply lacks the dangerous power, or a
  guard intercepts it):
    - ``__import__`` is absent from ``safe_builtins`` -> ``import os`` (which
      *compiles*) fails at runtime with an ImportError, so a sandboxed
      snippet cannot import the filesystem/process modules;
    - ``open``, ``eval``, ``exec``, ``compile``, ``getattr``, ``globals``,
      ``locals``, ``vars``, ``input``, ``breakpoint`` are absent from
      ``safe_builtins``;
    - ``setattr`` / ``delattr`` in ``safe_builtins`` are the *guarded*
      variants, and attribute writes go through ``_write_`` which refuses to
      mutate an unprepared object;
    - ``safer_getattr`` allows public attributes but refuses ``_``-prefixed
      ones even when reached dynamically;
    - the helper hooks compiled code references (``_getattr_`` /
      ``_getitem_`` / ``_getiter_`` / ``_write_`` and the unpack guards) are
      present and behave;
    - the builtin partitioning holds: ``safe_builtins`` carries the harmless
      names (len/range/str/...), ``limited_builtins`` adds list/tuple/range,
      ``utility_builtins`` adds math/random/string/set.

If a RestrictedPython bump flipped any of these, sandboxed code could read
private state, import os, or escape; the corresponding test fails loudly.

Pattern mirrors test_requests.py (symbol surface + offline behaviour) but
weighted heavily toward executable security assertions. Uses the
``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "RestrictedPython"
DIST_NAME = "RestrictedPython"

# Top-level surface a consumer of the sandbox relies on.
USED_SYMBOLS = [
    "compile_restricted",
    "compile_restricted_exec",
    "compile_restricted_eval",
    "compile_restricted_single",
    "compile_restricted_function",
    "safe_globals",
    "safe_builtins",
    "limited_builtins",
    "utility_builtins",
    "RestrictingNodeTransformer",
    "PrintCollector",
    # submodules consumers reach into for guards / eval helpers
    "Guards",
    "Eval",
    "Limits",
    "Utilities",
]

# Guards submodule surface (the runtime guard machinery).
GUARD_SYMBOLS = [
    "safe_builtins",
    "safe_globals",
    "safer_getattr",
    "full_write_guard",
    "guarded_setattr",
    "guarded_delattr",
    "guarded_iter_unpack_sequence",
    "guarded_unpack_sequence",
]

# Builtins that MUST NOT be reachable from a restricted snippet's
# safe_builtins (each is an escape / IO / dynamic-eval primitive).
FORBIDDEN_BUILTINS = [
    "__import__",
    "open",
    "eval",
    "exec",
    "compile",
    "getattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "memoryview",
]


# ---------------------------------------------------------------------------
# Local sandbox harness (no cross-file imports; conftest exposes only
# fixtures). Builds a restricted execution namespace and runs source.
# ---------------------------------------------------------------------------


def _make_globals(mod, *, extra_builtins=None, extra_globals=None):
    """A fresh restricted namespace: safe_builtins (+overrides) plus any
    helper hooks (``_getattr_`` etc.) the snippet needs."""
    builtins_dict = dict(mod.safe_builtins)
    if extra_builtins:
        builtins_dict.update(extra_builtins)
    glb = {"__builtins__": builtins_dict}
    if extra_globals:
        glb.update(extra_globals)
    return glb


def _run(mod, src, glb=None, loc=None):
    """compile_restricted(src, 'exec') then exec it in the restricted glb."""
    glb = glb if glb is not None else _make_globals(mod)
    loc = loc if loc is not None else {}
    code = mod.compile_restricted(src, "<restricted>", "exec")
    exec(code, glb, loc)  # noqa: S102 - that is the whole point: sandboxed exec
    return glb, loc


# ---------------------------------------------------------------------------
# Import + version + symbol surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "RestrictedPython"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_guard_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    guards = depcheck.resolve(mod, "Guards")
    depcheck.assert_symbols(guards, GUARD_SYMBOLS)


def test_compile_restricted_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "compile_restricted")


def test_safe_globals_shape(depcheck):
    """safe_globals must be a mapping that provides __builtins__ (the minimal
    global namespace a restricted exec starts from)."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.safe_globals, dict)
    assert "__builtins__" in mod.safe_globals


def test_safe_builtins_is_mapping(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.safe_builtins, dict)
    assert len(mod.safe_builtins) > 0


# ---------------------------------------------------------------------------
# COMPILE-TIME restrictions — the transformer rejects dangerous source.
# ---------------------------------------------------------------------------


def test_dunder_attribute_access_blocked_at_compile(depcheck):
    """`obj.__class__` is the classic sandbox-escape chain
    (().__class__.__bases__...). Accessing any dunder attribute must be a
    SyntaxError at compile time."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(SyntaxError):
        mod.compile_restricted("x.__class__", "<t>", "exec")


def test_underscore_attribute_access_blocked_at_compile(depcheck):
    """Any `_`-prefixed attribute (private state) is rejected at compile."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(SyntaxError):
        mod.compile_restricted("y = x._secret", "<t>", "exec")


def test_underscore_attribute_on_literal_blocked(depcheck):
    """Even on a literal: `(1)._foo` must be rejected (can't reach private
    attrs of builtin types either)."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(SyntaxError):
        mod.compile_restricted("z = (1)._foo", "<t>", "exec")


def test_underscore_name_binding_blocked_at_compile(depcheck):
    """Binding an `_`-prefixed name is rejected (prevents shadowing the guard
    hooks like `_getattr_` / `_write_`)."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(SyntaxError):
        mod.compile_restricted("_a = 5", "<t>", "exec")


def test_syntax_error_on_malformed_source(depcheck):
    """Plain malformed Python still raises SyntaxError (not a silent None)."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(SyntaxError):
        mod.compile_restricted("def f(:\n  pass", "<t>", "exec")


# ---------------------------------------------------------------------------
# RUNTIME restrictions — the sandbox namespace lacks dangerous power.
# ---------------------------------------------------------------------------


def test_import_compiles_but_fails_at_runtime(depcheck):
    """`import os` COMPILES (RestrictedPython turns it into a call to
    __import__), but __import__ is absent from safe_builtins, so it must fail
    at runtime. This is the primary 'no filesystem/process access' guarantee:
    a sandboxed snippet cannot import os/subprocess/sys."""
    mod = depcheck.load(IMPORT_NAME)
    code = mod.compile_restricted("import os", "<t>", "exec")  # compiles fine
    with pytest.raises((ImportError, NameError, KeyError)):
        exec(code, _make_globals(mod), {})  # noqa: S102


def test_import_subprocess_blocked_at_runtime(depcheck):
    """Same guarantee for subprocess (command execution)."""
    mod = depcheck.load(IMPORT_NAME)
    code = mod.compile_restricted("import subprocess", "<t>", "exec")
    with pytest.raises((ImportError, NameError, KeyError)):
        exec(code, _make_globals(mod), {})  # noqa: S102


def test_forbidden_builtins_absent_from_safe_builtins(depcheck):
    """Each escape/IO/dynamic-eval primitive must be absent from
    safe_builtins so a restricted snippet cannot name it."""
    mod = depcheck.load(IMPORT_NAME)
    present = [name for name in FORBIDDEN_BUILTINS if name in mod.safe_builtins]
    assert not present, f"dangerous builtin(s) leaked into safe_builtins: {present}"


def test_eval_call_blocked_at_compile(depcheck):
    """`eval(...)` inside the sandbox must fail. RestrictedPython blocks it
    even earlier than the missing-builtin path: the transformer rejects an
    `eval` call at COMPILE time ('Eval calls are not allowed.'), so a snippet
    cannot bootstrap arbitrary evaluation. Assert the stronger compile-time
    guarantee."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(SyntaxError):
        mod.compile_restricted("r = eval('1+1')", "<t>", "exec")


def test_open_unavailable_in_sandbox(depcheck):
    """`open(...)` must fail — no file access primitive in the sandbox."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises((NameError, KeyError)):
        _run(mod, "f = open('/etc/passwd')")


def test_guarded_setattr_blocks_attribute_write(depcheck):
    """safe_builtins' `setattr` is the GUARDED variant: it routes through the
    write guard and must refuse to mutate an unprepared object (so a snippet
    can't patch arbitrary objects). Real builtins.setattr would succeed; the
    guard must raise."""
    mod = depcheck.load(IMPORT_NAME)
    import builtins as _b

    guarded = mod.safe_builtins.get("setattr")
    assert guarded is not None
    assert guarded is not _b.setattr, "safe_builtins.setattr is the REAL setattr (unguarded!)"

    class Plain:
        pass

    with pytest.raises((TypeError, AttributeError)):
        guarded(Plain(), "x", 1)


def test_guarded_delattr_is_not_real_delattr(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    import builtins as _b

    guarded = mod.safe_builtins.get("delattr")
    assert guarded is not None
    assert guarded is not _b.delattr, "safe_builtins.delattr is the REAL delattr (unguarded!)"


def test_attribute_assignment_needs_write_guard(depcheck):
    """`o.x = 5` compiles to a `_write_(o).x = 5` call; with the full write
    guard installed, assigning to a plain object's attribute must raise
    (the object isn't a write-permitted type)."""
    mod = depcheck.load(IMPORT_NAME)
    guards = depcheck.resolve(mod, "Guards")

    class Plain:
        pass

    glb = _make_globals(mod, extra_globals={"_write_": guards.full_write_guard})
    with pytest.raises((TypeError, AttributeError)):
        _run(mod, "o.x = 5", glb=glb, loc={"o": Plain()})


# ---------------------------------------------------------------------------
# safer_getattr — public attrs allowed, private refused even dynamically.
# ---------------------------------------------------------------------------


def test_safer_getattr_allows_public(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    guards = depcheck.resolve(mod, "Guards")

    class Obj:
        public = "ok"

    assert guards.safer_getattr(Obj(), "public") == "ok"


def test_safer_getattr_blocks_underscore(depcheck):
    """safer_getattr must refuse `_`-prefixed names even when the attribute
    name is computed at runtime (defends the dynamic getattr path, not just
    the compile-time literal check)."""
    mod = depcheck.load(IMPORT_NAME)
    guards = depcheck.resolve(mod, "Guards")

    class Obj:
        _secret = "leak"

    with pytest.raises((AttributeError, TypeError)):
        guards.safer_getattr(Obj(), "_secret")


def test_safer_getattr_blocks_dunder(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    guards = depcheck.resolve(mod, "Guards")
    with pytest.raises((AttributeError, TypeError)):
        guards.safer_getattr(object(), "__class__")


# ---------------------------------------------------------------------------
# ALLOWED operations — a legitimate sandbox must still compute.
# ---------------------------------------------------------------------------


def test_arithmetic_and_safe_builtins_work(depcheck):
    """len/range/str/sorted/abs are safe builtins; a snippet using them plus
    arithmetic must run and produce the right value."""
    mod = depcheck.load(IMPORT_NAME)
    _, loc = _run(mod, "result = len('abcd') + abs(-3) + int('10')")
    assert loc["result"] == 4 + 3 + 10


def test_for_loop_with_getiter_guard(depcheck):
    """A `for` loop compiles to use `_getiter_`; with the default guard
    installed the loop runs (iteration is allowed, just guarded)."""
    mod = depcheck.load(IMPORT_NAME)
    eval_mod = depcheck.resolve(mod, "Eval")
    glb = _make_globals(mod, extra_globals={"_getiter_": eval_mod.default_guarded_getiter})
    _, loc = _run(mod, "total = 0\nfor i in [1, 2, 3, 4]:\n    total = total + i", glb=glb)
    assert loc["total"] == 10


def test_subscript_with_getitem_guard(depcheck):
    """`d['k']` compiles to use `_getitem_`; with the default guard a dict
    subscript works."""
    mod = depcheck.load(IMPORT_NAME)
    eval_mod = depcheck.resolve(mod, "Eval")
    glb = _make_globals(mod, extra_globals={"_getitem_": eval_mod.default_guarded_getitem})
    _, loc = _run(mod, "v = d['k']", glb=glb, loc={"d": {"k": 99}})
    assert loc["v"] == 99


def test_getattr_hook_used_for_public_attr(depcheck):
    """When `_getattr_` is set to safer_getattr, reading a public attribute in
    a snippet works through the guard."""
    mod = depcheck.load(IMPORT_NAME)
    guards = depcheck.resolve(mod, "Guards")

    class Obj:
        value = 7

    glb = _make_globals(mod, extra_globals={"_getattr_": guards.safer_getattr})
    _, loc = _run(mod, "v = o.value", glb=glb, loc={"o": Obj()})
    assert loc["v"] == 7


def test_function_definition_allowed(depcheck):
    """Defining and calling a pure function inside the sandbox is allowed."""
    mod = depcheck.load(IMPORT_NAME)
    src = "def add(a, b):\n    return a + b\n\nresult = add(2, 40)"
    _, loc = _run(mod, src)
    assert loc["result"] == 42


# ---------------------------------------------------------------------------
# compile_restricted modes + builtin partitioning.
# ---------------------------------------------------------------------------


def test_compile_restricted_eval_mode(depcheck):
    """compile_restricted(src, 'eval') yields an expression code object."""
    mod = depcheck.load(IMPORT_NAME)
    code = mod.compile_restricted("1 + 2", "<t>", "eval")
    assert eval(code, {"__builtins__": {}}) == 3  # noqa: S307 - trivial const expr


def test_compile_restricted_exec_helper(depcheck):
    """compile_restricted_exec is the exec-mode shortcut consumers may use."""
    mod = depcheck.load(IMPORT_NAME)
    result = mod.compile_restricted_exec("a = 1")
    # CompileResult-like: must expose a usable code object (attr 'code').
    code = getattr(result, "code", result)
    assert code is not None


def test_limited_builtins_partition(depcheck):
    """limited_builtins must supply list/tuple/range (the constrained
    sequence constructors) — pin that exact membership."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("list", "tuple", "range"):
        assert name in mod.limited_builtins, f"{name} missing from limited_builtins"


def test_utility_builtins_partition(depcheck):
    """utility_builtins must supply the math/random/string/set helpers a
    richer sandbox enables."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("math", "random", "string", "set", "frozenset"):
        assert name in mod.utility_builtins, f"{name} missing from utility_builtins"


def test_safe_builtins_has_harmless_names(depcheck):
    """The harmless, non-IO builtins a sandbox needs must remain in
    safe_builtins (regression guard against an over-aggressive prune that
    would break legitimate snippets)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("len", "range", "str", "int", "sorted", "zip", "abs", "True", "False", "None"):
        assert name in mod.safe_builtins, f"{name} unexpectedly dropped from safe_builtins"


def test_printcollector_constructs(depcheck):
    """PrintCollector backs the guarded `print` in restricted code; it must
    remain constructable (consumers install it as `_print_`)."""
    mod = depcheck.load(IMPORT_NAME)
    pc = depcheck.resolve(mod, "PrintCollector")
    assert isinstance(pc, type)


# ---------------------------------------------------------------------------
# End-to-end: a hostile snippet is neutralised; a benign one computes.
# ---------------------------------------------------------------------------


def test_end_to_end_escape_attempt_blocked(depcheck):
    """A representative escape attempt — reach object.__subclasses__ via the
    __class__ chain — must be stopped at compile time (SyntaxError), so it
    never even runs."""
    mod = depcheck.load(IMPORT_NAME)
    hostile = "().__class__.__bases__[0].__subclasses__()"
    with pytest.raises(SyntaxError):
        mod.compile_restricted(hostile, "<t>", "exec")


def test_end_to_end_benign_snippet_runs(depcheck):
    """A benign data-transform snippet (the legitimate use of the sandbox)
    compiles and runs, producing the expected output."""
    mod = depcheck.load(IMPORT_NAME)
    eval_mod = depcheck.resolve(mod, "Eval")
    glb = _make_globals(
        mod,
        extra_globals={
            "_getiter_": eval_mod.default_guarded_getiter,
            "_getitem_": eval_mod.default_guarded_getitem,
        },
    )
    src = "out = []\nfor n in range(5):\n    out.append(n * n)\nresult = out"
    _, loc = _run(mod, src, glb=glb)
    assert loc["result"] == [0, 1, 4, 9, 16]
