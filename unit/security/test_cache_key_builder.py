"""Regression guard: aiocache @cached must not use `key=` for per-call keys.

In aiocache 0.12 (`aiocache==0.12.3`), `cached.get_cache_key` does
`if self.key: return self.key` — it returns the `key=` value verbatim
WITHOUT calling it. So `@cached(key=lambda ...)` is a static key: the
lambda object becomes one constant cache key and every caller collides to
a single entry within the TTL. The per-call hook is `key_builder=`
(invoked as `key_builder(func, *args, **kwargs)`).

routers/openai.py and routers/ollama.py shipped
`@cached(key=lambda _, user: f'..._{user.id}' ...)` on get_all_models,
which meant the intended per-user namespacing never happened — one user's
permission-filtered model list could be served to another user (or an
anonymous caller) during the cache window. A cross-user data-exposure /
cache-poisoning bug. Fixed by switching to `key_builder=`
(branch fix/cached-key-builder-per-user-models).

This audit fails if ANY `@cached(...)` in the backend passes a lambda to
`key=` (the static-key footgun), and specifically checks the two model
caches use `key_builder`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CACHED = re.compile(r"@cached\s*\(")
# `key=` set to a callable (lambda or a bare name) — the static-key bug.
_KEY_CALLABLE = re.compile(r"(?<![A-Za-z0-9_])key\s*=\s*lambda\b")


def _decorator_call(src: str, at: int) -> str:
    """Return the balanced `(...)` of a `@cached(` starting near index `at`,
    with `# ...` comments stripped so the audit analyses code, not prose
    (a comment that mentions `key=lambda` must not trip the check)."""
    open_paren = src.index("(", at)
    depth = 0
    for j in range(open_paren, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                call = src[open_paren : j + 1]
                return re.sub(r"#[^\n]*", "", call)
    raise AssertionError("unbalanced parentheses in @cached(...)")


def _backend_py_files(open_webui_backend: Path):
    return sorted((open_webui_backend / "open_webui").rglob("*.py"))


@pytest.mark.regression
def test_no_cached_decorator_uses_static_key_lambda(open_webui_backend: Path) -> None:
    """No @cached(...) may pass a lambda to `key=` — that's a static key in
    aiocache 0.12 and collapses every caller to one shared entry. Per-call
    keys must use `key_builder=`."""
    offenders: list[str] = []
    for py in _backend_py_files(open_webui_backend):
        src = py.read_text(encoding="utf-8")
        for m in _CACHED.finditer(src):
            call = _decorator_call(src, m.start())
            if _KEY_CALLABLE.search(call):
                offenders.append(py.relative_to(open_webui_backend).as_posix())
    assert not offenders, (
        "Regression: @cached(key=lambda ...) is a STATIC key in aiocache 0.12 "
        "(get_cache_key returns it verbatim, never calls it), so all callers "
        "share one cache entry — a cross-user exposure risk. Use key_builder= "
        f"instead. Offending file(s): {sorted(set(offenders))}"
    )


@pytest.mark.regression
def test_model_caches_use_key_builder(open_webui_backend: Path) -> None:
    """The per-user model caches must namespace by user via key_builder."""
    for rel in ("open_webui/routers/openai.py", "open_webui/routers/ollama.py"):
        src = (open_webui_backend / rel).read_text(encoding="utf-8")
        # locate the @cached on get_all_models
        idx = src.find("@cached")
        assert idx != -1, f"{rel}: no @cached decorator found"
        call = _decorator_call(src, idx)
        assert "key_builder" in call, (
            f"{rel}: get_all_models @cached must use key_builder= (per-user key), "
            f"not a static key= — else one user's model list leaks to others."
        )
        assert not _KEY_CALLABLE.search(call), (
            f"{rel}: still passes a lambda to key= (static-key bug)."
        )
