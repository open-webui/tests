"""Regression guard: aiocache `@cached` must not use `key=` for per-call keys.

In aiocache 0.12 `cached.get_cache_key` returns the `key=` value verbatim without calling it, so
`@cached(key=lambda ...)` is one constant key and every caller collides into a single entry for
the whole TTL. `routers/openai.py` and `routers/ollama.py` shipped exactly that on
`get_all_models`, so one user's permission-filtered model list could be served to another.
The per-call hook is `key_builder=` (branch fix/cached-key-builder-per-user-models).

The integration twin (integration/security/test_cache_key_builder.py) pins the OpenAI model
cache over HTTP; this `ast` sweep covers every other `@cached` in the backend, the Ollama one
included.

Discriminates: passes on dev bbfa876af; fails with `key_builder=` turned back into `key=` on
either `get_all_models`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression


def _cached_decorators(open_webui_backend: Path):
    """Every `@cached(...)` call in the backend, with the file it sits in."""
    for path in sorted((open_webui_backend / "open_webui").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                callee = decorator.func
                if getattr(callee, "attr", getattr(callee, "id", None)) == "cached":
                    yield path.relative_to(open_webui_backend).as_posix(), node.name, decorator


def test_no_cached_decorator_passes_a_lambda_as_its_key(open_webui_backend):
    decorators = list(_cached_decorators(open_webui_backend))
    assert decorators, "no @cached decorator found in the backend; retarget this sweep"

    static_keys = [
        f"{path}:{function}"
        for path, function, decorator in decorators
        for keyword in decorator.keywords
        if keyword.arg == "key" and isinstance(keyword.value, ast.Lambda)
    ]
    assert static_keys == [], (
        f"@cached(key=lambda ...) is one static key in aiocache 0.12, so every caller shares "
        f"one entry; use key_builder= in {static_keys}"
    )
