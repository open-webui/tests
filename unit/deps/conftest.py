"""Shared machinery for the per-dependency contract tests (`unit/deps/`).

Every `test_<dep>.py` in this package uses the `depcheck` fixture (and,
where it needs to scan how Open WebUI uses the dependency, the
`open_webui_backend` fixture from `unit/conftest.py`). Nothing is imported
across test files — the shared surface is exposed purely as fixtures so
dozens of independently-authored modules compose without import-path or
collision issues.
"""

from __future__ import annotations

import importlib
import inspect
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any

import pytest


class DepCheck:
    """Helper handed to every per-dependency test via the `depcheck` fixture.

    Conventions:
      - `load()` SKIPS (not fails) when the package isn't importable, so the
        suite stays runnable in any environment; the test only does real
        work where the dependency is actually installed.
      - symbol checks resolve dotted paths against a module object, so they
        work for nested attributes (`x.y.Z`) and submodules alike.
    """

    def load(self, import_name: str):
        """Import and return a module, or skip the test if unavailable.

        `import_name` is the *import* name (e.g. ``PIL``, ``jwt``, ``bs4``),
        which often differs from the PyPI/distribution name.
        """
        try:
            return importlib.import_module(import_name)
        except Exception as e:  # ImportError or any import-time failure
            pytest.skip(f"{import_name!r} not importable in this env: {e}")

    def try_load(self, import_name: str):
        """Import and return a module, or None (no skip) — for optional probes."""
        try:
            return importlib.import_module(import_name)
        except Exception:
            return None

    def has(self, obj: Any, dotted: str) -> bool:
        """True if the dotted attribute path resolves against `obj`."""
        cur = obj
        for part in dotted.split("."):
            try:
                cur = getattr(cur, part)
            except AttributeError:
                # Might be an unimported submodule — try importing it.
                modname = f"{getattr(cur, '__name__', '')}.{part}".lstrip(".")
                try:
                    cur = importlib.import_module(modname)
                except Exception:
                    return False
        return True

    def resolve(self, obj: Any, dotted: str) -> Any:
        """Return the object at the dotted path, importing submodules as needed."""
        cur = obj
        for part in dotted.split("."):
            try:
                cur = getattr(cur, part)
            except AttributeError:
                modname = f"{getattr(cur, '__name__', '')}.{part}".lstrip(".")
                cur = importlib.import_module(modname)
        return cur

    def assert_symbols(self, obj: Any, names: list[str]) -> None:
        """Assert every dotted name resolves; report all misses at once."""
        missing = [n for n in names if not self.has(obj, n)]
        root = getattr(obj, "__name__", obj)
        assert not missing, (
            f"{root}: missing symbol(s) the Open WebUI codebase relies on: "
            f"{missing}. A dependency bump likely removed/renamed them."
        )

    def assert_callable(self, obj: Any, dotted: str) -> None:
        target = self.resolve(obj, dotted)
        assert callable(target), f"{dotted} is not callable (got {type(target)!r})"

    def assert_params(self, func: Any, names: list[str]) -> None:
        """Assert a callable accepts the given parameter names (or **kwargs)."""
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            pytest.skip(f"no introspectable signature for {func!r}")
        params = sig.parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw:
            return
        missing = [n for n in names if n not in params]
        assert not missing, (
            f"{getattr(func, '__name__', func)} no longer accepts parameter(s): "
            f"{missing} (signature: {sig})"
        )

    def dist_version(self, dist_name: str) -> str | None:
        """Installed distribution version (PyPI name), or None."""
        try:
            return _dist_version(dist_name)
        except PackageNotFoundError:
            return None


@pytest.fixture(scope="session")
def depcheck() -> DepCheck:
    return DepCheck()
