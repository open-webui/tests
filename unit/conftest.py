"""Shared fixtures for source-level tests against the open-webui backend.

These tests don't talk to a running Open WebUI — they import the backend
Python modules directly from a local checkout. This conftest resolves
where that checkout is and loads the modules we exercise, stubbing only
the third-party dependencies we don't have in this test repo's env.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest


def _resolve_open_webui_backend() -> Path | None:
    env = os.getenv("OPEN_WEBUI_SOURCE_DIR")
    if env:
        candidate = Path(env).expanduser()
        return candidate if candidate.is_dir() else None

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "open-webui" / "backend"
        if (candidate / "open_webui" / "retrieval" / "web" / "utils.py").is_file():
            return candidate
    return None


@pytest.fixture(scope="session")
def open_webui_backend() -> Path:
    """Path to the open-webui backend source directory.

    Honors $OPEN_WEBUI_SOURCE_DIR; otherwise walks up from this file
    looking for `open-webui/backend/open_webui/retrieval/web/utils.py`.
    Skips dependent tests if neither resolves.
    """
    backend = _resolve_open_webui_backend()
    if backend is None:
        pytest.skip(
            "open-webui backend source not found. Set OPEN_WEBUI_SOURCE_DIR "
            "or place this tests repo next to the open-webui checkout."
        )
    return backend


def _install_langchain_document_stub() -> None:
    """Make `langchain_core.documents.Document` importable.

    Prefer the real `langchain_core` if it's installed in the env (the
    openwebui-venv has it), because stubbing the parent package breaks
    other langchain_core.* submodules that open_webui pulls in via
    main.py → utils.py. Only fall back to a stub if neither is
    importable, in which case open_webui itself probably won't import
    cleanly either.
    """
    try:
        import langchain_core.documents  # noqa: F401
        return
    except ImportError:
        pass

    sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
    sub = types.ModuleType("langchain_core.documents")

    class _Document:
        def __init__(
            self, page_content: str = "", metadata: dict | None = None
        ) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    sub.Document = _Document
    sys.modules["langchain_core.documents"] = sub


@pytest.fixture(scope="session")
def firecrawl_module(open_webui_backend: Path):
    """Load `open_webui.retrieval.web.firecrawl` from the local checkout.

    Session-scoped: each test patches its own targets within a `with`
    block, so the module object can be safely shared.
    """
    if str(open_webui_backend) not in sys.path:
        sys.path.insert(0, str(open_webui_backend))
    _install_langchain_document_stub()
    sys.modules.pop("open_webui.retrieval.web.firecrawl", None)
    try:
        return importlib.import_module("open_webui.retrieval.web.firecrawl")
    except Exception as e:
        pytest.skip(f"Could not import open_webui.retrieval.web.firecrawl: {e}")


@pytest.fixture(scope="session")
def retrieval_web_utils_module(open_webui_backend: Path):
    """Load `open_webui.retrieval.web.utils` from the local checkout.

    Heavier than firecrawl — pulls in langchain_community, aiohttp,
    fastapi, the whole open_webui.config tree, and triggers alembic
    migration setup on first load. The openwebui-venv has everything;
    we skip if anything's missing.
    """
    if str(open_webui_backend) not in sys.path:
        sys.path.insert(0, str(open_webui_backend))
    sys.modules.pop("open_webui.retrieval.web.utils", None)
    try:
        return importlib.import_module("open_webui.retrieval.web.utils")
    except Exception as e:
        pytest.skip(f"Could not import open_webui.retrieval.web.utils: {e}")
