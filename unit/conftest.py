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
import tempfile
import types
from pathlib import Path

import pytest

# open_webui.env raises SystemExit on import if WEBUI_SECRET_KEY is unset
# (hard requirement when auth is enabled). The supported launchers set it;
# these source-level tests import env.py directly, so set a throwaway value
# before any open_webui import. Respects a real value if the caller set one.
os.environ.setdefault("WEBUI_SECRET_KEY", "test-secret-key-for-unit-tests")

# Importing open_webui.config creates DATA_DIR and deletes tracked files under
# STATIC_DIR. Unset, both resolve inside the open-webui checkout itself, so a
# bare `pytest` run mutates the very source tree under test.
_SCRATCH = Path(tempfile.gettempdir()) / "owui-unit-tests"
os.environ.setdefault("DATA_DIR", str(_SCRATCH / "data"))
os.environ.setdefault("STATIC_DIR", str(_SCRATCH / "static"))
for _path in (os.environ["DATA_DIR"], os.environ["STATIC_DIR"]):
    Path(_path).mkdir(parents=True, exist_ok=True)


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
        def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    sub.Document = _Document
    sys.modules["langchain_core.documents"] = sub


@pytest.fixture(scope="session")
def owui_module(open_webui_backend: Path):
    """Generic loader: `owui_module("open_webui.routers.folders")`.

    Lets a test pull in whatever backend module it needs without every new
    target growing its own fixture. Never evicts from sys.modules: re-executing
    a backend module hands the test a second module object whose globals are not
    the ones the routers hold, so patches silently miss, and re-executing one
    that declares ORM tables raises "Table is already defined".
    """
    if str(open_webui_backend) not in sys.path:
        sys.path.insert(0, str(open_webui_backend))

    def _load(dotted_name: str):
        try:
            return importlib.import_module(dotted_name)
        except ModuleNotFoundError as e:
            # Only a genuinely absent target is an environment problem worth
            # skipping for. Any other import error is real breakage, and
            # swallowing it turns a broken backend into a green run.
            if e.name != dotted_name:
                raise
            pytest.skip(f"{dotted_name} not present in this checkout")

    return _load


@pytest.fixture(scope="session")
def terminals_router_module(owui_module):
    """`open_webui.routers.terminals` (_sanitize_proxy_path)."""
    return owui_module("open_webui.routers.terminals")


@pytest.fixture(scope="session")
def access_control_module(owui_module):
    """`open_webui.utils.access_control` (has_base_model_access etc.)."""
    return owui_module("open_webui.utils.access_control")


@pytest.fixture(scope="session")
def firecrawl_module(owui_module):
    """`open_webui.retrieval.web.firecrawl`.

    Session-scoped: each test patches its own targets within a `with`
    block, so the module object can be safely shared.
    """
    _install_langchain_document_stub()
    return owui_module("open_webui.retrieval.web.firecrawl")


@pytest.fixture(scope="session")
def misc_module(owui_module):
    """`open_webui.utils.misc`.

    Lightweight: only pulls aiohttp, mimeparse, and open_webui.env —
    all present in the openwebui-venv.
    """
    return owui_module("open_webui.utils.misc")


@pytest.fixture(scope="session")
def web_search_main_module(owui_module):
    """`open_webui.retrieval.web.main` (get_filtered_results)."""
    return owui_module("open_webui.retrieval.web.main")


@pytest.fixture(scope="session")
def automations_module(owui_module):
    """`open_webui.utils.automations` (_resolve_model_features)."""
    return owui_module("open_webui.utils.automations")


@pytest.fixture(scope="session")
def config_model_module(owui_module):
    """`open_webui.models.config` (the per-key Config store)."""
    # Importing open_webui.config runs the alembic migrations (creating the
    # `config` table), so Config.upsert/get have a schema to hit.
    owui_module("open_webui.config")
    return owui_module("open_webui.models.config")


@pytest.fixture(scope="session")
def builtin_tools_module(owui_module):
    """`open_webui.tools.builtin`.

    Pulls the open_webui model layer (Notes/Chats/Knowledges/calendar)
    and triggers alembic setup on first load.
    """
    return owui_module("open_webui.tools.builtin")


@pytest.fixture(scope="session")
def retrieval_utils_module(owui_module):
    """`open_webui.retrieval.utils`.

    Heavy: pulls langchain, huggingface_hub, the vector-DB clients, the
    whole open_webui model layer, and triggers alembic setup on first load.
    """
    return owui_module("open_webui.retrieval.utils")


@pytest.fixture(scope="session")
def retrieval_web_utils_module(owui_module):
    """`open_webui.retrieval.web.utils`.

    Heavier than firecrawl — pulls in langchain_community, aiohttp,
    fastapi, the whole open_webui.config tree, and triggers alembic
    migration setup on first load.
    """
    return owui_module("open_webui.retrieval.web.utils")
