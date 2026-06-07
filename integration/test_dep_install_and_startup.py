"""Dependency-bump gate: resolution + backend startup + browser visibility.

Automated form of the dep-bump workflow's Step 3 gate (`memory/workflow.md`).
Editing version pins is not verification — a bump is only validated once the
new versions actually install together AND the app boots on them. Three layers,
each skips cleanly unless its prerequisite is present so the normal suite stays
fast:

  1. resolution — the bumped backend/requirements.txt installs together
     (uv/pip dry-run). Heavy + network, so gated behind OPEN_WEBUI_DEP_GATE=1.
  2. startup — a backend boots and /health returns 200. Against a running
     instance ($OPEN_WEBUI_URL), or, under the gate, a subprocess uvicorn
     (optionally an isolated interpreter via $OPEN_WEBUI_TEST_PYTHON).
  3. browser visibility (dev) — covered by e2e/test_page_accessibility.py
     loading pages against $OPEN_WEBUI_URL and asserting no console errors.

Run the full gate during a dep bump:
    OPEN_WEBUI_DEP_GATE=1 pytest integration/test_dep_install_and_startup.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

GATE_ENABLED = os.getenv("OPEN_WEBUI_DEP_GATE") == "1"
_gate = pytest.mark.skipif(
    not GATE_ENABLED,
    reason="dep-bump gate is opt-in; set OPEN_WEBUI_DEP_GATE=1 to run it",
)


def _resolve_backend() -> Path | None:
    """Locate `.../open-webui/backend` via $OPEN_WEBUI_SOURCE_DIR or a sibling walk."""
    env = os.getenv("OPEN_WEBUI_SOURCE_DIR")
    if env:
        p = Path(env).expanduser()
        return p if (p / "requirements.txt").is_file() else None
    for parent in Path(__file__).resolve().parents:
        cand = parent / "open-webui" / "backend"
        if (cand / "requirements.txt").is_file():
            return cand
    return None


# -----------------------------------------------------------------------------
# 1. Resolution — the bumped pin set installs together
# -----------------------------------------------------------------------------


@pytest.mark.slow
@_gate
def test_requirements_resolve() -> None:
    """The bumped backend/requirements.txt must resolve into a consistent,
    installable set. A conflict here = a bumped pin clashes with another
    package's constraint; bisect to the culprit.
    """
    backend = _resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    req = backend / "requirements.txt"

    uv = shutil.which("uv")
    if uv:
        # uv resolves the whole graph without installing; fast + offline-ish.
        proc = subprocess.run(
            [uv, "pip", "compile", "--quiet", str(req), "-o", os.devnull],
            capture_output=True,
            text=True,
            timeout=600,
        )
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "-r", str(req)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    assert proc.returncode == 0, (
        "Bumped requirements failed to resolve:\n"
        f"--- stderr (tail) ---\n{proc.stderr[-3000:]}\n"
        f"--- stdout (tail) ---\n{proc.stdout[-1500:]}"
    )


# -----------------------------------------------------------------------------
# 2. Startup — the app boots and /health is green
# -----------------------------------------------------------------------------


@pytest.mark.api
@pytest.mark.requires_instance
def test_running_instance_health(config) -> None:
    """If an Open WebUI is running at $OPEN_WEBUI_URL, /health must return 200.
    Lightweight liveness check usable against a dev server."""
    url = f"{config.base_url}/health"
    try:
        resp = httpx.get(url, timeout=10.0)
    except httpx.HTTPError as e:
        pytest.skip(f"no running Open WebUI at {config.base_url}: {e}")
    assert resp.status_code == 200, f"/health returned HTTP {resp.status_code}"


@pytest.mark.slow
@_gate
def test_backend_boots_subprocess(tmp_path: Path) -> None:
    """Boot the backend in a subprocess against a throwaway sqlite DB and poll
    /health until it comes up. Validates the app imports and serves on the
    installed dependency set.

    Point $OPEN_WEBUI_TEST_PYTHON at an interpreter with the BUMPED versions
    installed to make this the real bump gate; otherwise it uses this env's
    interpreter (validates code health).
    """
    backend = _resolve_backend()
    if backend is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")

    py = os.getenv("OPEN_WEBUI_TEST_PYTHON", sys.executable)
    port = "8615"
    env = {
        **os.environ,
        "WEBUI_SECRET_KEY": "dep-gate-test",
        "DATA_DIR": str(tmp_path),
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'webui.db').as_posix()}",
        "OFFLINE_MODE": "true",
        "RAG_EMBEDDING_ENGINE": "",  # don't pull a model during boot
        "ENABLE_OLLAMA_API": "false",
        "ENABLE_OPENAI_API": "false",
    }
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "open_webui.main:app", "--host", "127.0.0.1", "--port", port],
        cwd=str(backend),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 120
        last_err = None
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"backend exited early (code {proc.returncode}):\n{out[-3000:]}")
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
                if r.status_code == 200:
                    return  # booted cleanly
                last_err = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                last_err = str(e)
            time.sleep(2)
        pytest.fail(f"backend did not become healthy within 120s (last: {last_err})")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
