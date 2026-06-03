"""Regression test for backend/start.sh — container crashes on startup
when WEB_LOADER_ENGINE is not set in the environment.

In commit 070ab2650 ("refac: reorganize scripts and ci workflows",
2026-05-12), start.sh was rewritten to use `set -euo pipefail` for
strict-mode safety. The rewrite added a new conditional:

    if [[ "${WEB_LOADER_ENGINE,,}" == "playwright" ]]; then

…but the `${VAR,,}` lowercase expansion can't be combined with `:-` in
a single substitution, and the `:-` default was dropped. Every other
env-var reference in start.sh uses `${VAR:-...}`; this one was missed.

Result: any container that doesn't explicitly set `WEB_LOADER_ENGINE`
(i.e. the default for most deployments — anyone not opting into
playwright/firecrawl/tavily/external) crashes on startup with:

    start.sh: line 15: WEB_LOADER_ENGINE: unbound variable

Reported by urbenlegend on open-webui#24560. Fix on
Classic298/open-webui:fix/start-sh-unbound-web-loader-engine.

The test exercises the script up to (but not including) the uvicorn
launch — we want to verify the preamble doesn't crash, not actually
start a server.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_LAUNCH_MARKERS = (
    "# ── Launch uvicorn",
    "exec env",
    "exec ",
)


def _truncate_at_launch(source: str) -> str | None:
    """Strip the actual server launch from start.sh and replace it with
    `exit 0`. Returns None if no recognisable launch marker is present.
    """
    for marker in _LAUNCH_MARKERS:
        idx = source.find(marker)
        if idx != -1:
            return source[:idx] + "exit 0\n"
    return None


@pytest.mark.regression
def test_start_sh_does_not_crash_when_web_loader_engine_unset(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Regression: start.sh aborts under `set -u` when WEB_LOADER_ENGINE
    is unset (the default for most deployments).
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on PATH")

    start_sh = open_webui_backend / "start.sh"
    if not start_sh.is_file():
        pytest.skip(f"start.sh not found at {start_sh}")

    source = start_sh.read_text(encoding="utf-8")

    if "WEB_LOADER_ENGINE" not in source:
        pytest.skip("start.sh no longer references WEB_LOADER_ENGINE")

    stubbed = _truncate_at_launch(source)
    if stubbed is None:
        pytest.skip(
            "Could not locate the uvicorn launch in start.sh — the script "
            "structure changed; update this regression test."
        )

    stub_path = tmp_path / "start.sh"
    stub_path.write_text(stubbed, encoding="utf-8")
    stub_path.chmod(0o755)

    # Strip WEB_LOADER_ENGINE from the env even if the test runner
    # happens to have it set; this is the user-facing scenario.
    env = {k: v for k, v in os.environ.items() if k != "WEB_LOADER_ENGINE"}

    result = subprocess.run(
        [bash, str(stub_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    # The specific symptom from urbenlegend's report.
    bug_signature = "WEB_LOADER_ENGINE: unbound variable"
    if bug_signature in result.stderr:
        pytest.fail(
            "Regression: backend/start.sh references ${WEB_LOADER_ENGINE,,} "
            "without a `:-` default. Under `set -euo pipefail`, this kills "
            "container startup whenever WEB_LOADER_ENGINE is not in the env "
            "(the default for most deployments).\n"
            f"stderr: {result.stderr.strip()}"
        )

    assert result.returncode == 0, (
        f"start.sh preamble exited with {result.returncode}\n"
        f"stderr: {result.stderr.strip()}\n"
        f"stdout: {result.stdout.strip()}"
    )
