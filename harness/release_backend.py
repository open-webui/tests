"""The backend of a published Open WebUI release, for a test to start on data the checkout left.

`release_backend(tag, root)` unpacks the release's backend from the checkout's git repository
under `root`, fetching the tag from GitHub when a shallow checkout lacks it, as `serving(...,
backend=...)` takes it. Its `settings` put the release's pins of the packages it imports at boot
and this environment lacks on its `PYTHONPATH`, never into the environment itself.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from harness.instance import resolve_backend

UPSTREAM_REPOSITORY = "https://github.com/open-webui/open-webui.git"
# only what a release imports at boot; the rest of its requirements load lazily
BOOT_PACKAGES = ("fpdf2", "fonttools")


@dataclass
class ReleaseBackend:
    backend: Path
    settings: dict[str, str]


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], capture_output=True, timeout=300
    )


def _release_commit(repository: Path, tag: str) -> str:
    found = _git(repository, "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    if found.returncode == 0:
        return found.stdout.decode().strip()
    fetched = _git(repository, "fetch", "--quiet", "--depth", "1", UPSTREAM_REPOSITORY, tag)
    assert fetched.returncode == 0, f"fetching {tag} failed: {fetched.stderr.decode()}"
    return _git(repository, "rev-parse", "FETCH_HEAD^{commit}").stdout.decode().strip()


def _boot_overlay(backend: Path, into: Path) -> Path:
    pins = {}
    for line in (backend / "requirements.txt").read_text().splitlines():
        requirement = line.split("#")[0].strip()
        name = requirement.split("=")[0].split("~")[0].split(">")[0].strip()
        if name in BOOT_PACKAGES:
            pins[name] = requirement
    missing = []
    for name in BOOT_PACKAGES:
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(pins.get(name, name))
    into.mkdir(parents=True, exist_ok=True)
    if missing:
        installer = ["uv", "pip", "install", "--python", sys.executable, "--target", str(into)]
        subprocess.run([*installer, "--no-deps", *missing], check=True, timeout=300)
    return into


def release_backend(tag: str, root: Path) -> ReleaseBackend:
    checkout = resolve_backend()
    if checkout is None:
        pytest.skip("open-webui backend source not found (set OPEN_WEBUI_SOURCE_DIR)")
    repository = checkout.parent
    commit = _release_commit(repository, tag)
    archive = _git(
        repository, "archive", commit, "backend", "static", "CHANGELOG.md", "package.json"
    )
    assert archive.returncode == 0, f"unpacking {tag} failed: {archive.stderr.decode()}"
    source = root / tag
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(source, filter="data")
    backend = source / "backend"
    # the release's version module reads its changelog at import
    shutil.copy(source / "CHANGELOG.md", backend / "open_webui" / "CHANGELOG.md")
    overlay = _boot_overlay(backend, root / f"{tag}-overlay")
    return ReleaseBackend(backend=backend, settings={"PYTHONPATH": str(overlay)})
