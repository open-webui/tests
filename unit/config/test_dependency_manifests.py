"""Guard: the three dependency manifests must agree with each other.

Open WebUI ships its Python dependency list three times over:

  backend/requirements.txt      what the Docker image installs
  pyproject.toml [project]      what `pip install open-webui` installs
  backend/requirements-min.txt  the slim set used by the minimal image

A bump applied to one file and not the others is invisible until somebody
installs through the path nobody tested and gets a different version of
fastapi, pydantic or sqlalchemy than CI ever ran. Nothing in the product
compares these files, so nothing catches the drift.

Pure text and TOML parsing — no backend import, no dependency needed.

Only entries whose package name appears in both files are compared, so an
extra that deliberately lives in pyproject alone (mariadb) stays legal.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

# `name[extra1,extra2] >= 1.2` -> ("name", "[extra1,extra2]", ">= 1.2")
_ENTRY = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*(.*)$")


def _split(entry: str) -> tuple[str, str]:
    """Return (normalised package name, version specifier) for one entry."""
    match = _ENTRY.match(entry.strip())
    assert match is not None, f"unparseable requirement: {entry!r}"
    name = match.group(1).lower().replace("_", "-")
    return name, (match.group(3) or "").strip()


def _read_requirements(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#")[0].strip()
        if line:
            entries.append(line)
    return entries


@pytest.fixture(scope="module")
def requirements(open_webui_backend: Path) -> list[str]:
    return _read_requirements(open_webui_backend / "requirements.txt")


@pytest.fixture(scope="module")
def requirements_min(open_webui_backend: Path) -> list[str]:
    path = open_webui_backend / "requirements-min.txt"
    if not path.is_file():
        pytest.skip("no requirements-min.txt in this checkout")
    return _read_requirements(path)


@pytest.fixture(scope="module")
def pyproject(open_webui_backend: Path) -> dict:
    path = open_webui_backend.parent / "pyproject.toml"
    if not path.is_file():
        pytest.skip(f"no pyproject.toml at {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pinned_versions(requirements: list[str]) -> dict[str, str]:
    return dict(_split(entry) for entry in requirements)


def test_every_requirement_is_pinned(requirements: list[str]) -> None:
    """An unpinned entry means two builds of the same commit can install
    different code — the whole point of shipping a lockfile-style list."""
    unpinned = [entry for entry in requirements if "==" not in entry]
    assert not unpinned, f"requirements.txt entries without an == pin: {unpinned}"


def test_no_package_is_listed_twice(requirements: list[str]) -> None:
    """pip silently honours the last line, so a stale duplicate above a fresh
    pin quietly wins or loses depending on the order."""
    seen: dict[str, list[str]] = {}
    for entry in requirements:
        seen.setdefault(_split(entry)[0], []).append(entry)
    duplicates = {name: entries for name, entries in seen.items() if len(entries) > 1}
    assert not duplicates, f"packages listed more than once in requirements.txt: {duplicates}"


def test_every_pyproject_dependency_is_in_requirements(
    pyproject: dict, pinned_versions: dict[str, str]
) -> None:
    missing = [
        entry
        for entry in pyproject["project"]["dependencies"]
        if _split(entry)[0] not in pinned_versions
    ]
    assert not missing, (
        f"pyproject dependencies absent from backend/requirements.txt: {missing}. "
        "A pip install would pull them, the Docker image would not."
    )


def test_pyproject_dependencies_carry_the_same_versions(
    pyproject: dict, pinned_versions: dict[str, str]
) -> None:
    drifted = []
    for entry in pyproject["project"]["dependencies"]:
        name, spec = _split(entry)
        if name in pinned_versions and pinned_versions[name] != spec:
            drifted.append((entry, f"{name}{pinned_versions[name]}"))
    assert not drifted, (
        f"pyproject vs requirements.txt version drift (pyproject, requirements): {drifted}"
    )


def test_pyproject_extras_carry_the_same_versions(
    pyproject: dict, pinned_versions: dict[str, str]
) -> None:
    drifted = []
    for group, entries in pyproject["project"].get("optional-dependencies", {}).items():
        for entry in entries:
            name, spec = _split(entry)
            if name in pinned_versions and pinned_versions[name] != spec:
                drifted.append((group, entry, f"{name}{pinned_versions[name]}"))
    assert not drifted, f"pyproject extras drifted from requirements.txt: {drifted}"


def test_requirements_min_is_a_subset_of_requirements(
    requirements_min: list[str], pinned_versions: dict[str, str]
) -> None:
    """The minimal image must not install anything the full one never sees."""
    extra = [entry for entry in requirements_min if _split(entry)[0] not in pinned_versions]
    assert not extra, f"requirements-min.txt entries absent from requirements.txt: {extra}"


def test_requirements_min_pins_match(
    requirements_min: list[str], pinned_versions: dict[str, str]
) -> None:
    """Unpinned entries in the minimal list are deliberate; the ones that do
    carry a version must not disagree with the full list."""
    drifted = []
    for entry in requirements_min:
        name, spec = _split(entry)
        if spec and name in pinned_versions and pinned_versions[name] != spec:
            drifted.append((entry, f"{name}{pinned_versions[name]}"))
    assert not drifted, f"requirements-min.txt version drift: {drifted}"
