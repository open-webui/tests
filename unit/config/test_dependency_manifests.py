"""Guard: the dependency manifests must agree with each other.

Open WebUI lists its Python dependencies in several places:

  backend/requirements.txt     what the standard Docker image installs
  pyproject.toml [project]     what `pip install open-webui` installs
  backend/requirements-*.txt   the other images' sets (the slim image today)

A bump applied to one and not the others is invisible until somebody installs through the path
nobody tested and gets another version of fastapi, pydantic or sqlalchemy than CI ever ran.
Nothing in the product compares these files, so nothing catches the drift.

Entries are parsed with `packaging` as pip reads them (extras, markers, specifier spacing), and
only packages that appear in both files of a pair are compared, so a package that deliberately
lives in one list alone stays legal.

Discriminates: passes on bbfa876af; an unpinned or repeated requirement, a pyproject dependency
missing from requirements.txt, and a pin that differs in pyproject or requirements-slim.txt each
fail their test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

COMMENT = re.compile(r"(^|\s)#.*")


def _read_requirements(path: Path) -> list[Requirement]:
    """The requirements pip reads from a file; option lines (`-r`, `--index-url`) skipped."""
    entries = []
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = COMMENT.sub("", raw_line).strip()
        if not line or line.startswith("-"):
            continue
        try:
            entries.append(Requirement(line))
        except InvalidRequirement as error:
            raise AssertionError(f"{path.name}:{number} is not a requirement pip reads: {error}")
    return entries


def _by_name(entries: list[Requirement]) -> dict[str, Requirement]:
    return {canonicalize_name(entry.name): entry for entry in entries}


def _drift(entries: list[Requirement], pinned: dict[str, Requirement]) -> list[str]:
    return [
        f"{entry} vs {pinned[name]}"
        for entry in entries
        if (name := canonicalize_name(entry.name)) in pinned
        and entry.specifier
        and entry.specifier != pinned[name].specifier
    ]


@pytest.fixture(scope="module")
def requirements(open_webui_backend: Path) -> list[Requirement]:
    path = open_webui_backend / "requirements.txt"
    assert path.is_file(), f"no {path}; retarget this guard at the image's requirements list"
    return _read_requirements(path)


@pytest.fixture(scope="module")
def pinned(requirements: list[Requirement]) -> dict[str, Requirement]:
    return _by_name(requirements)


@pytest.fixture(scope="module")
def project(open_webui_backend: Path) -> dict:
    path = open_webui_backend.parent / "pyproject.toml"
    assert path.is_file(), f"no {path}; retarget this guard at the package metadata"
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def test_every_requirement_is_pinned(requirements: list[Requirement]) -> None:
    """Unpinned, two builds of the same commit can install different code."""
    unpinned = [
        str(entry)
        for entry in requirements
        if not entry.url and not any(spec.operator in ("==", "===") for spec in entry.specifier)
    ]
    assert not unpinned, f"requirements.txt entries without an == pin: {unpinned}"


def test_no_package_is_listed_twice(requirements: list[Requirement]) -> None:
    """pip honours the last line, so a stale duplicate quietly wins or loses by position."""
    seen: dict[tuple[str, str], list[str]] = {}
    for entry in requirements:
        seen.setdefault((canonicalize_name(entry.name), str(entry.marker)), []).append(str(entry))
    duplicates = [entries for entries in seen.values() if len(entries) > 1]
    assert not duplicates, f"packages listed more than once in requirements.txt: {duplicates}"


def test_every_pyproject_dependency_is_in_requirements(
    project: dict, pinned: dict[str, Requirement]
) -> None:
    missing = [
        entry
        for entry in project["dependencies"]
        if canonicalize_name(Requirement(entry).name) not in pinned
    ]
    assert not missing, (
        f"pyproject dependencies absent from backend/requirements.txt: {missing}. "
        "A pip install would pull them, the Docker image would not."
    )


def test_pyproject_carries_the_same_versions(project: dict, pinned: dict[str, Requirement]) -> None:
    entries = [
        Requirement(entry)
        for group in [project["dependencies"], *project.get("optional-dependencies", {}).values()]
        for entry in group
    ]
    drifted = _drift(entries, pinned)
    assert not drifted, f"pyproject vs requirements.txt version drift: {drifted}"


def test_the_other_images_pin_the_same_versions(
    open_webui_backend: Path, pinned: dict[str, Requirement]
) -> None:
    others = sorted(open_webui_backend.glob("requirements-*.txt"))
    drifted = {
        path.name: drift for path in others if (drift := _drift(_read_requirements(path), pinned))
    }
    assert not drifted, f"version drift from requirements.txt: {drifted}"
