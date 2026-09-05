"""Guard: the Dockerfile must keep the build arguments and files the app needs.

Everything here fails silently rather than loudly. A build argument that is
declared before the first FROM is not visible inside a stage unless that stage
re-declares it, so renaming an ARG (or adding an ENV that reads one the stage
never re-declared) leaves the variable expanding to an empty string: the image
builds green and ships with an empty embedding model.

The copied files are the same shape of accident. env.py reads CHANGELOG.md at
module level and falls back to `pkgutil.get_data('open_webui', 'CHANGELOG.md')`,
which raises when the file is not in the image — an unguarded exception inside
the module every other backend module imports.

A source audit of the Dockerfile text; nothing is built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Provided by BuildKit itself, never declared in the file.
BUILDKIT_ARGS = {
    "BUILDPLATFORM",
    "BUILDOS",
    "BUILDARCH",
    "TARGETPLATFORM",
    "TARGETOS",
    "TARGETARCH",
    "TARGETVARIANT",
}

# Instructions whose arguments docker expands at build time. RUN is excluded on
# purpose: the shell inside it has its own variables that docker never sees.
EXPANDING_INSTRUCTIONS = {
    "ARG",
    "ENV",
    "COPY",
    "ADD",
    "EXPOSE",
    "WORKDIR",
    "USER",
    "VOLUME",
    "LABEL",
}

VARIABLE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


class Stage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.declared: set[str] = set()
        self.references: list[str] = []


def _logical_lines(dockerfile: str) -> list[str]:
    """Docker drops comment lines before joining `\\` continuations, so a
    comment inside a multi-line ENV does not end the instruction."""
    without_comments = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    joined = re.sub(r"\\\s*\n", " ", without_comments)
    return [line.strip() for line in joined.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def dockerfile_text(open_webui_backend: Path) -> str:
    path = open_webui_backend.parent / "Dockerfile"
    if not path.is_file():
        pytest.skip(f"no Dockerfile at {path}")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def stages(dockerfile_text: str) -> list[Stage]:
    stages: list[Stage] = []
    current = Stage("<global>")
    for line in _logical_lines(dockerfile_text):
        instruction, _, rest = line.partition(" ")
        instruction = instruction.upper()
        if instruction == "FROM":
            stages.append(current)
            match = re.search(r"\bAS\s+(\S+)", rest, re.IGNORECASE)
            current = Stage(match.group(1) if match else rest.split()[0])
        if instruction not in EXPANDING_INSTRUCTIONS:
            continue
        if instruction == "ARG":
            current.declared.add(rest.split("=")[0].split()[0])
        elif instruction == "ENV":
            current.declared.update(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", rest))
        current.references.extend(VARIABLE.findall(rest))
    stages.append(current)
    assert len(stages) > 1, "no FROM instruction found in the Dockerfile"
    return stages


def test_every_build_variable_is_declared_in_its_own_stage(stages: list[Stage]) -> None:
    """An ARG from before the first FROM is invisible until the stage
    re-declares it. Missing that leaves the value empty in the shipped image."""
    global_args = stages[0].declared
    undeclared = []
    for stage in stages[1:]:
        for variable in stage.references:
            if variable in stage.declared or variable in BUILDKIT_ARGS:
                continue
            hint = (
                " (declared globally, not re-declared in this stage)"
                if variable in global_args
                else ""
            )
            undeclared.append(f"{stage.name}: ${variable}{hint}")
    assert not undeclared, f"Dockerfile variables that expand to nothing: {sorted(set(undeclared))}"


def test_the_changelog_is_copied_into_the_image(dockerfile_text: str) -> None:
    """env.py reads it at import time and does not survive its absence."""
    assert re.search(r"^COPY .*CHANGELOG\.md", dockerfile_text, re.M), (
        "the Dockerfile no longer copies CHANGELOG.md; open_webui.env raises on import without it"
    )


def test_the_package_manifest_is_copied_into_the_image(dockerfile_text: str) -> None:
    """env.py reads VERSION out of package.json; without it every instance
    reports itself as 0.0.0."""
    assert re.search(r"^COPY .*package\.json", dockerfile_text, re.M), (
        "the Dockerfile no longer copies package.json into the runtime image"
    )


def test_the_exposed_port_matches_the_launcher_default(
    dockerfile_text: str, open_webui_backend: Path
) -> None:
    """EXPOSE and start.sh's PORT default have to name the same port, or every
    published container maps a port nothing is listening on."""
    exposed = re.findall(r"^EXPOSE\s+(\d+)", dockerfile_text, re.M)
    assert exposed, "the Dockerfile exposes no port"

    start_sh = (open_webui_backend / "start.sh").read_text(encoding="utf-8")
    default_port = re.search(r'PORT="\$\{PORT:-(\d+)\}"', start_sh)
    assert default_port, "could not read the PORT default out of start.sh"
    assert default_port.group(1) in exposed, (
        f"start.sh serves on {default_port.group(1)} but the Dockerfile exposes {exposed}"
    )


def test_the_base_image_python_matches_the_supported_range(
    dockerfile_text: str, open_webui_backend: Path
) -> None:
    """Building on a python the project does not claim to support is how a
    dependency resolves to a wheel nobody tested."""
    image = re.search(r"^FROM python:(\d+)\.(\d+)", dockerfile_text, re.M)
    assert image, "no python base image found in the Dockerfile"
    built_on = (int(image.group(1)), int(image.group(2)))

    pyproject = (open_webui_backend.parent / "pyproject.toml").read_text(encoding="utf-8")
    requires = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject)
    assert requires, "no requires-python in pyproject.toml"

    floor = re.search(r">=\s*(\d+)\.(\d+)", requires.group(1))
    assert floor, f"could not read a lower bound from requires-python: {requires.group(1)}"
    assert built_on >= (int(floor.group(1)), int(floor.group(2))), (
        f"the image builds on python {built_on[0]}.{built_on[1]} but pyproject requires "
        f"{requires.group(1)}"
    )

    ceiling = re.search(r"<\s*(\d+)\.(\d+)", requires.group(1))
    if ceiling:
        assert built_on < (int(ceiling.group(1)), int(ceiling.group(2))), (
            f"the image builds on python {built_on[0]}.{built_on[1]}, outside {requires.group(1)}"
        )
