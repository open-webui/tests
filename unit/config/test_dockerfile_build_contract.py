"""Guard: the Dockerfile must keep the build arguments, files and port the app needs.

Everything here fails silently rather than loudly. A build argument declared before the first FROM
is invisible inside a stage that does not re-declare it, so renaming an ARG (or reading one in an
ENV the stage never re-declared) leaves it empty: the image builds green and ships with an empty
embedding model. env.py reads CHANGELOG.md and package.json from the directory above the backend
at import, so an image without them raises on import or reports itself as 0.0.0. An EXPOSE that
names another port than the one the server listens on publishes nothing.

The Dockerfile is parsed the way docker reads it (see `container_files`); nothing is built.

Discriminates: passes on bbfa876af; a global ARG the runtime stage reads without re-declaring it,
the runtime COPY of CHANGELOG.md or package.json removed, another EXPOSEd port and a python base
image outside requires-python each fail their test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

import pytest
from packaging.specifiers import SpecifierSet

from unit.config.container_files import Dockerfile, read_dockerfile, served_port

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

# Instructions whose arguments docker expands at build time. RUN is left out on purpose: the
# shell inside it has variables of its own that docker never sees.
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


@pytest.fixture(scope="module")
def repo_root(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent


@pytest.fixture(scope="module")
def dockerfile(repo_root: Path) -> Dockerfile:
    return read_dockerfile(repo_root)


def _declared(stage) -> set[str]:
    names = {word.split("=", 1)[0] for arg in stage.of("ARG") for word in arg.words()}
    return names | set(stage.env())


def test_every_build_variable_is_declared_in_its_own_stage(dockerfile: Dockerfile) -> None:
    """A global ARG is invisible until the stage re-declares it; missing that leaves it empty."""
    global_args = _declared(dockerfile.global_args)
    undeclared = set()
    for stage in dockerfile.build_stages:
        declared = _declared(stage) | BUILDKIT_ARGS
        for instruction in stage.instructions:
            if instruction.keyword not in EXPANDING_INSTRUCTIONS:
                continue
            for variable in VARIABLE.findall(instruction.arguments):
                if variable in declared:
                    continue
                hint = " (global, not re-declared here)" if variable in global_args else ""
                undeclared.add(f"{stage.name}: ${variable}{hint}")
    assert not undeclared, f"Dockerfile variables that expand to nothing: {sorted(undeclared)}"


@pytest.mark.parametrize("filename", ["CHANGELOG.md", "package.json"])
def test_the_runtime_image_carries_what_env_py_reads(dockerfile: Dockerfile, filename: str) -> None:
    """env.py reads both from the parent of the backend directory, the runtime WORKDIR."""
    runtime = dockerfile.runtime
    backend_dir = runtime.of("WORKDIR")[-1].workdir
    expected = str(PurePosixPath(backend_dir).parent / filename)

    assert expected in runtime.copied_to(filename), (
        f"the runtime stage no longer copies {filename} to {expected}; open_webui.env reads it "
        f"at import (copies found: {runtime.copied_to(filename)})"
    )


def test_the_exposed_port_is_the_served_port(dockerfile: Dockerfile, repo_root: Path) -> None:
    exposed = {
        word.split("/")[0] for expose in dockerfile.runtime.of("EXPOSE") for word in expose.words()
    }

    assert served_port(repo_root) in exposed, (
        f"the image serves on {served_port(repo_root)} but EXPOSEs {sorted(exposed)}"
    )


def test_the_base_image_python_is_a_supported_version(
    dockerfile: Dockerfile, repo_root: Path
) -> None:
    """Building on a python the project does not claim to support resolves untested wheels."""
    images = [stage.base for stage in dockerfile.build_stages if stage.base.startswith("python:")]
    assert images, "no stage builds on a python: image; retarget this guard at the runtime base"
    version = re.match(r"python:(\d+\.\d+)", images[-1])
    assert version, f"cannot read a python version from {images[-1]}"

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    supported = SpecifierSet(pyproject["project"]["requires-python"])
    assert supported.contains(version.group(1)), (
        f"the image builds on python {version.group(1)}, outside requires-python {supported}"
    )
