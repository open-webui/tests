"""Guard: the shipped docker compose stack must come up on a clean machine.

docker-compose.yaml is what the install docs tell people to run, and the
overlays next to it are merged on top by docker-compose-launcher.sh. All of it
is hand-maintained YAML that nothing validates: an overlay the launcher names
but nobody added, a `${VAR}` with no default (compose substitutes an empty
string and the container gets a broken port mapping or an empty image tag), or
a volume used by a service but never declared, which compose rejects outright.

The one that costs users data rather than a startup is the open-webui service
losing its volume: every chat, file and setting lives under /app/backend/data
and goes away with the container.

Reads the YAML, runs no docker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

DATA_PATH = "/app/backend/data"


@pytest.fixture(scope="module")
def repo_root(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent


@pytest.fixture(scope="module")
def compose_files(repo_root: Path) -> list[Path]:
    files = sorted(repo_root.glob("docker-compose*.yaml"))
    if not files:
        pytest.skip(f"no compose files under {repo_root}")
    return files


@pytest.fixture(scope="module")
def base_compose(repo_root: Path) -> dict:
    path = repo_root / "docker-compose.yaml"
    if not path.is_file():
        pytest.skip(f"no docker-compose.yaml at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_every_compose_file_is_valid_yaml(compose_files: list[Path]) -> None:
    broken = []
    for path in compose_files:
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            broken.append((path.name, str(error).splitlines()[0]))
            continue
        if not isinstance(parsed, dict) or not parsed.get("services"):
            broken.append((path.name, "no services mapping"))
    assert not broken, f"unusable compose files: {broken}"


def test_every_named_volume_is_declared(compose_files: list[Path]) -> None:
    """`docker compose up` refuses to start a service that mounts a named
    volume the file never declares."""
    undeclared = []
    for path in compose_files:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = set((document.get("volumes") or {}).keys())
        for service, definition in document["services"].items():
            for mount in definition.get("volumes") or []:
                if not isinstance(mount, str):
                    continue
                source = mount.split(":")[0]
                if source.startswith((".", "/", "~", "$")):
                    continue  # a bind mount, not a named volume
                if source not in declared:
                    undeclared.append((path.name, service, source))
    assert not undeclared, f"named volumes used but not declared: {undeclared}"


def test_every_interpolation_has_a_default(compose_files: list[Path]) -> None:
    """Compose substitutes an unset variable with an empty string and carries
    on, so a missing default turns into an empty image tag or a broken port
    mapping for anyone running without a .env file."""
    bare = []
    for path in compose_files:
        text = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        # `$${VAR}` is an escaped literal that reaches the container's own shell.
        for match in re.finditer(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text):
            bare.append((path.name, match.group(1)))
    assert not bare, f"compose interpolations with no default value: {sorted(set(bare))}"


def test_the_launcher_only_names_compose_files_that_exist(repo_root: Path) -> None:
    launcher = repo_root / "docker-compose-launcher.sh"
    if not launcher.is_file():
        pytest.skip("no docker-compose-launcher.sh in this checkout")
    referenced = sorted(
        set(
            re.findall(r"docker-compose[A-Za-z0-9._-]*\.yaml", launcher.read_text(encoding="utf-8"))
        )
    )
    assert referenced, "the launcher references no compose files"
    missing = [name for name in referenced if not (repo_root / name).is_file()]
    assert not missing, f"docker-compose-launcher.sh passes -f for missing files: {missing}"


def test_the_default_stack_defines_the_open_webui_service(base_compose: dict) -> None:
    assert "open-webui" in base_compose["services"], (
        f"docker-compose.yaml no longer defines open-webui: {sorted(base_compose['services'])}"
    )


def test_the_data_directory_is_persisted(base_compose: dict) -> None:
    """Without this mount every chat, file and setting dies with the container."""
    mounts = base_compose["services"]["open-webui"].get("volumes") or []
    assert any(isinstance(m, str) and m.endswith(DATA_PATH) for m in mounts), (
        f"the open-webui service does not mount {DATA_PATH}: {mounts}"
    )


def test_the_published_port_reaches_the_server(base_compose: dict, repo_root: Path) -> None:
    """The container half of the mapping has to be the port the app listens
    on, or the published port answers nothing."""
    mappings = base_compose["services"]["open-webui"].get("ports") or []
    container_ports = {str(m).rsplit(":", 1)[-1] for m in mappings}

    start_sh = (repo_root / "backend" / "start.sh").read_text(encoding="utf-8")
    default_port = re.search(r'PORT="\$\{PORT:-(\d+)\}"', start_sh)
    assert default_port, "could not read the PORT default out of start.sh"
    assert default_port.group(1) in container_ports, (
        f"compose publishes to container port(s) {sorted(container_ports)} "
        f"but start.sh serves on {default_port.group(1)}"
    )
