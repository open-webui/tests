"""Guard: the shipped docker compose stack must come up on a clean machine.

docker-compose.yaml is what the install docs tell people to run, and the overlays next to it are
merged on top by docker-compose-launcher.sh. All of it is hand-maintained YAML that nothing
validates: an overlay the launcher names but nobody added, a variable with no default (compose
substitutes an empty string, so the container gets a broken port mapping or an empty image tag),
or a named volume nobody declared, which compose refuses outright.

The one that costs users data rather than a startup is the open-webui service losing its data
volume: every chat, file and setting lives under /app/backend/data and goes with the container.

The YAML is parsed and walked, interpolations are read with compose's own grammar; no docker runs.

Discriminates: passes on bbfa876af; a file without services, a named volume nobody declares, a
bare `${VAR}`, a launcher `-f` for a missing file, the data mount moved and the container port
changed each fail their test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pytest

from unit.config.container_files import served_port

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

BASE_FILE = "docker-compose.yaml"
DATA_PATH = "/app/backend/data"
# `$$` is a literal dollar; `${NAME}` and `$NAME` substitute; `${NAME-x}`, `${NAME:?x}` and the
# other modifiers carry a default or an explicit error.
INTERPOLATION = re.compile(r"\$(?:(\$)|\{([^}]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
BARE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@pytest.fixture(scope="module")
def repo_root(open_webui_backend: Path) -> Path:
    return open_webui_backend.parent


@pytest.fixture(scope="module")
def compose_files(repo_root: Path) -> dict[str, dict]:
    """Every compose file by name, parsed."""
    paths = sorted(repo_root.glob("docker-compose*.yaml"))
    assert any(path.name == BASE_FILE for path in paths), f"no {BASE_FILE} under {repo_root}"
    return {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths}


@pytest.fixture(scope="module")
def open_webui_service(compose_files: dict[str, dict]) -> dict:
    services = compose_files[BASE_FILE]["services"]
    assert "open-webui" in services, f"{BASE_FILE} no longer defines open-webui: {sorted(services)}"
    return services["open-webui"]


def _strings(node) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


def _named_volumes(service: dict) -> Iterator[str]:
    for mount in service.get("volumes") or []:
        if isinstance(mount, dict):
            if mount.get("type", "volume") == "volume" and mount.get("source"):
                yield mount["source"]
            continue
        source, separator, _ = str(mount).partition(":")
        if separator and not source.startswith((".", "/", "~", "$")):
            yield source


def _mount_targets(service: dict) -> set[str]:
    targets = set()
    for mount in service.get("volumes") or []:
        target = mount.get("target") if isinstance(mount, dict) else str(mount).split(":")[1:2]
        targets.update([target] if isinstance(target, str) else target)
    return {target.rstrip("/") for target in targets}


def _container_ports(service: dict) -> set[str]:
    ports = set()
    for mapping in service.get("ports") or []:
        container = mapping.get("target") if isinstance(mapping, dict) else mapping
        ports.add(str(container).rsplit(":", 1)[-1].split("/")[0])
    return ports


def test_every_compose_file_defines_services(compose_files: dict[str, dict]) -> None:
    broken = [
        name for name, document in compose_files.items() if not (document or {}).get("services")
    ]
    assert not broken, f"compose files with no services mapping: {broken}"


def test_every_named_volume_is_declared(compose_files: dict[str, dict]) -> None:
    """compose refuses a service mounting a named volume its file (or the base) never declares."""
    base_volumes = set(compose_files[BASE_FILE].get("volumes") or {})
    undeclared = [
        (name, service_name, volume)
        for name, document in compose_files.items()
        for service_name, service in (document.get("services") or {}).items()
        for volume in _named_volumes(service or {})
        if volume not in set(document.get("volumes") or {}) | base_volumes
    ]
    assert not undeclared, f"named volumes used but never declared: {undeclared}"


def test_every_interpolation_has_a_default(compose_files: dict[str, dict]) -> None:
    """An unset variable becomes an empty string for anyone running without a .env file."""
    bare = sorted(
        {
            (name, match.group(2) or match.group(3))
            for name, document in compose_files.items()
            for text in _strings(document)
            for match in INTERPOLATION.finditer(text)
            if match.group(3) or (match.group(2) and BARE_NAME.fullmatch(match.group(2)))
        }
    )
    assert not bare, f"compose interpolations with no default value: {bare}"


def test_the_launcher_only_names_compose_files_that_exist(repo_root: Path) -> None:
    launcher = repo_root / "docker-compose-launcher.sh"
    assert launcher.is_file(), f"no {launcher.name}; retarget this guard at the stack's launcher"
    named = set(re.findall(r"docker-compose[\w.-]*\.yaml", launcher.read_text(encoding="utf-8")))

    assert named, "the launcher names no compose files"
    missing = sorted(name for name in named if not (repo_root / name).is_file())
    assert not missing, f"docker-compose-launcher.sh passes -f for missing files: {missing}"


def test_the_data_directory_is_persisted(open_webui_service: dict) -> None:
    """Without this mount every chat, file and setting dies with the container."""
    assert DATA_PATH in _mount_targets(open_webui_service), (
        f"the open-webui service no longer mounts {DATA_PATH}: {open_webui_service.get('volumes')}"
    )


def test_the_published_port_reaches_the_server(open_webui_service: dict, repo_root: Path) -> None:
    """The container half of the mapping must be the port the image serves on."""
    published = _container_ports(open_webui_service)

    assert served_port(repo_root) in published, (
        f"compose publishes container port(s) {sorted(published)} but the image serves on "
        f"{served_port(repo_root)}"
    )
