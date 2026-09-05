"""Guard: backend/start.sh must survive the ordinary container setups.

start.sh is the entry point of the official image, so everything it does
happens before a single line of Python runs. When it exits early nobody gets
a stack trace, an HTTP error or a log line from the app — the container just
dies, which is how open-webui#24560 (an unset WEB_LOADER_ENGINE under
`set -u`) reached users. test_start_sh.py pins that specific regression; this
file covers the boot paths around it that no test exercised.

Each test runs the real script with the uvicorn launch replaced by an echo of
the arguments it was about to exec with, so the preamble runs for real
(secret-key generation, defaults, argument assembly) without starting a
server. A fake `python3` on PATH keeps the `command -v` lookup happy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCH_MARKERS = ("exec env", "exec ")
LAUNCH_ECHO = 'echo "LAUNCH host=$HOST port=$PORT args=${ARGS[*]}"\n'


@pytest.fixture(scope="module")
def bash() -> str:
    found = shutil.which("bash")
    if found is None:
        pytest.skip("bash not available on PATH")
    return found


@pytest.fixture(scope="module")
def start_sh_stub(open_webui_backend: Path) -> str:
    """start.sh with the uvicorn exec swapped for an echo of its arguments."""
    start_sh = open_webui_backend / "start.sh"
    if not start_sh.is_file():
        pytest.skip(f"start.sh not found at {start_sh}")
    source = start_sh.read_text(encoding="utf-8")
    for marker in LAUNCH_MARKERS:
        index = source.find(marker)
        if index != -1:
            return source[:index] + LAUNCH_ECHO
    pytest.skip("could not locate the uvicorn launch in start.sh — update this test")


class Run:
    """The result of one stubbed start.sh run, plus its working directory."""

    def __init__(self, completed: subprocess.CompletedProcess[str], workdir: Path) -> None:
        self.returncode = completed.returncode
        self.stdout = completed.stdout
        self.stderr = completed.stderr
        self.workdir = workdir

    @property
    def launch_line(self) -> str:
        for line in self.stdout.splitlines():
            if line.startswith("LAUNCH "):
                return line
        raise AssertionError(
            f"start.sh never reached the launch (exit {self.returncode})\n"
            f"stdout: {self.stdout}\nstderr: {self.stderr}"
        )


@pytest.fixture
def run_start_sh(bash: str, start_sh_stub: str, tmp_path: Path):
    """Run the stubbed script in a scratch directory with a clean environment."""

    def _run(
        env: dict[str, str] | None = None,
        argv: tuple[str, ...] = (),
        files: dict[str, str] | None = None,
    ) -> Run:
        workdir = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        workdir.mkdir()
        script = workdir / "start.sh"
        script.write_text(start_sh_stub, encoding="utf-8")
        for name, content in (files or {}).items():
            (workdir / name).write_text(content, encoding="utf-8")

        # `PYTHON_CMD=$(command -v python3 || command -v python)` runs under
        # `set -e`, so the lookup has to resolve to something.
        bin_dir = workdir / "bin"
        bin_dir.mkdir()
        fake_python = bin_dir / "python3"
        fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)

        # Strip anything the test runner's own environment would inject: the
        # scenario under test is a container that sets only what it sets.
        inherited = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "WEBUI_SECRET_KEY",
                "WEBUI_JWT_SECRET_KEY",
                "WEBUI_SECRET_KEY_FILE",
                "WEBUI_SECRET_KEY_LENGTH",
                "WEB_LOADER_ENGINE",
                "HOST",
                "PORT",
                "UVICORN_WORKERS",
            }
        }
        inherited["PATH"] = str(bin_dir) + os.pathsep + inherited.get("PATH", "")
        inherited.update(env or {})

        completed = subprocess.run(
            [bash, str(script), *argv],
            capture_output=True,
            text=True,
            env=inherited,
            cwd=str(workdir),
            timeout=60,
        )
        return Run(completed, workdir)

    return _run


def test_a_missing_secret_key_file_is_generated(run_start_sh) -> None:
    """First boot of a fresh container: no key in the env, no key on disk."""
    run = run_start_sh()
    key_file = run.workdir / ".webui_secret_key"
    assert run.returncode == 0, run.stderr
    assert key_file.is_file(), f"start.sh did not create a key file\nstdout: {run.stdout}"
    assert key_file.read_text(encoding="utf-8").strip(), "generated key file is empty"


def test_an_existing_secret_key_file_is_reused_not_overwritten(run_start_sh) -> None:
    """Overwriting it would invalidate every session and API token on restart."""
    run = run_start_sh(files={".webui_secret_key": "do-not-touch-me\n"})
    assert run.returncode == 0, run.stderr
    assert (run.workdir / ".webui_secret_key").read_text(encoding="utf-8") == "do-not-touch-me\n"


@pytest.mark.parametrize("length", ["abc", "0", "-4", " ", "12.5"])
def test_a_nonsense_secret_key_length_fails_loudly(run_start_sh, length: str) -> None:
    """Better a named error than a zero-length key silently accepted as the
    signing secret for every JWT the instance ever issues."""
    run = run_start_sh({"WEBUI_SECRET_KEY_LENGTH": length})
    assert run.returncode != 0, f"start.sh accepted WEBUI_SECRET_KEY_LENGTH={length!r}"
    assert "positive integer" in run.stderr, run.stderr


def test_an_empty_secret_key_length_falls_back_to_the_default(run_start_sh) -> None:
    """An env var set but left blank is the normal shape of a docker-compose
    variable that never got a value; `${VAR:-24}` must absorb it."""
    run = run_start_sh({"WEBUI_SECRET_KEY_LENGTH": ""})
    assert run.returncode == 0, run.stderr
    assert (run.workdir / ".webui_secret_key").read_text(encoding="utf-8").strip()


def test_a_secret_key_in_the_environment_skips_the_file_entirely(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "from-the-env"})
    assert run.returncode == 0, run.stderr
    assert not (run.workdir / ".webui_secret_key").exists(), (
        "start.sh wrote a key file even though WEBUI_SECRET_KEY was set"
    )


def test_the_key_file_path_may_contain_spaces(run_start_sh) -> None:
    """A bind-mounted secret under a path like /run/my secrets/key."""
    run = run_start_sh({"WEBUI_SECRET_KEY_FILE": "my secret key"})
    assert run.returncode == 0, run.stderr
    assert (run.workdir / "my secret key").is_file(), run.stdout


def test_host_and_port_default_to_the_documented_values(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "x"})
    assert "host=0.0.0.0" in run.launch_line
    assert "port=8080" in run.launch_line


def test_host_and_port_honour_the_environment(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "x", "HOST": "127.0.0.1", "PORT": "9099"})
    assert "host=127.0.0.1" in run.launch_line
    assert "port=9099" in run.launch_line


def test_the_default_launch_passes_the_worker_count(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "x", "UVICORN_WORKERS": "4"})
    assert "--workers 4" in run.launch_line, run.launch_line


def test_arguments_given_to_the_script_replace_the_defaults(run_start_sh) -> None:
    """`docker run ... start.sh --reload` must not also get --workers."""
    run = run_start_sh({"WEBUI_SECRET_KEY": "x"}, argv=("--reload",))
    assert "args=--reload" in run.launch_line, run.launch_line
    assert "--workers" not in run.launch_line, run.launch_line
