"""Guard: backend/start.sh, the image's entry point, must get ordinary containers to uvicorn.

Regression, open-webui#24560: commit 070ab2650 put start.sh under `set -euo pipefail` and tested
`${WEB_LOADER_ENGINE,,}` without a default, so every container that did not set
WEB_LOADER_ENGINE (most of them) died with "WEB_LOADER_ENGINE: unbound variable" before a line of
Python ran. When start.sh exits early there is no stack trace, no HTTP error and no app log, the
container just dies, so the boot paths around it are pinned too: secret key generation and
reuse, the key length check, host and port defaults and the uvicorn arguments.

Each run executes the real, unmodified script from a scratch copy with a stand-in `python3` first
on PATH. The script's closing `exec` launches the stand-in, which records the arguments and the
secret key it was handed, so the whole preamble runs for real without starting a server.

Discriminates: passes on bbfa876af, fails with WEB_LOADER_ENGINE dropped from the defaulting line
(the unset case exits with "unbound variable").
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

LAUNCH_MARKER = "LAUNCH "
LAUNCH_RECORDER = f"""#!{sys.executable}
import json, os, sys

record = {{"argv": sys.argv[1:], "secret_key": os.environ.get("WEBUI_SECRET_KEY", "")}}
print({LAUNCH_MARKER!r} + json.dumps(record))
"""
# What a test runner's own environment could inject; a container sets only what it sets.
STRIPPED_ENV = {
    "WEBUI_SECRET_KEY",
    "WEBUI_JWT_SECRET_KEY",
    "WEBUI_SECRET_KEY_FILE",
    "WEBUI_SECRET_KEY_LENGTH",
    "WEB_LOADER_ENGINE",
    "HOST",
    "PORT",
    "UVICORN_WORKERS",
    "SPACE_ID",
    "USE_OLLAMA_DOCKER",
    "USE_CUDA_DOCKER",
}


@dataclass
class Run:
    returncode: int
    stdout: str
    stderr: str
    workdir: Path

    @property
    def launch(self) -> dict:
        """What start.sh exec'd the server with; fails if it never got there."""
        for line in self.stdout.splitlines():
            if line.startswith(LAUNCH_MARKER):
                return json.loads(line.removeprefix(LAUNCH_MARKER))
        raise AssertionError(
            f"start.sh never reached the launch (exit {self.returncode})\n"
            f"stdout: {self.stdout}\nstderr: {self.stderr}"
        )

    def option(self, name: str) -> str:
        argv = self.launch["argv"]
        assert name in argv, f"{name} is not among the launch arguments {argv}"
        return argv[argv.index(name) + 1]

    def key_file(self, name: str = ".webui_secret_key") -> Path:
        return self.workdir / name


@pytest.fixture(scope="module")
def start_sh(open_webui_backend: Path) -> Path:
    if shutil.which("bash") is None:
        pytest.skip("start.sh needs bash")
    path = open_webui_backend / "start.sh"
    assert path.is_file(), f"no start.sh at {path}; retarget at the image's entry point"
    return path


@pytest.fixture
def run_start_sh(start_sh: Path, tmp_path: Path):
    def run(env: dict[str, str] | None = None, *arguments: str, files: dict | None = None) -> Run:
        workdir = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        bin_dir = workdir / "bin"
        bin_dir.mkdir(parents=True)
        shutil.copy(start_sh, workdir / "start.sh")
        recorder = bin_dir / "python3"
        recorder.write_text(LAUNCH_RECORDER, encoding="utf-8")
        recorder.chmod(0o755)
        for name, content in (files or {}).items():
            (workdir / name).write_text(content, encoding="utf-8")

        inherited = {key: value for key, value in os.environ.items() if key not in STRIPPED_ENV}
        inherited["PATH"] = f"{bin_dir}{os.pathsep}{inherited.get('PATH', '')}"
        completed = subprocess.run(
            ["bash", str(workdir / "start.sh"), *arguments],
            env={**inherited, **(env or {})},
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return Run(completed.returncode, completed.stdout, completed.stderr, workdir)

    return run


# regression: an unset WEB_LOADER_ENGINE killed the container (#24560)


@pytest.mark.regression
def test_an_unset_web_loader_engine_still_reaches_the_launch(run_start_sh) -> None:
    run = run_start_sh()

    assert "unbound variable" not in run.stderr, (
        "start.sh reads an optional variable without a default under `set -u`, which kills "
        f"every container that does not set it (#24560): {run.stderr.strip()}"
    )
    assert "open_webui.main:app" in run.launch["argv"]


# secret key


def test_a_missing_secret_key_is_generated_and_handed_to_the_server(run_start_sh) -> None:
    run = run_start_sh()

    generated = run.key_file().read_text(encoding="utf-8").strip()
    assert generated, "start.sh wrote an empty key file"
    assert run.launch["secret_key"] == generated


def test_an_existing_key_file_is_reused_not_overwritten(run_start_sh) -> None:
    """Overwriting it would invalidate every session and API token on restart."""
    run = run_start_sh(files={".webui_secret_key": "do-not-touch-me\n"})

    assert run.key_file().read_text(encoding="utf-8") == "do-not-touch-me\n"
    assert run.launch["secret_key"] == "do-not-touch-me"


def test_a_key_in_the_environment_skips_the_file(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "from-the-env"})

    assert run.launch["secret_key"] == "from-the-env"
    assert not run.key_file().exists(), "start.sh wrote a key file although the env set one"


def test_the_key_file_path_may_contain_spaces(run_start_sh) -> None:
    """A bind-mounted secret under a path like /run/my secrets/key."""
    run = run_start_sh({"WEBUI_SECRET_KEY_FILE": "my secret key"})

    assert run.launch["secret_key"] == run.key_file("my secret key").read_text().strip()


@pytest.mark.parametrize("length", ["abc", "0", "-4", " ", "12.5"])
def test_a_nonsense_key_length_fails_loudly(run_start_sh, length: str) -> None:
    """Better a named error than a zero-length key signing every JWT the instance issues."""
    run = run_start_sh({"WEBUI_SECRET_KEY_LENGTH": length})

    assert run.returncode != 0, f"start.sh accepted WEBUI_SECRET_KEY_LENGTH={length!r}"
    assert "positive integer" in run.stderr, run.stderr


def test_an_empty_key_length_falls_back_to_the_default(run_start_sh) -> None:
    """A compose variable that never got a value arrives set but blank."""
    run = run_start_sh({"WEBUI_SECRET_KEY_LENGTH": ""})

    generated = run.key_file().read_text(encoding="utf-8").strip()
    assert generated, "start.sh wrote an empty key file"
    assert run.launch["secret_key"] == generated


# uvicorn arguments


def test_host_and_port_default_to_every_interface_on_8080(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "x"})

    assert (run.option("--host"), run.option("--port")) == ("0.0.0.0", "8080")


def test_host_and_port_honour_the_environment(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "x", "HOST": "127.0.0.1", "PORT": "9099"})

    assert (run.option("--host"), run.option("--port")) == ("127.0.0.1", "9099")


def test_the_default_launch_passes_the_worker_count(run_start_sh) -> None:
    run = run_start_sh({"WEBUI_SECRET_KEY": "x", "UVICORN_WORKERS": "4"})

    assert run.option("--workers") == "4"


def test_arguments_given_to_the_script_replace_the_defaults(run_start_sh) -> None:
    """`docker run ... start.sh --reload` must not also get --workers."""
    run = run_start_sh({"WEBUI_SECRET_KEY": "x"}, "--reload")

    assert run.launch["argv"][-1] == "--reload"
    assert "--workers" not in run.launch["argv"]
