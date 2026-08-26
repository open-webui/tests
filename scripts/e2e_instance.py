"""Start an isolated Open WebUI for the e2e suite and seed the accounts it expects.

The e2e tests need a running instance plus a regular user and an admin. This starts one
on its own port against a scratch data directory, so an instance you are already running
is left alone, and creates both accounts through the API.

    python scripts/e2e_instance.py --clone /path/to/open-webui

It prints the environment to export, then serves until interrupted. Run the suite from
another shell:

    OPEN_WEBUI_URL=http://localhost:8081 python -m pytest e2e -q

Nothing here writes to the checkout or to any data directory you already use. The scratch
directory is recreated on every start, so each run begins from an empty database.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "adminpassword123"
USER_EMAIL = "test@example.com"
USER_PASSWORD = "testpassword123"

LAUNCHER = """
import os, sys
from pathlib import Path

BACKEND = Path(sys.argv[1])
os.environ["DATA_DIR"] = sys.argv[2]
os.environ["STATIC_DIR"] = sys.argv[3]
os.environ["FRONTEND_BUILD_DIR"] = sys.argv[4]
os.environ["WEBUI_SECRET_KEY"] = "e2e-secret-key"
os.environ["WEBUI_URL"] = "http://localhost:" + sys.argv[5]
# socket.io derives its allowed origins from this; a mismatch surfaces as a 403 on the
# websocket handshake rather than as anything obviously wrong.
os.environ["CORS_ALLOW_ORIGIN"] = "*"
os.environ["WEBUI_AUTH"] = "true"
os.environ.setdefault("OFFLINE_MODE", "true")

sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import uvicorn
import open_webui.main  # noqa: F401  pay the import cost before the port opens

uvicorn.run(
    "open_webui.main:app",
    host="0.0.0.0",
    port=int(sys.argv[5]),
    forwarded_allow_ips="*",
    workers=1,
    # Replicates what `open-webui serve` does. A bare uvicorn default loop on Windows
    # leaves the websocket transport unwired, which shows up as chats that never stream.
    loop="none" if sys.platform == "win32" else "auto",
    log_level="warning",
)
"""


def wait_for_health(base_url: str, deadline_seconds: int = 300) -> None:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=3.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise SystemExit(f"{base_url} did not become healthy within {deadline_seconds}s")


def seed_accounts(base_url: str) -> None:
    """First signup becomes the admin; signup closes afterwards, so add the second
    account through the admin endpoint rather than a second signup."""
    admin = httpx.post(
        f"{base_url}/api/v1/auths/signup",
        json={"name": "E2E Admin", "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        timeout=60.0,
    )
    if admin.status_code != 200:
        raise SystemExit(
            f"could not create the admin account: HTTP {admin.status_code} {admin.text}"
        )

    token = admin.json()["token"]
    user = httpx.post(
        f"{base_url}/api/v1/auths/add",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "E2E User", "email": USER_EMAIL, "password": USER_PASSWORD, "role": "user"},
        timeout=60.0,
    )
    if user.status_code != 200:
        raise SystemExit(f"could not create the test user: HTTP {user.status_code} {user.text}")


def main() -> None:
    # Progress goes to a pipe when this is started in the background.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone", required=True, type=Path, help="path to the open-webui checkout")
    parser.add_argument("--port", default=8081, type=int)
    parser.add_argument("--scratch", type=Path, default=None)
    args = parser.parse_args()

    backend = args.clone / "backend"
    build = args.clone / "build"
    if not (backend / "open_webui").is_dir():
        raise SystemExit(f"{backend} does not look like an open-webui backend")
    if not build.is_dir():
        raise SystemExit(f"{build} is missing; the backend serves the built frontend from there")

    scratch = args.scratch or Path(tempfile.gettempdir()) / "owui-e2e"
    shutil.rmtree(scratch, ignore_errors=True)
    data, static = scratch / "data", scratch / "static"
    data.mkdir(parents=True)
    static.mkdir(parents=True)

    base_url = f"http://localhost:{args.port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-c",
            LAUNCHER,
            str(backend),
            str(data),
            str(static),
            str(build),
            str(args.port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        print(f"starting Open WebUI on {base_url} (first boot loads the embedding model)")
        wait_for_health(base_url)
        seed_accounts(base_url)
        print("\nready. Run the suite from another shell with:\n")
        print(f"  OPEN_WEBUI_URL={base_url} \\")
        print(f"  TEST_USER_EMAIL={USER_EMAIL} TEST_USER_PASSWORD={USER_PASSWORD} \\")
        print(f"  ADMIN_USER_EMAIL={ADMIN_EMAIL} ADMIN_USER_PASSWORD={ADMIN_PASSWORD} \\")
        print("  python -m pytest e2e -q\n")
        print("Ctrl-C to stop.")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
