"""Regression test for 0.11.2's code interpreter failing to start on Windows.

Commit `d8133c905` (PR #29139, issue #29133). `main.py` only registered `text/javascript` for
`.js`, and only when a frontend build existed. Windows hosts carry registry entries that
`mimetypes` reads at init, so `.js`, `.mjs` and `.wasm` came back as whatever the host said
(commonly `text/plain`), the browser refused the module script and the WebAssembly stream, and
Pyodide never loaded. The fix registers all three at import, over whatever the host supplied.

The host entry has to be in place before `open_webui.main` is imported, and a module is imported
once per process, so a child interpreter poisons its `mimetypes` table the way a registry entry
does, imports the app with a scratch frontend build and fetches the Pyodide assets from the app's
own `/pyodide` route.

Discriminates: passes on bbfa876af, fails with the three `mimetypes.add_type` calls removed from
`main.py` (the poisoned type is what gets served).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.regression

# What a Windows registry entry commonly turns these into.
HOST_WRONG_TYPE = "text/plain"
PYODIDE_TYPES = {
    "pyodide.js": "text/javascript",
    "pyodide.mjs": "text/javascript",
    "pyodide.asm.wasm": "application/wasm",
}
ORDINARY_TYPES = {"style.css": "text/css", "logo.png": "image/png", "data.json": "application/json"}
CONTROL_EXTENSION = ".owui-mime-probe"
PROBE_MARKER = "@@mime-probe@@"

PROBE = """
import asyncio, json, mimetypes, sys

mimetypes.guess_type("probe.js")
for extension in (".js", ".mjs", ".wasm", {control!r}):
    mimetypes.add_type({wrong!r}, extension)

sys.path.insert(0, {backend!r})
import httpx
import open_webui.main as main


async def fetch_all(names):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
        served = {{}}
        for name in names:
            response = await client.get("/pyodide/" + name)
            served[name] = {{
                "status": response.status_code,
                "content_type": response.headers.get("content-type", "").split(";")[0].strip(),
                "cors": response.headers.get("access-control-allow-origin"),
            }}
        return served


report = {{
    "served": asyncio.run(fetch_all({names!r})),
    "control": mimetypes.guess_type("file" + {control!r})[0],
}}
print({marker!r} + json.dumps(report), flush=True)
"""


@pytest.fixture(scope="module")
def probe(open_webui_backend, tmp_path_factory) -> dict:
    """What the app serves from `/pyodide` when the host's table says `text/plain`."""
    build = tmp_path_factory.mktemp("build")
    assets = build / "pyodide"
    assets.mkdir()
    names = [*PYODIDE_TYPES, *ORDINARY_TYPES]
    for name in names:
        (assets / name).write_bytes(b"x")

    body = PROBE.format(
        control=CONTROL_EXTENSION,
        wrong=HOST_WRONG_TYPE,
        backend=str(open_webui_backend),
        names=names,
        marker=PROBE_MARKER,
    )
    result = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "FRONTEND_BUILD_DIR": str(build)},
    )
    line = next((it for it in result.stdout.splitlines() if it.startswith(PROBE_MARKER)), None)
    assert line, (
        f"probe did not report (rc={result.returncode})\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    report = json.loads(line.removeprefix(PROBE_MARKER))
    unserved = {
        name: entry["status"] for name, entry in report["served"].items() if entry["status"] != 200
    }
    assert not unserved, f"/pyodide did not serve the scratch build: {unserved}; retarget the route"
    return report


def test_poisoning_the_host_table_takes_effect(probe):
    """Control: without this the tests below would prove nothing."""
    assert probe["control"] == HOST_WRONG_TYPE


@pytest.mark.parametrize("name, expected", sorted(PYODIDE_TYPES.items()))
def test_pyodide_assets_are_served_with_the_browser_safe_type(probe, name, expected):
    served = probe["served"][name]["content_type"]

    assert served == expected, (
        f"{name} was served as {served!r}, the host registry's type, so the browser refuses it "
        "and the code interpreter never loads (#29133)"
    )


@pytest.mark.parametrize("name, expected", sorted(ORDINARY_TYPES.items()))
def test_ordinary_assets_keep_their_type(probe, name, expected):
    assert probe["served"][name]["content_type"] == expected


def test_pyodide_assets_still_allow_cross_origin_loading(probe):
    assert all(entry["cors"] == "*" for entry in probe["served"].values())
