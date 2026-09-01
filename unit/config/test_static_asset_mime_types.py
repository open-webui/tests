"""Regression test for 0.11.2's code interpreter failing to start on Windows.

Commit `d8133c905` (PR #29139, issue #29133). `main.py` only registered `text/javascript` for
`.js`, and only inside the `FRONTEND_BUILD_DIR` branch. Windows hosts carry registry entries
that `mimetypes` reads at init, so `.js`, `.mjs` and `.wasm` came back as whatever the host said
(commonly `text/plain`), the browser refused the module script and the WebAssembly stream, and
Pyodide never loaded. The fix registers all three unconditionally at import, which overrides
whatever the host registry supplied.

The poisoning has to be in place before `open_webui.main` is imported, and a module is imported
once per process, so the probe runs in a child interpreter. Doing it in-process would make the
whole file depend on no earlier test having imported `main` first.

Discriminates: passes on v0.11.3, fails on v0.11.1 (nothing overrides the host entry for `.js`,
`.mjs` or `.wasm`, so the poisoned type is what gets served).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.regression

# What a Windows registry entry commonly turns these into.
HOST_WRONG_TYPE = 'text/plain'
EXPECTED_TYPES = {'.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm'}
POISONED_EXTENSIONS = tuple(EXPECTED_TYPES)
CONTROL_EXTENSION = '.owui-mime-probe'
ORDINARY_ASSETS = {
    'style.css': 'text/css',
    'logo.png': 'image/png',
    'data.json': 'application/json',
}
UNKNOWN_ASSET = 'notes.qqq'

PROBE_MARKER = '@@mime-probe@@'

# Poisons the live mimetypes table before importing main, which is where a registry entry sits.
PROBE = '''
import asyncio, json, mimetypes, os, sys

mimetypes.guess_type("probe.js")
for extension in {poisoned!r}:
    mimetypes.add_type({wrong!r}, extension)

sys.path.insert(0, {backend!r})
import open_webui.main as main

assets = {assets!r}
scope = {{"type": "http", "method": "GET", "headers": []}}


async def serve(name):
    files = main.CORSStaticFiles(directory=assets)
    response = await files.get_response(name, scope)
    return {{
        "content_type": response.headers["content-type"].split(";")[0].strip(),
        "cors": response.headers.get("Access-Control-Allow-Origin"),
    }}


payload = {{
    "served": {{name: asyncio.run(serve(name)) for name in sorted(os.listdir(assets))}},
    "guessed": {{ext: mimetypes.guess_type("pyodide" + ext)[0] for ext in {probed!r}}},
}}
sys.stdout.write({marker!r} + json.dumps(payload) + "\\n")
sys.stdout.flush()
os._exit(0)
'''


@pytest.fixture(scope='session')
def probe(open_webui_backend, tmp_path_factory):
    """Types the real `CORSStaticFiles` serves from a `main` imported over a poisoned table."""
    assets = tmp_path_factory.mktemp('assets')
    names = [f'pyodide{ext}' for ext in EXPECTED_TYPES] + [*ORDINARY_ASSETS, UNKNOWN_ASSET]
    for name in names:
        (assets / name).write_bytes(b'x')

    body = PROBE.format(
        poisoned=(*POISONED_EXTENSIONS, CONTROL_EXTENSION),
        wrong=HOST_WRONG_TYPE,
        backend=str(open_webui_backend),
        assets=str(assets),
        probed=(*POISONED_EXTENSIONS, CONTROL_EXTENSION),
        marker=PROBE_MARKER,
    )
    result = subprocess.run(
        [sys.executable, '-c', body],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    )
    line = next((it for it in result.stdout.splitlines() if it.startswith(PROBE_MARKER)), None)
    assert line, (
        f'probe did not report (rc={result.returncode})\n'
        f'{result.stdout[-2000:]}\n{result.stderr[-2000:]}'
    )
    return json.loads(line[len(PROBE_MARKER) :])


# -----------------------------------------------------------------------------
# Narrow: the served type comes from the app, not from the host registry
# -----------------------------------------------------------------------------


def test_poisoning_the_registry_actually_takes_effect(probe):
    """Control: without this the narrow tests below would prove nothing."""
    assert probe['guessed'][CONTROL_EXTENSION] == HOST_WRONG_TYPE


@pytest.mark.parametrize('extension', POISONED_EXTENSIONS)
def test_poisoned_asset_is_served_with_the_browser_safe_type(probe, extension):
    served = probe['served'][f'pyodide{extension}']['content_type']
    assert served == EXPECTED_TYPES[extension]
    assert served != HOST_WRONG_TYPE


@pytest.mark.parametrize('extension', POISONED_EXTENSIONS)
def test_poisoned_extension_is_overridden_in_the_mimetypes_table(probe, extension):
    assert probe['guessed'][extension] == EXPECTED_TYPES[extension]


# -----------------------------------------------------------------------------
# Broad: no code-interpreter asset is served as something a browser will reject
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('extension, expected', sorted(EXPECTED_TYPES.items()))
def test_every_pyodide_asset_type_is_registered(probe, extension, expected):
    assert probe['served'][f'pyodide{extension}']['content_type'] == expected


# -----------------------------------------------------------------------------
# Nearby: unchanged on both refs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('name, expected', sorted(ORDINARY_ASSETS.items()))
def test_ordinary_static_assets_keep_their_type(probe, name, expected):
    assert probe['served'][name]['content_type'] == expected


def test_unknown_extension_falls_back_to_the_starlette_default(probe):
    assert probe['served'][UNKNOWN_ASSET]['content_type'] == 'application/octet-stream'


def test_cors_header_is_still_set_on_served_assets(probe):
    assert probe['served']['pyodide.mjs']['cors'] == '*'
