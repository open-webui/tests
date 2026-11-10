"""Journey: what only a full Open Terminal has works through Open WebUI: notebook cells, a
Chrome tab and the desktop.

The slim image and the PyPI package report `notebooks` and `desktop` off and have no Chromium.
The full image has all three, and Open WebUI only passes the calls on, so each is checked by
what comes back through it. A notebook the person opens runs its cells on a kernel and the
output is in the answer and in the saved file; the session belongs to that person, so another
person with the same terminal gets 404 for it. A Browser tab switched to Chrome mode loads a
page served inside the terminal: its accessibility snapshot names the page's heading, and the
viewer's stream, opened with the token Open WebUI issued for the tab, sends the page's frames.
The desktop reports a 1280 by 800 VNC screen with sound, its screenshot is a PNG of that size,
`desktop/stream` starts with the VNC greeting and `desktop/audio` with a WebM header. On the
Terminals orchestrator a terminal's image decides the same: a template on the full image
reports notebooks and a desktop and runs a cell, one on the default image reports neither.

The terminal is the full image in a container (`full_open_terminal`, `full_terminal_image`);
the tests skip where `OPEN_TERMINAL_FULL_IMAGE` is not set.

Discriminates: on dev c15f28649, in a backend copy: refusing the `notebooks` path like the
terminal's admin paths fails the notebook test and the template test; sending every caller to
the terminal under one user id fails the notebook test (the other person reaches the session)
and the Chrome test; handing the page the terminal's own tab token fails the Chrome test;
relaying `desktop/stream` and `desktop/audio` to the events stream fails the desktop test.
"""

from __future__ import annotations

import json
import struct
import uuid

import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from harness.backends import wait_until
from harness.chat_history import seed_chat
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    close_of,
    configure_terminals,
    read_grant,
    terminal_socket,
)
from harness.terminals import create_template

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VNC_GREETING = b"RFB 003.008\n"
WEBM_HEADER = b"\x1a\x45\xdf\xa3"


def _notebook(*sources: str) -> str:
    cells = [
        {
            "cell_type": "code",
            "source": source,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }
        for source in sources
    ]
    kernel = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return json.dumps(
        {"cells": cells, "metadata": {"kernelspec": kernel}, "nbformat": 4, "nbformat_minor": 5}
    )


def _printed(executed) -> str:
    assert executed.status_code == 200, f"running the cell failed: {executed.text}"
    return "".join(output.get("text", "") for output in executed.json()["outputs"])


@pytest.fixture
def full_terminal(full_open_terminal, admin, preserve):
    """`full_terminal(*people)`: the id of the full terminal, shared with those people."""

    def share(*people) -> str:
        preserve(TERMINAL_SERVERS_CONFIG)
        grants = [read_grant(person.id) for person in people]
        connection = full_open_terminal.connection(config={"access_grants": grants})
        with admin.client() as client:
            configure_terminals(client, connection)
        return connection["id"]

    return share


def test_a_notebook_cell_runs_on_the_terminal_for_the_person_who_opened_it(
    full_terminal, make_user
):
    person, other = make_user(), make_user()
    terminal = f"/api/v1/terminals/{full_terminal(person, other)}"
    tag = uuid.uuid4().hex[:8]
    path = f"notebooks-{tag}/sums.ipynb"
    with person.client() as client:
        written = client.post(
            f"{terminal}/files/write",
            json={"path": path, "content": _notebook(f"print('{tag}', 6 * 7)", "1 / 0")},
        )
        assert written.status_code == 200, f"writing the notebook failed: {written.text}"
        opened = client.post(f"{terminal}/notebooks", json={"path": path})
        assert opened.status_code == 200, f"opening the notebook failed: {opened.text}"
        session = opened.json()["id"]
        ran = client.post(f"{terminal}/notebooks/{session}/execute", json={"cell_index": 0})
        failed = client.post(f"{terminal}/notebooks/{session}/execute", json={"cell_index": 1})
        saved = client.get(f"{terminal}/files/read", params={"path": path})
        with other.client() as stranger:
            foreign = stranger.post(
                f"{terminal}/notebooks/{session}/execute", json={"cell_index": 0}
            )
        closed = client.delete(f"{terminal}/notebooks/{session}")
        after = client.post(f"{terminal}/notebooks/{session}/execute", json={"cell_index": 0})

    assert _printed(ran) == f"{tag} 42\n", f"the cell printed {ran.text}"
    assert ran.json()["status"] == "ok", ran.text
    assert failed.status_code == 200 and failed.json()["status"] == "error", failed.text
    assert "ZeroDivisionError" in failed.text, f"the failing cell answered {failed.text}"
    assert saved.status_code == 200, saved.text
    assert f"{tag} 42" in saved.text, "the saved notebook does not hold the cell's output"
    assert foreign.status_code == 404, (
        f"another person ran a cell in the session: {foreign.status_code} {foreign.text}"
    )
    assert closed.status_code == 200, closed.text
    assert after.status_code == 404, f"a closed notebook still runs cells: {after.text}"


def _closed(socket) -> int | None:
    closed = close_of(socket, timeout=10)
    return closed[0] if closed else None


def test_a_browser_tab_in_chrome_mode_shows_a_page_served_in_the_terminal(
    full_terminal, full_open_terminal, make_user
):
    person = make_user()
    terminal = f"/api/v1/terminals/{full_terminal(person)}"
    tag = uuid.uuid4().hex[:8]
    port = 20000 + uuid.uuid4().int % 20000
    page = f"<html><head><title>Harbour {tag}</title></head><body><h1>Keeper {tag}</h1></body>"
    with person.client() as client:
        chat = {
            "X-Session-Id": seed_chat(client, [{"role": "user", "content": "open the page"}])[0]
        }
        written = client.post(
            f"{terminal}/files/write", json={"path": f"site-{tag}/index.html", "content": page}
        )
        assert written.status_code == 200, f"writing the page failed: {written.text}"
        serving = client.post(
            f"{terminal}/execute",
            params={"wait": 1},
            headers=chat,
            json={"command": f"cd site-{tag} && exec python3 -m http.server {port}"},
        )
        assert serving.status_code == 200, f"starting the page's server failed: {serving.text}"
        opened = client.post(
            f"{terminal}/browser/sessions", headers=chat, json={"url": f"http://localhost:{port}/"}
        )
        assert opened.status_code == 200, f"opening the Browser tab failed: {opened.text}"
        tab = opened.json()
        switched = client.patch(
            f"{terminal}/browser/sessions/{tab['id']}", headers=chat, json={"mode": "chrome"}
        )
        assert switched.status_code == 200, f"switching to Chrome mode failed: {switched.text}"
        snapshot = client.get(
            f"{terminal}/browser/snapshot", headers=chat, params={"session_id": tab["id"]}
        )
        with full_open_terminal.client() as direct:
            held = direct.get(
                f"/browser/sessions/{tab['id']}", headers={"X-User-Id": person.id, **chat}
            )
        assert held.status_code == 200, f"the terminal does not hold the tab: {held.text}"
        view = (
            f"{person.base_url.replace('http://', 'ws://', 1)}{terminal}/browser/view/{tab['id']}"
        )
        with connect(f"{view}/{switched.json()['token']}/stream", proxy=None) as viewer:
            state = json.loads(viewer.recv(timeout=15))
        with connect(f"{view}/{switched.json()['token']}/vnc", proxy=None) as screen:
            greeting = screen.recv(timeout=15)
        with pytest.raises(InvalidStatus) as refused:
            connect(f"{view}/{held.json()['token']}/stream", proxy=None).close()
        closed = client.delete(f"{terminal}/browser/sessions/{tab['id']}", headers=chat)

    assert (tab["mode"], tab["chrome_available"]) == ("proxy", True), f"the new tab is {tab}"
    assert switched.json()["mode"] == "chrome", switched.text
    assert snapshot.status_code == 200, f"the tab's snapshot failed: {snapshot.text}"
    assert snapshot.json()["title"] == f"Harbour {tag}", snapshot.text
    assert f'heading "Keeper {tag}"' in snapshot.json()["snapshot"], (
        f"Chrome did not render the page's heading: {snapshot.text}"
    )
    assert switched.json()["token"] != held.json()["token"], (
        "the page was handed the terminal's own token for the tab"
    )
    assert (state["type"], state["title"], state["url"]) == (
        "state",
        f"Harbour {tag}",
        f"http://localhost:{port}/",
    ), f"the viewer's stream opened with {state}"
    assert greeting == VNC_GREETING, f"the tab's screen opened with {greeting!r}"
    assert refused.value.response.status_code == 403, (
        f"the terminal's own token opened the viewer: {refused.value}"
    )
    assert closed.status_code == 200, closed.text


def test_the_desktop_is_a_vnc_screen_with_sound(full_terminal, make_user):
    person, stranger = make_user(), make_user()
    terminal_id = full_terminal(person)
    terminal = f"/api/v1/terminals/{terminal_id}"
    with person.client() as client:
        desktop = client.get(f"{terminal}/desktop")
        screenshot = client.get(f"{terminal}/desktop/screenshot")
    with terminal_socket(person.base_url, person.token, terminal_id, "desktop/stream") as screen:
        greeting = screen.recv(timeout=15)
    with terminal_socket(person.base_url, person.token, terminal_id, "desktop/audio") as sound:
        audio = sound.recv(timeout=15)
    with terminal_socket(
        stranger.base_url, stranger.token, terminal_id, "desktop/stream"
    ) as screen:
        refused = _closed(screen)

    assert desktop.status_code == 200, f"the desktop did not start: {desktop.text}"
    assert desktop.json() == {
        "width": 1280,
        "height": 800,
        "transport": "vnc",
        "platform": "linux",
        "audio": True,
    }, desktop.text
    assert screenshot.status_code == 200, screenshot.text
    assert screenshot.headers["content-type"] == "image/png"
    assert screenshot.content[:8] == PNG_SIGNATURE, screenshot.content[:16]
    assert struct.unpack("!II", screenshot.content[16:24]) == (1280, 800), (
        "the screenshot is not the size of the desktop"
    )
    assert greeting == VNC_GREETING, f"the desktop stream opened with {greeting!r}"
    assert isinstance(audio, bytes) and audio[:4] == WEBM_HEADER, (
        f"the desktop sound opened with {audio[:16]!r}"
    )
    assert refused == 4004, f"a person without the terminal got its desktop: closed {refused}"


def test_a_terminals_image_decides_whether_it_has_notebooks_and_a_desktop(
    full_terminal_image, admin, connect_terminals, terminals
):
    connection = connect_terminals()
    tag = uuid.uuid4().hex[:8]
    path = f"sums-{tag}.ipynb"
    with admin.client() as client:
        full = create_template(client, connection, image=full_terminal_image, memory_limit="1Gi")[
            "terminal_id"
        ]
        slim = create_template(client, connection)["terminal_id"]
        features = {}
        for name, terminal_id in (("full", full), ("slim", slim)):
            assert wait_until(
                lambda: client.get(f"/api/v1/terminals/{terminal_id}/files/cwd").status_code == 200,
                30,
            ), f"the {name} terminal's machine did not start:\n{terminals.logs()[-2000:]}"
            config = client.get(f"/api/v1/terminals/{terminal_id}/api/config")
            assert config.status_code == 200, config.text
            features[name] = config.json()["features"]
        written = client.post(
            f"/api/v1/terminals/{full}/files/write",
            json={"path": path, "content": _notebook(f"print('{tag}', 6 * 7)")},
        )
        assert written.status_code == 200, f"writing the notebook failed: {written.text}"
        opened = client.post(f"/api/v1/terminals/{full}/notebooks", json={"path": path})
        assert opened.status_code == 200, f"opening the notebook failed: {opened.text}"
        ran = client.post(
            f"/api/v1/terminals/{full}/notebooks/{opened.json()['id']}/execute",
            json={"cell_index": 0},
        )
        missing = client.post(f"/api/v1/terminals/{slim}/notebooks", json={"path": path})

    assert (features["full"]["notebooks"], features["full"]["desktop"]) == (True, True), (
        f"the terminal on the full image reports {features['full']}"
    )
    assert (features["slim"]["notebooks"], features["slim"]["desktop"]) == (False, False), (
        f"the terminal on the default image reports {features['slim']}"
    )
    assert _printed(ran) == f"{tag} 42\n", f"the cell printed {ran.text}"
    assert missing.status_code == 404, (
        f"the default image opened a notebook: {missing.status_code} {missing.text}"
    )
