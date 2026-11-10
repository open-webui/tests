"""Live-update watchers on a terminal: two invariants, on every way to it.

A page with the Files tab open holds one live-update connection to its terminal and, through
it, a watcher on the folder it shows. The ways to a terminal here: straight to an Open Terminal,
through Open WebUI to an Open Terminal, straight to a Terminals orchestrator, and the whole
chain of Open WebUI, Terminals and the Open Terminal of a machine.

Invariant 1, file requests never wait behind watchers. A folder listing and the working
directory take as long with many pages watching as with none, whatever their number, and every
one of those pages is served: it is told that its folder is watched and hears of a change in
it. Open Terminal 0.17.7 breaks this from 40 watchers on (open-webui/open-terminal#230: every
file request then takes a second or more), so these tests are red until that is fixed. A fix
that only breaks later is a stopgap and stays red here: a bigger pool, a pool of the watchers'
own with a cap and a limit on watchers all give out below the 400 pages of the largest case, as
does anything that costs the host one watcher per page (the kernel allows a user 128 by
default).

Invariant 2, pages that are gone are cleaned up. However a page leaves (it closes, its
connection is cut or reset, its browser is killed, it leaves right after connecting, it goes
silent and never closes), the terminal holds a connection for each page still connected and
watches for them alone, and holds nothing once the last one has left, however many came and
went before. Nothing here allows a number of leftovers: one left per page that went is the
leak that Terminals 0.3.1 had (it kept its connection to the terminal until the next change in
the watched folder), and such leftovers are what gets a single person to the 40 of invariant 1.
How many watchers the pages of one folder share is left open, since the fix of invariant 1 may
change it: at least one while any page watches, never more than one per page.

`integration/tools/test_terminal_websocket_proxy.py` holds invariant 2 for every stream of
Open WebUI's proxy on a fake terminal, and `test_terminal_live_streams.py` for a real shell.
This module counts on the real terminal what each way to it left there, and
`e2e/terminals/test_files_tab_watchers.py` does so for the Files tab in a browser.

Discriminates: on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Invariant 2:
ending a stream only when both of its directions have ended, as Terminals 0.3.1 did, fails
every case through the hop that does it and passes the others: in a backend copy of Open WebUI
the cases through Open WebUI and through the chain (5 connections and 5 watchers left of 5
pages), in a Terminals 0.4.1 image rebuilt with that relay the cases straight to Terminals and
through the chain (the connection and watcher of the page that started the machine are still
there), each time with the two hundred pages on those ways and the silent pages. An Open
Terminal 0.17.7 that does not stop the watcher of a page that has gone fails every case
straight to it and through Open WebUI (5 watchers left of 5 pages; of the two hundred, the
leftovers use up the host's watchers and later pages are turned away). Invariant 1: red on
Open Terminal 0.17.7 as released (a listing takes 1.5 to 1.9 s beside 100 watching pages and
2 to 37 ms without; of 400 pages a host that allows 256 watchers turns away those past 164,
and a listing takes 3 s). In a patched install of 0.17.7, reached straight: with its thread
pool raised from 40 to 1000 the 100 pages pass and the 400 fail (pages turned away with 1011);
with a limit of 30 watchers both fail (pages turned away); with one watcher for each watched
folder, shared by its pages and run on a thread of its own, both pass, as do the 100 pages
through Open WebUI and every case of invariant 2 on those two ways.
"""

from __future__ import annotations

import json
import selectors
import signal
import statistics
import time
import uuid
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import httpx
import pytest

from harness.actors import Actor
from harness.open_terminal import (
    EventCounts,
    inotify_limit,
    serving_open_terminal,
    wait_for_event_counts,
)
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    BarePage,
    configure_terminals,
    pages_of_another_process,
    read_grant,
    stream_url,
)
from harness.terminals import create_template, share_template

pytestmark = [pytest.mark.api, pytest.mark.requires_source]

OPEN_TERMINAL = "open-terminal"
OPEN_WEBUI = "open-webui"
TERMINALS = "terminals"
CHAIN = "open-webui-and-terminals"
WAYS = [OPEN_TERMINAL, OPEN_WEBUI, TERMINALS, CHAIN]

NOBODY = EventCounts(connections=0, watchers=0)
READY = {"type": "fs:ready"}
CHANGED = {"type": "fs:change"}
RELEASE_BOUND_SECONDS = 5
# a server gives up a page that answers no keep-alive ping after two 20 s periods
SILENT_BOUND_SECONDS = 60
QUIET_BOUND_SECONDS = 15
SERVED_BOUND_SECONDS = 10
MACHINE_START_SECONDS = 60
MACHINE_HOME = "/home/user"
STAYING = 2
LEAVING = 5
WAVE = 25
WAVES = 8


@dataclass
class Way:
    """A terminal and one way to it, for the pages that watch its home folder."""

    name: str
    url: str
    sign_in: dict
    headers: dict[str, str]
    folder: str
    counts: Callable[[], EventCounts]
    change_a_file: Callable[[], None]
    # the terminal's file requests on the same way
    files: httpx.Client

    @property
    def subscription(self) -> str:
        return json.dumps({"type": "subscribe", "path": self.folder})


@pytest.fixture
def way_to(request, admin, make_user, preserve) -> Iterator[Callable[[str], Way]]:
    """`way_to(name)`: a terminal of its own reached that way, by a person of its own."""
    clients: list[httpx.Client] = []
    terminals_of_the_test = ExitStack()

    def file_requests(base_url: str, token: str, **headers: str) -> httpx.Client:
        # no connection is kept between requests, so that only watching pages hold any
        headers = {"Authorization": f"Bearer {token}", "Connection": "close", **headers}
        clients.append(httpx.Client(base_url=base_url, headers=headers, timeout=30))
        return clients[-1]

    def connected(connection: dict) -> dict:
        preserve(TERMINAL_SERVERS_CONFIG)
        with admin.client() as client:
            saved = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
            configure_terminals(client, *saved, connection)
        return connection

    def through_open_webui(person: Actor, terminal_id: str) -> tuple[str, dict, httpx.Client]:
        files = file_requests(f"{person.base_url}/api/v1/terminals/{terminal_id}/", person.token)
        sign_in = {"type": "auth", "token": person.token}
        return stream_url(person.base_url, terminal_id, "events"), sign_in, files

    def an_open_terminal(name: str) -> Way:
        person = make_user()
        # the session's Open Terminal is shared: here only the pages of the test may connect
        terminal = terminals_of_the_test.enter_context(serving_open_terminal())
        if name == OPEN_WEBUI:
            grants = {"access_grants": [read_grant(person.id)]}
            connection = connected(terminal.connection(config=grants))
            url, sign_in, files = through_open_webui(person, connection["id"])
        else:
            url = terminal.base_url.replace("http://", "ws://", 1) + "/events"
            sign_in = {"type": "auth", "token": terminal.api_key}
            files = file_requests(terminal.base_url, terminal.api_key)
        return Way(
            name=name,
            url=url,
            sign_in=sign_in,
            headers={},
            folder=str(terminal.home),
            counts=terminal.event_counts,
            change_a_file=lambda: _make_a_file(terminal.home),
            files=files,
        )

    def a_machine(name: str) -> Way:
        person = make_user()
        terminals = request.getfixturevalue("terminals")
        grant = read_grant(person.id)
        connection = connected(terminals.connection(config={"access_grants": [grant]}))
        with admin.client() as client:
            terminal_id = create_template(client, connection)["terminal_id"]
            share_template(client, terminal_id, grant)
        # what Open WebUI names the person and their terminal by to the orchestrator
        named = {"X-User-Id": person.id, "X-Terminal-Template-Id": terminal_id.split(":")[2]}
        directly = terminals.base_url.replace("http://", "ws://", 1) + "/events"
        key = {"type": "auth", "token": terminals.api_key}
        # the person's machine starts with the first page that watches it
        first = BarePage(directly, key, named)
        first.send(json.dumps({"type": "subscribe", "path": MACHINE_HOME}))
        refused = _served([first], READY, MACHINE_START_SECONDS)
        assert not refused, f"the machine did not start: {refused}"
        first.close()
        if name == CHAIN:
            headers = {}
            url, sign_in, files = through_open_webui(person, terminal_id)
        else:
            headers, url, sign_in = named, directly, key
            files = file_requests(terminals.base_url, terminals.api_key, **named)
        machine = next(
            machine["id"]
            for machine in terminals.instances()
            if machine["context_id"] == f"user:{person.id}"
        )
        return Way(
            name=name,
            url=url,
            sign_in=sign_in,
            headers=headers,
            folder=MACHINE_HOME,
            counts=lambda: terminals.event_counts(machine),
            change_a_file=lambda: terminals.touch_file(machine, f"made-{uuid.uuid4().hex[:8]}"),
            files=files,
        )

    def way(name: str) -> Way:
        made = a_machine(name) if name in (TERMINALS, CHAIN) else an_open_terminal(name)
        quiet = wait_for_event_counts(made.counts, NOBODY, QUIET_BOUND_SECONDS)
        assert quiet == NOBODY, (
            f"with no page connected through {name} (a machine was started by a page that has "
            f"closed since), its terminal held {quiet}"
        )
        return made

    with terminals_of_the_test:
        yield way
        for client in clients:
            client.close()


def _make_a_file(folder: Path) -> None:
    (folder / f"made-{uuid.uuid4().hex[:8]}").write_text("")


def _served(pages: list[BarePage], expected: dict, bound: float) -> dict[int, str]:
    """Read `expected` as the next message of each page; what the others got instead, by the
    page's place in the list. All of them share the bound, however many they are."""
    waiting = selectors.DefaultSelector()
    unserved = dict.fromkeys(range(len(pages)), f"nothing within {bound}s")

    def read(place: int, patience: float) -> None:
        page = pages[place]
        try:
            message = page.receive(timeout=patience)
        except TimeoutError:
            # nothing yet, or a keep-alive ping, which reading it has answered
            return
        except ConnectionError:
            message = f"closed with {page.closed_with}"
        waiting.unregister(page.socket)
        if _is(message, expected):
            del unserved[place]
        else:
            unserved[place] = str(message)

    for place, page in enumerate(pages):
        waiting.register(page.socket, selectors.EVENT_READ, place)
        # a message that an earlier read brought along is not announced by the socket
        read(place, 0)
    deadline = time.monotonic() + bound
    while waiting.get_map() and (remaining := deadline - time.monotonic()) > 0:
        for key, _ in waiting.select(remaining):
            read(key.data, 0.01)
    waiting.close()
    return unserved


def _is(message: str | bytes, expected: dict) -> bool:
    try:
        return json.loads(message) == expected
    except ValueError:
        return False


def _watching(way: Way, count: int) -> list[BarePage]:
    """`count` pages that watch the terminal's folder and were told so."""
    pages = [BarePage(way.url, way.sign_in, way.headers) for _ in range(count)]
    for page in pages:
        page.send(way.subscription)
    unserved = _served(pages, READY, SERVED_BOUND_SECONDS)
    assert not unserved, (
        f"{len(unserved)} of {count} pages were not told that their folder is watched "
        f"through {way.name}: {sorted(set(unserved.values()))}"
    )
    return pages


def _stay_connected(pages: list[BarePage]) -> None:
    """Answer the keep-alive pings the pages got: a page answers one only when it reads."""
    _served(pages, CHANGED, 0.05)


HOW_A_PAGE_LEAVES = {
    "closes": BarePage.close,
    "is-cut-off": BarePage.cut,
    "is-reset": BarePage.reset,
}
LEAVING_WAYS = [*HOW_A_PAGE_LEAVES, "is-killed", "leaves-at-once"]


def _come_and_go(way: Way, leaving: str, count: int) -> None:
    """`count` pages watch the terminal's folder and leave again."""
    if leaving == "is-killed":
        pages = [(way.url, way.sign_in)] * count
        with pages_of_another_process(
            pages, then=(way.subscription,), headers=way.headers
        ) as browser:
            browser.kill()
        return
    if leaving == "leaves-at-once":
        for number in range(count):
            page = BarePage(way.url, way.sign_in, way.headers)
            # every other page is gone before it asked to watch anything
            if number % 2:
                page.send(way.subscription)
            leave = page.reset if number % 4 < 2 else page.cut
            leave()
        return
    for page in _watching(way, count):
        HOW_A_PAGE_LEAVES[leaving](page)


def _of_pages(pages: int) -> Callable[[EventCounts], bool]:
    """Whether a terminal holds what `pages` pages watching one folder account for: a connection
    each, and for the folder one watcher at least and one for each page at most."""

    def accounted_for(counts: EventCounts) -> bool:
        return counts.connections == pages and min(pages, 1) <= counts.watchers <= pages

    return accounted_for


def _left_behind(counts: EventCounts, expected: EventCounts) -> str:
    return (
        f"{counts.connections - expected.connections} connections and "
        f"{counts.watchers - expected.watchers} watchers"
    )


@pytest.mark.journey
@pytest.mark.parametrize("leaving", LEAVING_WAYS)
@pytest.mark.parametrize("through", WAYS)
def test_a_terminal_holds_the_watchers_of_the_pages_still_connected_and_no_others(
    way_to, through, leaving
):
    """A higher limit is a stopgap: the count must return to baseline."""
    way = way_to(through)

    staying = _watching(way, STAYING)
    the_stayers = wait_for_event_counts(way.counts, _of_pages(STAYING), RELEASE_BOUND_SECONDS)
    _come_and_go(way, leaving, LEAVING)
    after_the_leavers = wait_for_event_counts(way.counts, the_stayers, RELEASE_BOUND_SECONDS)
    way.change_a_file()
    unheard = _served(staying, CHANGED, SERVED_BOUND_SECONDS)
    for page in staying:
        page.close()
    at_the_end = wait_for_event_counts(way.counts, NOBODY, RELEASE_BOUND_SECONDS)

    assert _of_pages(STAYING)(the_stayers), (
        f"with {STAYING} pages watching through {through} the terminal held {the_stayers}"
    )
    assert after_the_leavers == the_stayers, (
        f"{RELEASE_BOUND_SECONDS}s after {LEAVING} pages had come and gone ({leaving}) through "
        f"{through}, {_left_behind(after_the_leavers, the_stayers)} of theirs were still on "
        f"the terminal beside those of the {STAYING} pages that stayed"
    )
    assert not unheard, f"the pages that stayed did not hear of a change in their folder: {unheard}"
    assert at_the_end == NOBODY, (
        f"{RELEASE_BOUND_SECONDS}s after the last page had closed, the terminal still held "
        f"{at_the_end} through {through}"
    )


@pytest.mark.journey
@pytest.mark.slow
@pytest.mark.parametrize("through", WAYS)
def test_a_terminal_holds_no_more_after_two_hundred_pages_came_and_went(way_to, through):
    """A higher limit is a stopgap: the count must return to baseline."""
    way = way_to(through)

    staying = _watching(way, STAYING)
    the_stayers = wait_for_event_counts(way.counts, _of_pages(STAYING), RELEASE_BOUND_SECONDS)
    left_behind = {}
    for wave in range(WAVES):
        leaving = LEAVING_WAYS[wave % len(LEAVING_WAYS)]
        _come_and_go(way, leaving, WAVE)
        _stay_connected(staying)
        held = wait_for_event_counts(way.counts, the_stayers, RELEASE_BOUND_SECONDS)
        if held != the_stayers:
            left_behind[f"wave {wave + 1} ({leaving})"] = _left_behind(held, the_stayers)
    way.change_a_file()
    unheard = _served(staying, CHANGED, SERVED_BOUND_SECONDS)
    for page in staying:
        page.close()
    at_the_end = wait_for_event_counts(way.counts, NOBODY, RELEASE_BOUND_SECONDS)

    assert _of_pages(STAYING)(the_stayers), (
        f"with {STAYING} pages watching through {through} the terminal held {the_stayers}"
    )
    assert not left_behind, (
        f"of {WAVES} waves of {WAVE} pages that came and went through {through}, these were "
        f"still on the terminal {RELEASE_BOUND_SECONDS}s after their wave: {left_behind}"
    )
    assert not unheard, f"the pages that stayed did not hear of a change in their folder: {unheard}"
    assert at_the_end == NOBODY, (
        f"after {WAVES * WAVE} pages and the last one closed, the terminal still held "
        f"{at_the_end} through {through}"
    )


@pytest.mark.journey
@pytest.mark.slow
def test_pages_that_go_silent_are_let_go_of_on_every_way_to_a_terminal(way_to):
    """A higher limit is a stopgap: the count must return to baseline."""
    ways = [way_to(name) for name in WAYS]

    with ExitStack() as browsers:
        for way in ways:
            browser = browsers.enter_context(
                pages_of_another_process(
                    [(way.url, way.sign_in)] * LEAVING,
                    then=(way.subscription,),
                    headers=way.headers,
                )
            )
            # the pages stay open and answer nothing from here on
            browser.send_signal(signal.SIGSTOP)
        while_open = {
            way.name: wait_for_event_counts(way.counts, _of_pages(LEAVING), RELEASE_BOUND_SECONDS)
            for way in ways
        }
        deadline = time.monotonic() + SILENT_BOUND_SECONDS
        still_held = {
            way.name: wait_for_event_counts(way.counts, NOBODY, deadline - time.monotonic())
            for way in ways
        }

    assert all(map(_of_pages(LEAVING), while_open.values())), (
        f"with {LEAVING} pages watching on each way, the terminals held {while_open}"
    )
    assert still_held == dict.fromkeys(WAYS, NOBODY), (
        f"{SILENT_BOUND_SECONDS}s after their pages had gone silent, the terminals still held "
        f"{still_held}"
    )


# invariant 1

SAMPLES = 9
FEWEST_SAMPLES = 3
SAMPLING_SECONDS = 8
FILE_REQUESTS = {
    "a folder listing": ("files/list", "directory"),
    "the working directory": ("files/cwd", None),
}


def _file_request_seconds(way: Way) -> dict[str, float]:
    """The median time of each file request the Files tab makes, asked for on the same way."""
    times: dict[str, list[float]] = {name: [] for name in FILE_REQUESTS}
    started_sampling = time.monotonic()
    for sample in range(SAMPLES):
        # starved requests take seconds each: a few of them are enough to show it
        out_of_time = time.monotonic() - started_sampling > SAMPLING_SECONDS
        if sample >= FEWEST_SAMPLES and out_of_time:
            break
        for name, (path, folder_parameter) in FILE_REQUESTS.items():
            query = {folder_parameter: way.folder} if folder_parameter else {}
            started = time.monotonic()
            answer = way.files.get(path, params=query)
            times[name].append(time.monotonic() - started)
            assert answer.status_code == 200, f"{name} failed: {answer.text}"
    return {name: statistics.median(taken) for name, taken in times.items()}


def _as_fast_as(alone: float) -> float:
    """What a request may take beside watchers: its time without any, with room for jitter."""
    return max(10 * alone, alone + 0.2)


@pytest.mark.regression
@pytest.mark.parametrize(
    ("through", "crowd"),
    [
        (OPEN_TERMINAL, 100),
        (OPEN_WEBUI, 100),
        (CHAIN, 100),
        pytest.param(OPEN_TERMINAL, 400, marks=pytest.mark.slow),
    ],
)
def test_file_requests_are_as_fast_with_many_pages_watching_as_with_none(way_to, through, crowd):
    """Watchers must not be able to starve file requests, however many there are
    (open-webui/open-terminal#230).

    A higher limit is a stopgap: the count must return to baseline.
    """
    way = way_to(through)

    alone = _file_request_seconds(way)
    watching: list[BarePage] = []
    turned_away: dict[int, str] = {}
    try:
        while len(watching) < crowd and not turned_away:
            wave = [BarePage(way.url, way.sign_in, way.headers) for _ in range(WAVE)]
            watching += wave
            for page in wave:
                page.send(way.subscription)
            _stay_connected(watching[: -len(wave)])
            turned_away = _served(wave, READY, SERVED_BOUND_SECONDS)
        unserved = [wave[place] for place in turned_away]
        served = [page for page in watching if page not in unserved]
        _stay_connected(served)
        crowded = _file_request_seconds(way)
        way.change_a_file()
        unheard = _served(served, CHANGED, SERVED_BOUND_SECONDS)
    finally:
        for page in watching:
            # a page that was turned away is closed already
            with suppress(OSError):
                page.cut()

    slower = {
        name: f"{crowded[name] * 1000:.0f} ms, {alone[name] * 1000:.0f} ms with none"
        for name in FILE_REQUESTS
        if crowded[name] > _as_fast_as(alone[name])
    }
    assert not slower, (
        f"with {len(served)} pages watching a folder through {through}, file requests waited: "
        f"{slower}"
    )
    assert not turned_away, (
        f"of {crowd} pages watching one folder through {through}, page {len(watching) - WAVE + 1} "
        f"to {len(watching)} were asked for and {len(turned_away)} of them were not served: "
        f"{sorted(set(turned_away.values()))} (the host allows a user {inotify_limit()} "
        f"watchers)"
    )
    assert not unheard, (
        f"{len(unheard)} of {len(served)} watching pages did not hear of a change in their "
        f"folder within {SERVED_BOUND_SECONDS}s: {sorted(set(unheard.values()))}"
    )
