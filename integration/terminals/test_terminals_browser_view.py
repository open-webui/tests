"""Journey: a Browser tab and a file preview link of a terminal on a Terminals orchestrator show
what is in the person's own machine, without a sign-in and only while the person has the terminal.

A Browser tab opened on a template terminal belongs to the machine the terminal's scope gives the
caller: per person, per chat or per model. Its frame has no session, so the token Open WebUI
hands the page carries who opened the tab, in which chat and with which model, and a page served
on localhost inside that machine loads through it. `POST files/preview` gives a link of the same
kind to the machine's files: it reads (GET, HEAD, a preflight) the files of the person it was
made for, never writes, opens no socket, and fits neither another terminal nor a Browser tab's
path. Both stop answering when the terminal is no longer shared with the person (403) and when
the account is deleted or set to pending (401).

The Ports list and the port preview follow the same machines: a web server a person starts is
listed with its process for that person and its pages come through `proxy/<port>/` with their
root links kept inside the preview, while a second person of the same terminal, who has a
machine of their own, neither sees the port nor reaches the page.

The terminal is a real Open Terminal container started by the real Terminals service
(`terminals`, `shared_terminal`).

Discriminates: on dev c15f28649, in a backend copy: a token without the tab's chat fails the
per-chat case of the page test, one without its model the per-model case; resolving every view
as the admin in place of the token's person fails the unshared test and the own-files test (and
all but the per-model case); letting a preview link write fails the read-only test (the terminal
answers the write, not Open WebUI), as does refusing its HEAD and letting it open a socket;
dropping the session check of a token fails the fits-nothing-else test and so does dropping the
server check; a token whose account is not checked for its role fails the account test.
For the ports test: sending every person to one machine shows the neighbour the server, and a
preview that leaves the page as the terminal sent it fails on the page's link.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from harness.chat_history import MOCK_MODEL_ID, seed_chat
from harness.terminal_server import close_of, read_grant
from harness.terminals import create_template, share_template
from utils.browser_tab import serve_site

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

EXPIRED = (401, "Browser session expired")
READ_ONLY = (405, "File preview tokens are read-only")
UNSHARED = (403, "Terminal access denied")


def _signed_out(instance) -> httpx.Client:
    return httpx.Client(base_url=instance.base_url, timeout=60.0)


def _answer(response) -> tuple[int, str | None]:
    """The status and the message of a refusal, whichever key carries it."""
    try:
        body = response.json()
    except ValueError:
        return response.status_code, response.text
    return response.status_code, body.get("detail") or body.get("error")


def _run(person, terminal_id: str, command: str, context: dict | None = None) -> None:
    """Run `command` to its end in the person's machine, as they would in its shell."""
    with person.client() as client:
        done = client.post(
            f"/api/v1/terminals/{terminal_id}/execute",
            params={"wait": 30},
            headers=context or {},
            json={"command": command},
        )
    assert done.status_code == 200, f"the command failed: {done.status_code} {done.text}"
    assert done.json()["exit_code"] == 0, f"preparing the machine failed: {done.json()}"


def _open_tab(person, terminal_id: str, url: str, context: dict | None = None) -> str:
    """Open a Browser tab on `url` and return the path its frame loads pages under."""
    with person.client() as client:
        opened = client.post(
            f"/api/v1/terminals/{terminal_id}/browser/sessions",
            headers=context or {},
            json={"url": url},
        )
    assert opened.status_code == 200, f"opening the Browser tab failed: {opened.text}"
    tab = opened.json()
    return f"/api/v1/terminals/{terminal_id}/browser/view/{tab['id']}/{tab['token']}/"


def _preview_link(person, terminal_id: str) -> str:
    with person.client() as client:
        preview = client.post(f"/api/v1/terminals/{terminal_id}/files/preview")
    assert preview.status_code == 200, f"no preview link: {preview.status_code} {preview.text}"
    return preview.json()["url"]


def _write_note(person, terminal_id: str, name: str, content: str) -> None:
    _run(person, terminal_id, f"printf %s '{content}' > ~/{name}")


@pytest.mark.parametrize("scope", ["user", "chat", "model"])
def test_a_browser_tab_loads_a_page_served_in_the_machine_its_scope_gives_the_caller(
    scope, instance, make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person, scope=scope)
    tag = uuid.uuid4().hex[:8]
    port = 20000 + uuid.uuid4().int % 20000
    with person.client() as client:
        chat_id = seed_chat(client, [{"role": "user", "content": "open the page"}])[0]
    context = {
        "user": {},
        "chat": {"X-Session-Id": chat_id},
        "model": {"X-Terminal-Model-Id": MOCK_MODEL_ID},
    }[scope]
    machine = {
        "user": f"user:{person.id}",
        "chat": f"chat:{chat_id}",
        "model": f"model:{MOCK_MODEL_ID}",
    }[scope]
    _run(
        person,
        terminal_id,
        f"mkdir site-{tag} && echo '<h1>Harbour {tag}</h1>' > site-{tag}/index.html",
        context,
    )
    view = _open_tab(person, terminal_id, f"http://localhost:{port}/", context)
    page = f"{view}frame/http/localhost:{port}/"

    with _signed_out(instance) as frame:
        before = frame.get(page)
        _run(
            person,
            terminal_id,
            f"cd site-{tag} && (nohup python3 -m http.server {port} >/dev/null 2>&1 &) && "
            f"curl -s --retry 20 --retry-connrefused --retry-delay 0 -o /dev/null "
            f"http://localhost:{port}/",
            context,
        )
        loaded = frame.get(page)

    assert before.status_code == 502, (
        f"the tab's frame answered {before.status_code} before the machine served a page: "
        f"{before.text[:300]}"
    )
    assert loaded.status_code == 200 and f"<h1>Harbour {tag}</h1>" in loaded.text, (
        f"the page served in the machine did not load in the tab's frame: "
        f"{loaded.status_code} {loaded.text[:300]}"
    )
    machines = [entry["context_id"] for entry in terminals.instances()]
    assert machines == [machine], f"the tab's frame was served by {machines}, not {machine}"


def test_a_tab_and_a_preview_link_stop_once_the_terminal_is_unshared(
    admin, instance, make_user, shared_terminal
):
    kept, unshared = make_user(), make_user()
    terminal_id = shared_terminal(kept, unshared)
    tag = uuid.uuid4().hex[:8]
    links = {}
    for label, person in (("kept", kept), ("unshared", unshared)):
        _write_note(person, terminal_id, f"note-{tag}.txt", f"{label} {tag}")
        links[f"{label} tab"] = _open_tab(person, terminal_id, "http://localhost:1/") + "runtime.js"
        links[f"{label} file"] = _preview_link(person, terminal_id) + f"home/user/note-{tag}.txt"

    with _signed_out(instance) as frame:
        before = {label: frame.get(link).status_code for label, link in links.items()}
        with admin.client() as client:
            share_template(client, terminal_id, read_grant(kept.id))
        after = {label: frame.get(link) for label, link in links.items()}

    assert before == dict.fromkeys(links, 200), f"the links did not load to begin with: {before}"
    assert after["kept file"].text == f"kept {tag}", (
        f"the file link of the person who kept the terminal answered {_answer(after['kept file'])}"
    )
    assert after["kept tab"].status_code == 200, (
        f"the tab of the person who kept the terminal answered {_answer(after['kept tab'])}"
    )
    refused = {label: _answer(after[label]) for label in ("unshared tab", "unshared file")}
    assert refused == dict.fromkeys(refused, UNSHARED), (
        f"a link outlived the share of its terminal: {refused}"
    )


def test_a_preview_link_only_reads_and_opens_no_socket(
    instance, make_user, shared_terminal, terminals
):
    person = make_user()
    terminal_id = shared_terminal(person)
    tag = uuid.uuid4().hex[:8]
    name = f"note-{tag}.txt"
    _write_note(person, terminal_id, name, f"tide {tag}")
    link = _preview_link(person, terminal_id)
    file_link = f"{link}home/user/{name}"
    tab = _open_tab(person, terminal_id, "http://localhost:1/")
    as_socket = instance.base_url.replace("http://", "ws://", 1)

    with _signed_out(instance) as page:
        read = page.get(file_link)
        head = page.head(file_link)
        preflight = page.options(
            file_link, headers={"Origin": "null", "Access-Control-Request-Method": "GET"}
        )
        writes = {
            method: _answer(page.request(method, file_link, content=b"overwritten"))
            for method in ("POST", "PUT", "PATCH", "DELETE")
        }
        read_again = page.get(file_link)
    with connect(f"{as_socket}{tab}stream", proxy=None, open_timeout=10) as socket:
        tab_socket = close_of(socket, timeout=10)
    with pytest.raises(InvalidStatus) as refused:
        connect(f"{as_socket}{link}stream", proxy=None, open_timeout=10).close()

    assert (read.status_code, read.text) == (200, f"tide {tag}"), (
        f"the link did not serve the file: {read.status_code} {read.text[:300]}"
    )
    # Open Terminal 0.17 answers a HEAD on a file with its own 405, naming what it allows
    assert head.status_code == 200 or "GET" in head.headers.get("allow", ""), (
        f"the link's HEAD did not reach the terminal: {head.status_code} {dict(head.headers)}"
    )
    assert preflight.status_code == 200, (
        f"the link's preflight did not reach the terminal: {_answer(preflight)}"
    )
    assert writes == dict.fromkeys(writes, READ_ONLY), f"the link took a write: {writes}"
    assert read_again.text == f"tide {tag}", f"the file changed: {read_again.text[:300]}"
    [machine] = terminals.instances()
    assert terminals.read_file(machine["id"], name) == f"tide {tag}", (
        "the machine's file changed through a read-only link"
    )
    assert tab_socket is not None, "the tab's own socket was not ended by the terminal"
    assert refused.value.response.status_code == 403, (
        f"a file link's socket answered {refused.value.response.status_code}"
    )


def test_a_preview_link_serves_the_files_of_the_person_it_was_made_for(
    instance, make_user, shared_terminal
):
    first, second = make_user(), make_user()
    terminal_id = shared_terminal(first, second)
    tag = uuid.uuid4().hex[:8]
    name = f"note-{tag}.txt"
    _write_note(first, terminal_id, name, f"first {tag}")
    _write_note(second, terminal_id, name, f"second {tag}")
    links = {
        "first": _preview_link(first, terminal_id),
        "second": _preview_link(second, terminal_id),
    }

    with _signed_out(instance) as page:
        served = {label: page.get(f"{link}home/user/{name}").text for label, link in links.items()}

    assert served == {"first": f"first {tag}", "second": f"second {tag}"}, (
        f"the two people's links served {served}"
    )


def test_a_preview_link_fits_no_other_terminal_and_no_browser_tab(
    admin, instance, make_user, connect_terminals
):
    person = make_user()
    grant = read_grant(person.id)
    connection = connect_terminals(grant)
    with admin.client() as client:
        terminal_id = create_template(client, connection)["terminal_id"]
        other_id = create_template(client, connection)["terminal_id"]
        share_template(client, terminal_id, grant)
        share_template(client, other_id, grant)
    tag = uuid.uuid4().hex[:8]
    name = f"note-{tag}.txt"
    for target in (terminal_id, other_id):
        _write_note(person, target, name, f"tide {tag}")
    link = _preview_link(person, terminal_id)
    token = link.rstrip("/").rsplit("/", 1)[1]
    tab = _open_tab(person, terminal_id, "http://localhost:1/")
    tab_id, tab_token = tab.rstrip("/").split("/")[-2:]
    view = f"/api/v1/terminals/{terminal_id}/browser/view"

    with _signed_out(instance) as page:
        own = page.get(f"{link}home/user/{name}")
        others_own = page.get(f"{_preview_link(person, other_id)}home/user/{name}")
        runtime = page.get(f"{tab}runtime.js")
        refused = {
            "the link on another terminal": page.get(
                f"/api/v1/terminals/{other_id}/browser/view/files/{token}/home/user/{name}"
            ),
            "the link's token on a tab": page.get(f"{view}/{tab_id}/{token}/runtime.js"),
            "a tab's token on the files": page.get(f"{view}/files/{tab_token}/home/user/{name}"),
        }

    assert (own.status_code, own.text) == (200, f"tide {tag}"), f"{own.status_code} {own.text}"
    assert (others_own.status_code, others_own.text) == (200, f"tide {tag}"), (
        f"the other terminal's own link did not serve its file: {_answer(others_own)}"
    )
    assert runtime.status_code == 200, f"the tab's own token did not load: {_answer(runtime)}"
    answers = {label: _answer(response) for label, response in refused.items()}
    assert answers == dict.fromkeys(refused, EXPIRED), f"a token fitted elsewhere: {answers}"


def test_a_preview_link_stops_once_its_person_is_deleted_or_set_to_pending(
    admin, instance, make_user, shared_terminal
):
    kept, deleted, pending = make_user(), make_user(), make_user()
    terminal_id = shared_terminal(kept, deleted, pending)
    tag = uuid.uuid4().hex[:8]
    name = f"note-{tag}.txt"
    links = {}
    for label, person in (("kept", kept), ("deleted", deleted), ("pending", pending)):
        _write_note(person, terminal_id, name, f"{label} {tag}")
        links[label] = f"{_preview_link(person, terminal_id)}home/user/{name}"

    with _signed_out(instance) as page:
        before = {label: page.get(link).text for label, link in links.items()}
        with admin.client() as client:
            removing = client.delete(f"/api/v1/users/{deleted.id}")
            demoting = client.post(f"/api/v1/users/{pending.id}/update", json={"role": "pending"})
        assert removing.status_code == 200, f"deleting the account failed: {removing.text}"
        assert demoting.status_code == 200, f"demoting the account failed: {demoting.text}"
        after = {label: _answer(page.get(link)) for label, link in links.items()}

    assert before == {label: f"{label} {tag}" for label in links}, (
        f"the links did not serve their files to begin with: {before}"
    )
    assert after == {"kept": (200, f"kept {tag}"), "deleted": EXPIRED, "pending": EXPIRED}, (
        f"the links answered {after} after one account was deleted and one set to pending"
    )


def test_a_machines_web_server_is_listed_and_previewed_for_its_own_person_only(
    make_user, shared_terminal
):
    owner, neighbour = make_user(), make_user()
    terminal_id = shared_terminal(owner, neighbour)
    tag = uuid.uuid4().hex[:8]
    port = 20000 + uuid.uuid4().int % 20000
    prefix = f"/api/v1/terminals/{terminal_id}/proxy/{port}/"
    with owner.client() as client:
        serve_site(
            client,
            terminal_id,
            f"site-{tag}",
            port,
            {
                "index.html": f'<h1>Harbour {tag}</h1><a href="/quay.html">to the quay</a>',
                "quay.html": f"<h1>Quay {tag}</h1>",
            },
        )
    _run(neighbour, terminal_id, "true")

    def seen_by(person) -> tuple[list[dict], httpx.Response, httpx.Response]:
        with person.client() as client:
            listed = client.get(f"/api/v1/terminals/{terminal_id}/ports")
            assert listed.status_code == 200, f"listing the ports failed: {listed.text}"
            servers = [entry for entry in listed.json()["ports"] if entry["port"] == port]
            return servers, client.get(prefix), client.get(f"{prefix}quay.html")

    servers, page, linked = seen_by(owner)
    hidden, missing, _ = seen_by(neighbour)

    assert [entry["process"] for entry in servers] == ["python3"], (
        f"the person's own server is listed as {servers}"
    )
    assert page.status_code == 200 and f"<h1>Harbour {tag}</h1>" in page.text, (
        f"the port's page came back as {page.status_code}: {page.text[-300:]}"
    )
    assert f'href="{prefix}quay.html"' in page.text, (
        f"the page's root link does not stay inside the preview: {page.text[-300:]}"
    )
    assert linked.status_code == 200 and f"<h1>Quay {tag}</h1>" in linked.text, (
        f"the linked page came back as {linked.status_code}: {linked.text[-300:]}"
    )
    assert hidden == [], f"the neighbour's machine lists the other person's server: {hidden}"
    assert _answer(missing) == (404, "Port not found"), (
        f"the neighbour reached the other person's server: {missing.status_code} "
        f"{missing.text[-300:]}"
    )
