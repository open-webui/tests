"""Journey: the terminal file browser's saves, moves and deletes reach a real Open Terminal.

The file browser in a chat changes files through the terminal proxy: Save uploads the whole file
into its folder, dragging a file onto a folder moves it, and Delete removes it. A person with a
grant on the connection does each of these the way the browser does and the Open Terminal's disk
changes to match, a saved workbook of several megabytes arriving byte for byte. A person without
a grant gets 403 for each, and the disk stays as it was.

Discriminates: passes on dev f659c0d12 (three runs). In a backend copy whose terminal proxy
forwards an empty body and skips its access check, every test here fails: the save and move tests
get 422 from the terminal (nothing written or moved), and the stranger's delete goes through.
"""

from __future__ import annotations

import hashlib
import uuid

import httpx
import pytest

from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture
def granted(admin, preserve, make_user, open_terminal):
    """A person with a grant on the Open Terminal, and the connection."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return person, connection


def _base(connection: dict) -> str:
    return f"/api/v1/terminals/{connection['id']}"


def _save(client: httpx.Client, connection: dict, folder, name: str, content: bytes, kind: str):
    """Upload the whole file into its folder, as the file browser's Save does."""
    return client.post(
        f"{_base(connection)}/files/upload",
        params={"directory": str(folder)},
        files={"file": (name, content, kind)},
    )


def _move(client: httpx.Client, connection: dict, source, destination):
    return client.post(
        f"{_base(connection)}/files/move",
        json={"source": str(source), "destination": str(destination)},
    )


def _delete(client: httpx.Client, connection: dict, path):
    return client.delete(f"{_base(connection)}/files/delete", params={"path": str(path)})


def _workbook_sized_bytes(size: int) -> bytes:
    """Deterministic bytes that do not compress, so every one has to cross the proxy."""
    blocks = (hashlib.sha256(str(n).encode()).digest() for n in range(size // 32 + 1))
    return b"PK\x03\x04" + b"".join(blocks)[: size - 4]


def test_saving_a_text_file_overwrites_it_on_the_terminal(granted, open_terminal):
    person, connection = granted
    note = open_terminal.home / f"note-{uuid.uuid4().hex[:6]}.md"
    note.write_text("# Draft\n")

    with person.client() as client:
        saved = _save(client, connection, note.parent, note.name, b"# Final\n", "text/plain")

    assert saved.status_code == 200, f"saving through the proxy failed: {saved.text}"
    assert note.read_text() == "# Final\n", f"the file on disk reads {note.read_text()!r}"


def test_a_saved_workbook_arrives_byte_for_byte(granted, open_terminal):
    person, connection = granted
    workbook = open_terminal.home / f"ledger-{uuid.uuid4().hex[:6]}.xlsx"
    content = _workbook_sized_bytes(6 * 1024 * 1024 + 7)

    with person.client() as client:
        saved = _save(client, connection, workbook.parent, workbook.name, content, XLSX_TYPE)

    assert saved.status_code == 200, f"saving the workbook through the proxy failed: {saved.text}"
    on_disk = workbook.read_bytes()
    assert len(on_disk) == len(content), f"{len(on_disk)} of {len(content)} bytes arrived"
    assert on_disk == content, "the workbook on disk differs from the one saved"


def test_moving_and_deleting_change_the_terminals_disk(granted, open_terminal):
    person, connection = granted
    home = open_terminal.home
    folder = home / f"archive-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    letter = home / "moved-letter.txt"
    letter.write_text("Dear harbour master,\n")
    receipt = home / f"receipt-{uuid.uuid4().hex[:6]}.txt"
    receipt.write_text("two ropes\n")

    with person.client() as client:
        moved = _move(client, connection, letter, folder / letter.name)
        deleted = _delete(client, connection, receipt)

    assert moved.status_code == 200, f"moving through the proxy failed: {moved.text}"
    assert (folder / letter.name).read_text() == "Dear harbour master,\n"
    assert not letter.exists(), "the moved file is still in its old folder"
    assert deleted.status_code == 200, f"deleting through the proxy failed: {deleted.text}"
    assert not receipt.exists(), "the deleted file is still on disk"


def test_a_person_without_a_grant_changes_no_file(granted, make_user, open_terminal):
    _, connection = granted
    stranger = make_user()
    home = open_terminal.home
    kept = home / f"kept-{uuid.uuid4().hex[:6]}.txt"
    kept.write_text("original\n")
    folder = home / f"vault-{uuid.uuid4().hex[:6]}"
    folder.mkdir()

    with stranger.client() as client:
        answers = {
            "save": _save(client, connection, home, kept.name, b"overwritten\n", "text/plain"),
            "move": _move(client, connection, kept, folder / kept.name),
            "delete": _delete(client, connection, kept),
        }

    refused = {action: answer.status_code for action, answer in answers.items()}
    assert refused == {"save": 403, "move": 403, "delete": 403}, f"the stranger got {refused}"
    assert kept.read_text() == "original\n", f"the file now reads {kept.read_text()!r}"
    assert not (folder / kept.name).exists(), "the stranger's move went through"
