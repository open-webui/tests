"""A scripted terminal that holds files in memory, for a file browser to list and open.

`serving_files()` is the fake terminal of `harness.terminal_server` with the calls a file browser
makes answered from two dicts the test fills: `files` maps a name in the terminal's home to its
bytes, and `converted` maps a document's name to the PDF the terminal makes of it for a preview
(a terminal with LibreOffice; without an entry the preview request gets the file itself, as from
a terminal without). Every other call (`files/compare`, `files/upload`, ...) is the test's to
route. `connection()` is the admin's connection to it, as for any fake terminal.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from urllib.parse import parse_qs, urlparse

from harness.listener import json_answer
from harness.terminal_server import FakeTerminalServer, serving_terminal

HOME = "/home/user"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
# a fixed day, so a listing reads the same in every run
MODIFIED = 1767225600


@contextmanager
def serving_files() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        server.files = {}
        server.converted = {}

        def listing(request):
            entries = [
                {"name": name, "type": "file", "size": len(content), "modified": MODIFIED}
                for name, content in server.files.items()
            ]
            return json_answer({"dir": HOME, "entries": entries, "writable": True})

        def view(request):
            query = parse_qs(urlparse(request.path).query)
            name = query["path"][0].rsplit("/", 1)[-1]
            if query.get("preview") == ["true"] and name in server.converted:
                return 200, {"Content-Type": "application/pdf"}, server.converted[name]
            if name not in server.files:
                return json_answer({"detail": "File not found"}, 404)
            return 200, {"Content-Type": "application/octet-stream"}, server.files[name]

        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
        server.route("GET", "/files/list", listing)
        server.route("GET", "/files/view", view)
        yield server
