"""A `display_file` terminal tool call without `inline` still resolves to a file entry.

open-webui 0.11.2 `64e6c9f01` (a `refac`) in `utils/middleware.py`:
`build_terminal_file_tool_result` returned `None` unless `inline` was exactly `True`, so a
`display_file` call made without it never became the `{'type': 'file', 'source':
'open_terminal', ...}` descriptor and reached the model as the raw terminal payload, with no
name or mime type resolved. The fix builds the descriptor for every call and sets `displayed`
only for `inline is True`, the flag the frontend reads to render the file inline.

Stays a unit test: exercising it over HTTP needs a terminal server whose OpenAPI tools the chat
path loads and calls, much machinery for a pure function with no I/O of its own.

Discriminates: passes on bbfa876af; fails with `inline is not True` restored to the bail-out
(the non-inline call returns None).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

TERMINAL_TOOL = {"tool_id": "terminal:term-1"}
REPORT = {"exists": True, "path": "/workspace/report.png"}


@pytest.fixture(scope="session")
def build(owui_module):
    middleware = owui_module("open_webui.utils.middleware")

    def build_result(params, result=REPORT, tool=TERMINAL_TOOL, metadata=None, name="display_file"):
        return middleware.build_terminal_file_tool_result(
            tool_function_name=name,
            tool_function_params=params,
            tool_result=result,
            tool=tool,
            metadata={"chat_id": "chat-1"} if metadata is None else metadata,
        )

    return build_result


@pytest.mark.parametrize("inline", [False, None, "true", 1], ids=["false", "absent", "str", "one"])
def test_a_display_file_call_without_inline_true_still_resolves_the_file(build, inline):
    params = {"path": "/workspace/report.png", **({"inline": inline} if inline is not None else {})}

    entry = build(params)

    assert entry is not None, "a display_file call without inline=True was left unresolved"
    assert "displayed" not in entry, "a file never shown inline was flagged as displayed"
    assert {key: entry[key] for key in ("type", "source", "name", "mime_type")} == {
        "type": "file",
        "source": "open_terminal",
        "name": "report.png",
        "mime_type": "image/png",
    }
    assert entry["terminal_id"] == entry["terminal_selector"] == "term-1"


def test_an_inline_call_is_marked_displayed(build):
    assert build({"path": "/workspace/report.png", "inline": True})["displayed"] is True


@pytest.mark.parametrize(
    ("changes", "why"),
    [
        ({"name": "read_file"}, "another tool"),
        ({"result": {"exists": False, "path": "/workspace/gone.png"}}, "a missing file"),
        ({"result": "not a dict"}, "a non-dict result"),
        ({"tool": {}, "metadata": {}}, "no terminal to select"),
        ({"result": {"exists": True}}, "no path"),
    ],
)
def test_calls_that_name_no_terminal_file_are_left_alone(build, changes, why):
    assert build({"inline": True}, **changes) is None, why


def test_a_server_backed_terminal_is_selected_by_its_url(build):
    server_tool = {"server": {"url": "http://terminal.local:9000"}}

    entry = build({"inline": True}, tool=server_tool, metadata={})

    assert entry["terminal_selector"] == entry["terminal_url"] == "http://terminal.local:9000"
    assert "terminal_id" not in entry


@pytest.mark.parametrize(
    ("result", "params", "expected"),
    [
        (
            [{"exists": True, "path": "/w/a.txt"}],
            {},
            {"path": "/w/a.txt", "mime_type": "text/plain"},
        ),
        ({"exists": True, "path": "/w/blob.zzz"}, {}, {"mime_type": "application/octet-stream"}),
        ({**REPORT, "content_type": "image/webp"}, {}, {"mime_type": "image/webp"}),
        (REPORT, {"page": 3}, {"page": 3}),
    ],
    ids=["list-wrapped", "unknown-extension", "result-content-type", "page-from-params"],
)
def test_the_file_fields_are_resolved(build, result, params, expected):
    entry = build({"inline": True, **params}, result=result)

    assert {key: entry.get(key) for key in expected} == expected
