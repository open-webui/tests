"""Regression: a non-inline `display_file` terminal tool call must still produce a
file entry, just without the `displayed` flag.

open-webui 0.11.2 commit `64e6c9f01` (labelled "refac"), in
`open_webui/utils/middleware.py`. `build_terminal_file_tool_result` bailed out with
`None` whenever `tool_function_params['inline']` was not exactly `True`, so a
`display_file` call made without `inline` never got rewritten into the
`{'type': 'file', 'source': 'open_terminal', ...}` descriptor and reached the model as
the raw terminal payload with no path, name or mime type resolved. The fix drops
`inline` from the bail-out condition and instead sets `'displayed': True` only when
`inline is True`, which is the flag the frontend uses to decide whether to render the
file inline.

Discriminates: passes on v0.11.3, fails on v0.11.1 (a `display_file` call without
`inline=True` returns None instead of a file descriptor).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

TERMINAL_TOOL = {"tool_id": "terminal:term-1"}
SERVER_TOOL = {"server": {"url": "http://terminal.local:9000"}}
METADATA = {"chat_id": "chat-1"}


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


@pytest.fixture(scope="session")
def build_result(middleware_module):
    def _build(params, result, tool=TERMINAL_TOOL, metadata=METADATA, name="display_file"):
        return middleware_module.build_terminal_file_tool_result(
            name, params, result, tool, metadata
        )

    return _build


def _read_result(path="/workspace/report.png", **extra):
    return {"exists": True, "path": path, **extra}


# Narrow: the non-inline call is what regressed.


def test_non_inline_display_file_still_builds_a_file_descriptor(build_result):
    entry = build_result({"path": "/workspace/report.png"}, _read_result())

    assert entry is not None, (
        "a display_file call without inline=True produced no terminal file descriptor, "
        "so the raw tool payload reached the model with no path, name or mime type (64e6c9f01)"
    )
    assert entry["type"] == "file"
    assert entry["source"] == "open_terminal"
    assert entry["terminal_selector"] == "term-1"
    assert entry["terminal_id"] == "term-1"
    assert entry["path"] == "/workspace/report.png"
    assert entry["name"] == "report.png"
    assert entry["mime_type"] == "image/png"


def test_non_inline_display_file_is_not_marked_displayed(build_result):
    entry = build_result({"path": "/workspace/report.png", "inline": False}, _read_result())

    assert entry is not None
    assert "displayed" not in entry, (
        "a file the user never asked to see inline was flagged as displayed, so the "
        "frontend renders it as if it had been shown (64e6c9f01)"
    )


# Broad: `displayed` tracks `inline is True` exactly, and nothing else gates the
# descriptor being built.


@pytest.mark.parametrize(
    "inline,expect_displayed",
    [
        (True, True),
        (False, False),
        (None, False),
        ("true", False),
        (1, False),
    ],
)
def test_displayed_is_set_only_for_an_exact_inline_true(build_result, inline, expect_displayed):
    params = {"path": "/workspace/report.png"}
    if inline is not None:
        params["inline"] = inline

    entry = build_result(params, _read_result())

    assert entry is not None
    assert entry.get("displayed", False) is expect_displayed


@pytest.mark.parametrize("inline", [True, False, None])
def test_every_inline_variant_resolves_the_same_file_identity(build_result, inline):
    params = {"path": "/workspace/notes/readme.md"}
    if inline is not None:
        params["inline"] = inline

    entry = build_result(params, _read_result(path="/workspace/notes/readme.md"))

    assert entry is not None
    identity = {key: entry[key] for key in ("type", "source", "path", "name", "mime_type")}
    assert identity == {
        "type": "file",
        "source": "open_terminal",
        "path": "/workspace/notes/readme.md",
        "name": "readme.md",
        "mime_type": "text/markdown",
    }


# Nearby: the other bail-outs and the field resolution are untouched by the fix. The
# positive-descriptor cases stay on inline=True, the only variant the pre-fix code built
# a descriptor for at all.


@pytest.mark.parametrize("inline", [True, False])
def test_a_non_display_file_tool_is_never_rewritten(build_result, inline):
    assert (
        build_result(
            {"path": "/workspace/a.png", "inline": inline}, _read_result(), name="read_file"
        )
        is None
    )


@pytest.mark.parametrize("inline", [True, False])
def test_a_missing_file_is_never_rewritten(build_result, inline):
    result = {"exists": False, "path": "/workspace/gone.png"}

    assert build_result({"path": "/workspace/gone.png", "inline": inline}, result) is None


@pytest.mark.parametrize("inline", [True, False])
def test_a_non_dict_tool_result_is_never_rewritten(build_result, inline):
    assert build_result({"path": "/workspace/a.png", "inline": inline}, "not a dict") is None


@pytest.mark.parametrize("inline", [True, False])
def test_a_result_without_a_terminal_selector_is_never_rewritten(build_result, inline):
    assert (
        build_result({"path": "/a.png", "inline": inline}, _read_result(), tool={}, metadata={})
        is None
    )


@pytest.mark.parametrize("inline", [True, False])
def test_a_result_without_a_path_is_never_rewritten(build_result, inline):
    assert build_result({"inline": inline}, {"exists": True}) is None


def test_a_list_wrapped_result_is_unwrapped(build_result):
    entry = build_result({"inline": True}, [_read_result(path="/workspace/a.txt")])

    assert entry is not None
    assert entry["path"] == "/workspace/a.txt"
    assert entry["mime_type"] == "text/plain"


def test_a_server_backed_terminal_is_selected_by_url(build_result):
    entry = build_result({"inline": True}, _read_result(), tool=SERVER_TOOL, metadata=None)

    assert entry is not None
    assert entry["terminal_selector"] == "http://terminal.local:9000"
    assert entry["terminal_url"] == "http://terminal.local:9000"
    assert "terminal_id" not in entry
    assert entry["session_id"] is None


def test_unknown_extensions_fall_back_to_octet_stream(build_result):
    entry = build_result({"inline": True}, _read_result(path="/workspace/blob.zzz"))

    assert entry is not None
    assert entry["mime_type"] == "application/octet-stream"
    assert entry["content_type"] == "application/octet-stream"


def test_the_tool_result_content_type_wins_over_the_guess(build_result):
    entry = build_result({"inline": True}, _read_result(content_type="image/webp"))

    assert entry is not None
    assert entry["mime_type"] == "image/webp"
    assert entry["content_type"] == "image/webp"


def test_the_page_is_carried_from_params_when_the_result_omits_it(build_result):
    entry = build_result({"inline": True, "page": 3}, _read_result())

    assert entry is not None
    assert entry["page"] == 3
    assert build_result({"inline": True}, _read_result()).get("page") is None
