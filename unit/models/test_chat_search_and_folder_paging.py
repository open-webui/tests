"""Background message writes during a reply must not move the chat up the sidebar.

`f1ded94` / `a9617ca` (open-webui 0.11.0): suggestions, the chosen model and the compaction
summary were saved through the chat upsert, which always bumped `Chat.updated_at`, so the sidebar
re-sorted in the middle of a reply. The upsert now takes `touch`, and every background write
passes `touch=False`. The socket event writes (sources, files, embeds) are pinned over HTTP by
integration/models/test_chat_search_and_folder_paging.py, with the folder paging and PostgreSQL
search fixes of the same release; the writes below only happen deep inside a live reply, so
their call sites are audited here.

Discriminates: passes on upstream dev `bbfa876af`; with `touch=False` removed from the
`followUps` write in `utils/middleware.py` its case fails.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.regression

CHAT_UPSERTS = {
    "upsert_message_to_chat_by_id_and_message_id",
    "add_message_status_to_chat_by_id_and_message_id",
}

# Message fields written only in the background of a reply, by the module that writes them.
BACKGROUND_WRITES = {
    "open_webui/utils/middleware.py": {"followUps", "selectedModelId"},
    "open_webui/utils/context_compaction.py": {"contextSummary"},
}


def _upserts_by_field(tree: ast.Module) -> dict[str, list[ast.Call]]:
    """Chat-upsert calls keyed by each message field their literal payload writes."""
    calls: dict[str, list[ast.Call]] = {}
    for node in ast.walk(tree):
        is_upsert = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in CHAT_UPSERTS
        )
        if not is_upsert:
            continue
        payloads = [argument for argument in node.args if isinstance(argument, ast.Dict)]
        for payload in payloads:
            for key in payload.keys:
                if isinstance(key, ast.Constant):
                    calls.setdefault(key.value, []).append(node)
    return calls


def _opts_out_of_touch(call: ast.Call) -> bool:
    return any(
        keyword.arg == "touch"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in call.keywords
    )


@pytest.mark.parametrize(("relative_path", "fields"), sorted(BACKGROUND_WRITES.items()))
def test_every_background_write_leaves_the_chat_where_it_is(
    open_webui_backend, relative_path, fields
):
    source_path = open_webui_backend / relative_path
    assert source_path.is_file(), f"{relative_path} is gone; retarget the audit at its writes"
    upserts = _upserts_by_field(ast.parse(source_path.read_text(encoding="utf-8")))

    for field in sorted(fields):
        writes = upserts.get(field)
        assert writes, f"no chat upsert in {relative_path} writes {field}; retarget the audit"
        touching = [call.lineno for call in writes if not _opts_out_of_touch(call)]
        assert not touching, (
            f"{relative_path}:{touching} re-sorts the sidebar when it writes {field}"
        )
