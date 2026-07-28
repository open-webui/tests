"""Regression: a folder's attached notes reached the model context unfiltered.

open-webui 0.11.0 fix `f89b50198` (PR #26739, related #26723):
`get_accessible_folder_files` is the server-side filter that reduces a folder's
attached-knowledge list to the entries the caller may read, before that list is
handed to the builtin knowledge tools as `__model_knowledge__` (and, in 0.10.2,
spliced into `form_data['files']` by the middleware). It validated `file` and
`collection` entries but let `note` entries fall into the keep-as-is `else`
branch, so a note the caller cannot read stayed in the list. Every note consumer
re-checked access independently, so no shipped caller leaked content, but the
filter was fail-open: any future path trusting list membership would be an IDOR.
The fix keeps a note only when the caller owns it or holds a read grant.

Discriminates: passes on v0.11.0, fails on v0.10.2 (unreadable notes survive the
filter and land in the list handed to the assistant).
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

CALLER = SimpleNamespace(id='alice', role='user')
ADMIN = SimpleNamespace(id='root', role='admin')


def _entry(entry_type: str, entry_id: str) -> dict:
    return {'type': entry_type, 'id': entry_id}


def _ids(entries: list[dict]) -> list[str]:
    return [entry['id'] for entry in entries]


@contextmanager
def _patch_access_layer(
    files_module,
    readable_file_ids: set[str] = frozenset(),
    readable_collection_ids: set[str] = frozenset(),
    notes: dict[str, str] | None = None,
    note_read_grants: set[str] = frozenset(),
):
    """Stand in for the four DB-backed checks the filter consults.

    `notes` maps note id to owner id; `note_read_grants` holds note ids the
    caller has a read grant on. Callables are recorded so a test can assert a
    check actually ran rather than the entry being waved through.
    """
    import open_webui.models.access_grants as access_grants_module
    import open_webui.models.groups as groups_module
    import open_webui.models.knowledge as knowledge_module
    import open_webui.models.notes as notes_module

    notes = notes or {}

    file_check = AsyncMock(side_effect=lambda file_id, *a, **kw: file_id in readable_file_ids)
    collection_check = AsyncMock(
        side_effect=lambda collection_id, *a, **kw: collection_id in readable_collection_ids
    )
    note_lookup = AsyncMock(
        side_effect=lambda note_id, *a, **kw: (
            SimpleNamespace(id=note_id, user_id=notes[note_id]) if note_id in notes else None
        )
    )
    grant_check = AsyncMock(
        side_effect=lambda **kw: kw.get('resource_type') == 'note'
        and kw.get('resource_id') in note_read_grants
    )

    with ExitStack() as stack:
        stack.enter_context(patch.object(files_module, 'has_access_to_file', file_check))
        stack.enter_context(
            patch.object(knowledge_module.Knowledges, 'check_access_by_user_id', collection_check)
        )
        stack.enter_context(patch.object(notes_module.Notes, 'get_note_by_id', note_lookup))
        stack.enter_context(
            patch.object(access_grants_module.AccessGrants, 'has_access', grant_check)
        )
        stack.enter_context(
            patch.object(
                groups_module.Groups, 'get_groups_by_member_id', AsyncMock(return_value=[])
            )
        )
        yield SimpleNamespace(file=file_check, collection=collection_check, note=note_lookup)


@pytest.fixture
def files_access_module(owui_module):
    return owui_module('open_webui.utils.access_control.files')


# --- Narrow: the note entries themselves ------------------------------------


@pytest.mark.asyncio
async def test_unreadable_note_is_dropped_from_folder_knowledge(files_access_module):
    entries = [_entry('note', 'bobs-private-note')]
    with _patch_access_layer(files_access_module, notes={'bobs-private-note': 'bob'}):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == [], (
        "a note the caller cannot read survived the folder-knowledge filter and would be "
        'handed to the assistant as attached knowledge (#26739)'
    )


@pytest.mark.asyncio
async def test_unreadable_note_dropped_while_readable_entries_survive(files_access_module):
    entries = [
        _entry('file', 'shared-file'),
        _entry('note', 'bobs-private-note'),
        _entry('note', 'alices-note'),
    ]
    with _patch_access_layer(
        files_access_module,
        readable_file_ids={'shared-file'},
        notes={'bobs-private-note': 'bob', 'alices-note': 'alice'},
    ):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == ['shared-file', 'alices-note'], (
        'the filter must keep exactly the entries the caller may read; a foreign note leaking '
        'through means its content reaches the model context (#26739)'
    )


@pytest.mark.asyncio
async def test_note_row_that_no_longer_exists_is_dropped(files_access_module):
    """A dangling note id must not be kept just because it cannot be resolved."""
    entries = [_entry('note', 'deleted-note')]
    with _patch_access_layer(files_access_module, notes={}):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == [], (
        'an unresolvable note id stayed in the folder-knowledge list, so the assistant is '
        'handed an entry nobody proved the caller may read (#26739)'
    )


# --- Broad: every content type in the list is checked where it is assembled ---


@pytest.mark.asyncio
@pytest.mark.parametrize('entry_type', ['file', 'collection', 'note'])
async def test_every_entry_type_is_denied_when_the_caller_has_no_access(
    files_access_module, entry_type
):
    """No content type may pass through on the strength of list membership."""
    entries = [_entry(entry_type, 'foreign-entry')]
    with _patch_access_layer(files_access_module, notes={'foreign-entry': 'bob'}):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == [], (
        f"a '{entry_type}' entry the caller cannot read survived the folder-knowledge filter; "
        'the filter must enforce access itself rather than lean on downstream re-checks (#26739)'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('entry_type', ['file', 'collection', 'note'])
async def test_every_entry_type_consults_its_access_check(files_access_module, entry_type):
    entries = [_entry(entry_type, 'some-entry')]
    with _patch_access_layer(
        files_access_module,
        readable_file_ids={'some-entry'},
        readable_collection_ids={'some-entry'},
        notes={'some-entry': 'alice'},
    ) as checks:
        await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert getattr(checks, entry_type).await_count == 1, (
        f"'{entry_type}' entries were kept without any access lookup, so the caller's own "
        'permissions never gated what reaches the model context (#26739)'
    )


# --- Nearby: behaviour that was already correct and must stay so -------------


@pytest.mark.asyncio
async def test_caller_own_note_is_kept(files_access_module):
    """Notes are private by default and carry no self-grant, so ownership alone must suffice."""
    entries = [_entry('note', 'alices-note')]
    with _patch_access_layer(files_access_module, notes={'alices-note': 'alice'}):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == ['alices-note']


@pytest.mark.asyncio
async def test_note_shared_with_the_caller_is_kept(files_access_module):
    entries = [_entry('note', 'bobs-shared-note')]
    with _patch_access_layer(
        files_access_module,
        notes={'bobs-shared-note': 'bob'},
        note_read_grants={'bobs-shared-note'},
    ):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == ['bobs-shared-note'], (
        'a read grant on a note must still admit it, otherwise the fix over-corrected and '
        'deliberately shared notes vanish from folder knowledge'
    )


@pytest.mark.asyncio
async def test_readable_file_and_collection_are_kept(files_access_module):
    entries = [_entry('file', 'shared-file'), _entry('collection', 'shared-kb')]
    with _patch_access_layer(
        files_access_module,
        readable_file_ids={'shared-file'},
        readable_collection_ids={'shared-kb'},
    ):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert _ids(accessible) == ['shared-file', 'shared-kb']


@pytest.mark.asyncio
@pytest.mark.parametrize('entries', [None, []])
async def test_folder_without_attached_knowledge(files_access_module, entries):
    with _patch_access_layer(files_access_module):
        accessible = await files_access_module.get_accessible_folder_files(entries, CALLER)

    assert accessible == []


@pytest.mark.asyncio
async def test_admin_bypasses_the_note_check(files_access_module):
    """Admins already see every note through the notes API, so the filter must not narrow them."""
    entries = [_entry('note', 'bobs-private-note'), _entry('file', 'bobs-file')]
    with _patch_access_layer(files_access_module, notes={'bobs-private-note': 'bob'}) as checks:
        accessible = await files_access_module.get_accessible_folder_files(entries, ADMIN)

    assert _ids(accessible) == ['bobs-private-note', 'bobs-file']
    assert checks.note.await_count == 0
