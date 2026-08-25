"""Regression: a document container that unpacks far beyond its stored size.

open-webui 0.11.1 fix `2a0274a0a`: `Loader._get_loader` in
`open_webui/retrieval/loaders/main.py` handed Office/OpenDocument/EPUB uploads
straight to their extractor. Those five formats (`docx`, `xlsx`, `pptx`, `odt`,
`epub`) are zip containers, so a few kilobytes on disk can declare gigabytes of
member data, and the extractor inflates all of it into the worker's memory
before anything gets a chance to reject it. A single upload could exhaust the
instance.

The fix reads the zip central directory first (declared sizes only, nothing is
inflated) and raises `ValueError('Document archive is too large after
decompression')` when the total exceeds
`min(max(10 MiB, stored_size * 100), FILE_MAX_SIZE)`. `FILE_MAX_SIZE` is the
admin's `rag.file.max_size` in MiB, now forwarded by `build_loader_from_config`
in `open_webui/retrieval/utils.py`, defaulting to 100 MiB.

Discriminates: passes on v0.11.1, fails on v0.11.0, where `_get_loader` returns
the extractor for a bomb with no size check at all.

Bounded by construction: every test stops at `_get_loader`, which never reads
member data, so the pre-fix run inspects the same central directory the fix
does and nothing inflates on either ref.
"""

import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

MIB = 1024 * 1024

# The five container formats the fix guards, with the content type each is
# uploaded under.
ARCHIVE_FORMATS = {
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'odt': 'application/vnd.oasis.opendocument.text',
    'epub': 'application/epub+zip',
}

TOO_LARGE = 'too large after decompression'

# Comfortably over the 10 MiB floor the guard never goes below.
BOMB_DECLARED_BYTES = 16 * MIB


@pytest.fixture(scope='session')
def loaders_main(owui_module):
    return owui_module('open_webui.retrieval.loaders.main')


def _write_bomb(path: Path, declared_bytes: int = BOMB_DECLARED_BYTES) -> Path:
    """A real zip whose single member declares `declared_bytes` but stores ~0.

    Written a chunk at a time: the bomb only ever exists as a compression
    stream, so building it costs the suite nothing.
    """
    chunk = b'\0' * MIB
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        with archive.open('word/document.xml', 'w') as member:
            for _ in range(declared_bytes // MIB):
                member.write(chunk)
    return path


def _write_ordinary(path: Path) -> Path:
    """A small, plausibly-shaped container with an unremarkable expansion ratio."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', b'<?xml version="1.0"?><Types/>')
        archive.writestr('content.xml', b'<document>Quarterly revenue summary.</document>')
    return path


def _declared_size(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(entry.file_size for entry in archive.infolist())


def test_zip_bomb_document_is_rejected_before_it_is_read(loaders_main, tmp_path):
    """The reported shape: a ~16 KiB .docx that declares 16 MiB of member data."""
    bomb = _write_bomb(tmp_path / 'invoice.docx')

    assert bomb.stat().st_size < MIB, 'bomb must stay small on disk to be a bomb'
    assert _declared_size(bomb) == BOMB_DECLARED_BYTES

    loader = loaders_main.Loader(engine='')
    with pytest.raises(ValueError) as excinfo:
        loader._get_loader('invoice.docx', ARCHIVE_FORMATS['docx'], str(bomb))

    assert TOO_LARGE in str(excinfo.value)


@pytest.mark.parametrize('file_ext', sorted(ARCHIVE_FORMATS))
def test_every_container_format_rejects_a_zip_bomb(loaders_main, tmp_path, file_ext):
    """The class, not just the docx instance: all five formats are containers."""
    bomb = _write_bomb(tmp_path / f'report.{file_ext}')

    loader = loaders_main.Loader(engine='')
    with pytest.raises(ValueError, match=TOO_LARGE):
        loader._get_loader(f'report.{file_ext}', ARCHIVE_FORMATS[file_ext], str(bomb))


@pytest.mark.parametrize('file_ext', sorted(ARCHIVE_FORMATS))
def test_zip_bomb_is_rejected_on_content_type_alone(loaders_main, tmp_path, file_ext):
    """Renaming the upload does not get it past the guard: the content type is
    checked too, so `bomb.bin` declared as a spreadsheet is still refused."""
    bomb = _write_bomb(tmp_path / f'{file_ext}-payload.bin')

    loader = loaders_main.Loader(engine='')
    with pytest.raises(ValueError, match=TOO_LARGE):
        loader._get_loader(f'{file_ext}-payload.bin', ARCHIVE_FORMATS[file_ext], str(bomb))


@pytest.mark.parametrize('file_ext', sorted(ARCHIVE_FORMATS))
def test_ordinary_container_document_still_reaches_its_extractor(loaders_main, tmp_path, file_ext):
    """The other half of the fix: normal documents must still be dispatched."""
    document = _write_ordinary(tmp_path / f'notes.{file_ext}')

    loader = loaders_main.Loader(engine='')
    extractor = loader._get_loader(f'notes.{file_ext}', ARCHIVE_FORMATS[file_ext], str(document))

    assert extractor is not None


def test_a_real_docx_still_loads_end_to_end(loaders_main, tmp_path):
    """A genuine Word document expands ~20x over its stored size, well inside
    the floor, and comes back out of `Loader.load` as text."""
    docx = pytest.importorskip('docx')

    path = tmp_path / 'memo.docx'
    document = docx.Document()
    document.add_paragraph('Quarterly revenue summary.')
    document.save(str(path))

    assert _declared_size(path) > path.stat().st_size, 'a real docx does expand'

    loader = loaders_main.Loader(engine='')
    docs = loader.load('memo.docx', ARCHIVE_FORMATS['docx'], str(path))

    assert 'Quarterly revenue summary.' in '\n'.join(doc.page_content for doc in docs)


def test_expansion_under_the_floor_is_allowed(loaders_main, tmp_path):
    """The guard never rejects below 10 MiB, however extreme the ratio, so
    small legitimate documents are not caught by the compression ratio alone."""
    document = _write_bomb(tmp_path / 'slides.pptx', declared_bytes=8 * MIB)

    loader = loaders_main.Loader(engine='')
    extractor = loader._get_loader('slides.pptx', ARCHIVE_FORMATS['pptx'], str(document))

    assert extractor is not None


def test_configured_file_max_size_tightens_the_ceiling(loaders_main, tmp_path):
    """`FILE_MAX_SIZE` (MiB) caps the allowance, so an admin who sets a 1 MiB
    upload limit gets a 1 MiB decompression limit rather than the 10 MiB floor."""
    document = _write_bomb(tmp_path / 'sheet.xlsx', declared_bytes=4 * MIB)
    content_type = ARCHIVE_FORMATS['xlsx']

    default_loader = loaders_main.Loader(engine='')
    assert default_loader._get_loader('sheet.xlsx', content_type, str(document)) is not None

    limited_loader = loaders_main.Loader(engine='', FILE_MAX_SIZE=1)
    with pytest.raises(ValueError, match=TOO_LARGE):
        limited_loader._get_loader('sheet.xlsx', content_type, str(document))


def test_non_container_uploads_are_not_size_checked(loaders_main, tmp_path):
    """Scoped to the five container formats: a PDF is not a zip and must not be
    run through the archive guard, whatever its bytes happen to look like."""
    disguised = _write_bomb(tmp_path / 'scan.pdf')

    loader = loaders_main.Loader(engine='')
    extractor = loader._get_loader('scan.pdf', 'application/pdf', str(disguised))

    assert extractor is not None


def test_unreadable_archive_is_left_to_the_extractor(loaders_main, tmp_path):
    """A file that is not a readable zip cannot have its expansion measured. The
    guard steps aside instead of turning every corrupt upload into a bomb report."""
    corrupt = tmp_path / 'broken.docx'
    corrupt.write_bytes(b'not a zip at all')

    loader = loaders_main.Loader(engine='')
    extractor = loader._get_loader('broken.docx', ARCHIVE_FORMATS['docx'], str(corrupt))

    assert extractor is not None


def test_loader_built_from_admin_config_carries_the_limit(owui_module, tmp_path):
    """The wiring: `build_loader_from_config` is what the upload path uses, and
    it must forward `rag.file.max_size` or the guard runs on defaults only."""
    retrieval_utils = owui_module('open_webui.retrieval.utils')

    document = _write_bomb(tmp_path / 'deck.pptx', declared_bytes=4 * MIB)
    config = {'CONTENT_EXTRACTION_ENGINE': '', 'file_max_size': 1}

    loader = retrieval_utils.build_loader_from_config(None, config)
    with pytest.raises(ValueError, match=TOO_LARGE):
        loader._get_loader('deck.pptx', ARCHIVE_FORMATS['pptx'], str(document))
