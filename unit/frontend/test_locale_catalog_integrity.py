"""Guard: the i18n locale catalogs must stay loadable and in sync.

The frontend picks a language from src/lib/i18n/locales/languages.json and
then fetches locales/<code>/translation.json. The two sides are maintained by
hand and by `npm run i18n:parse`, so they drift in both directions: a
contributor adds a locale directory and forgets languages.json (the language
never appears in the picker), or an entry is listed with no directory behind
it (choosing it loads nothing and the UI falls back to raw translation keys).

A trailing comma or an unescaped quote in any translation.json is the same
class of accident and breaks that language outright.

A python source audit — it reads the JSON files, so it stays runnable without
node or the frontend's dependencies installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LOCALES_DIR = Path("src") / "lib" / "i18n" / "locales"


@pytest.fixture(scope="module")
def locales_dir(open_webui_backend: Path) -> Path:
    path = open_webui_backend.parent / LOCALES_DIR
    if not path.is_dir():
        pytest.skip(f"no locales directory at {path}")
    return path


@pytest.fixture(scope="module")
def locale_dirs(locales_dir: Path) -> list[str]:
    dirs = sorted(entry.name for entry in locales_dir.iterdir() if entry.is_dir())
    assert dirs, f"no locale directories under {locales_dir}"
    return dirs


@pytest.fixture(scope="module")
def declared_languages(locales_dir: Path) -> list[dict]:
    manifest = locales_dir / "languages.json"
    if not manifest.is_file():
        pytest.skip(f"no languages.json at {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def test_languages_manifest_entries_have_a_code_and_a_title(declared_languages: list[dict]) -> None:
    incomplete = [
        entry for entry in declared_languages if not entry.get("code") or not entry.get("title")
    ]
    assert not incomplete, f"languages.json entries missing code or title: {incomplete}"


def test_no_language_is_declared_twice(declared_languages: list[dict]) -> None:
    codes = [entry["code"] for entry in declared_languages]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    assert not duplicates, f"languages.json lists these codes more than once: {duplicates}"


def test_every_declared_language_has_a_catalog(
    declared_languages: list[dict], locale_dirs: list[str]
) -> None:
    """Picking one of these in the UI would load nothing and leave the whole
    interface showing raw translation keys."""
    orphans = sorted(
        entry["code"] for entry in declared_languages if entry["code"] not in locale_dirs
    )
    assert not orphans, f"languages.json codes with no locales/<code>/ directory: {orphans}"


def test_every_catalog_is_offered_in_the_picker(
    declared_languages: list[dict], locale_dirs: list[str]
) -> None:
    """A translated language nobody can select is a wasted contribution."""
    declared = {entry["code"] for entry in declared_languages}
    unlisted = [code for code in locale_dirs if code not in declared]
    assert not unlisted, f"locale directories missing from languages.json: {unlisted}"


def test_every_catalog_is_valid_json(locales_dir: Path, locale_dirs: list[str]) -> None:
    broken = []
    for code in locale_dirs:
        catalog = locales_dir / code / "translation.json"
        if not catalog.is_file():
            broken.append((code, "no translation.json"))
            continue
        try:
            parsed = json.loads(catalog.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            broken.append((code, str(error)))
            continue
        if not isinstance(parsed, dict):
            broken.append((code, f"top level is {type(parsed).__name__}, expected an object"))
    assert not broken, f"unusable translation catalogs: {broken}"


def test_the_source_locale_is_present_and_populated(locales_dir: Path) -> None:
    """en-US is the fallback every other language falls back to."""
    catalog = locales_dir / "en-US" / "translation.json"
    assert catalog.is_file(), f"no en-US catalog at {catalog}"
    assert json.loads(catalog.read_text(encoding="utf-8")), "the en-US catalog is empty"
