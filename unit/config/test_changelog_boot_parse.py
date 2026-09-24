"""Guard: every release heading in CHANGELOG.md must survive the parse `open_webui.env` runs.

env.py renders CHANGELOG.md to HTML with `markdown` and walks its h2 headings with BeautifulSoup,
taking `text.split(' - ')[0][1:-1]` as the version and `[1]` as the date. The parse is unguarded:
a heading without the ` - <date>` half raises IndexError inside `import open_webui.env`, which every
backend module imports, so a release-note typo takes the application down before it can log
anything. A repeated version silently drops one release from the "What's New" dialog, and
anything around the brackets ends up in the version key.

The headings are found the way env.py finds them (h2 elements of the rendered markdown, so a
setext heading counts and a fenced one does not), and the dict env.py built on import is
compared against them.

Discriminates: passes on bbfa876af; a heading without its date fails the shape test and the
import, and a repeated or decorated version fails the duplicate, version and parsed-release tests.
"""

from __future__ import annotations

from pathlib import Path

import markdown
import pytest
from bs4 import BeautifulSoup
from packaging.version import InvalidVersion, Version


@pytest.fixture(scope="module")
def release_headings(open_webui_backend: Path) -> list[str]:
    changelog = open_webui_backend.parent / "CHANGELOG.md"
    assert changelog.is_file(), f"no CHANGELOG.md at {changelog}; retarget at what env.py reads"
    html = markdown.markdown(changelog.read_text(encoding="utf-8"))
    headings = [h2.get_text().strip() for h2 in BeautifulSoup(html, "html.parser").find_all("h2")]
    assert headings, "CHANGELOG.md renders no h2 headings"
    return headings


def _version(heading: str) -> str:
    return heading.split(" - ")[0][1:-1]


def test_every_release_heading_carries_a_bracketed_version_and_a_date(release_headings):
    malformed = [
        heading
        for heading in release_headings
        if len(heading.split(" - ")) < 2
        or not heading.split(" - ")[0].startswith("[")
        or not heading.split(" - ")[0].endswith("]")
        or not heading.split(" - ")[1].strip()
    ]
    assert not malformed, (
        f"release headings open_webui.env cannot parse on import: {malformed}. "
        "Expected: ## [0.0.0] - YYYY-MM-DD"
    )


def test_every_version_is_a_version_number(release_headings):
    invalid = []
    for heading in release_headings:
        try:
            Version(_version(heading))
        except InvalidVersion:
            invalid.append(heading)
    assert not invalid, f"release headings whose bracketed part is not a version: {invalid}"


def test_no_version_is_released_twice(release_headings):
    versions = [_version(heading) for heading in release_headings]
    repeated = sorted({version for version in versions if versions.count(version) > 1})
    assert not repeated, f"versions with more than one heading, one of them dropped: {repeated}"


def test_env_parsed_every_release(owui_module, release_headings):
    """The dict the "What's New" dialog renders holds one dated entry per release heading."""
    parsed = owui_module("open_webui.env").CHANGELOG

    assert list(parsed) == [_version(heading) for heading in release_headings]
    assert all(entry.get("date") for entry in parsed.values())
