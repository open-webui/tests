"""Guard: CHANGELOG.md must stay parseable by open_webui.env at import time.

env.py reads CHANGELOG.md at module level and turns it into the JSON the
"What's New" dialog renders. The parse is unguarded:

    version_number = version.get_text().strip().split(' - ')[0][1:-1]
    date = version.get_text().strip().split(' - ')[1]

A release heading without the ` - <date>` half raises IndexError inside
`import open_webui.env`, which every other backend module imports. That is a
release-note typo taking the whole application down before it can log
anything — and the file is edited by hand for every release.

A source audit, not an import: this asserts the shape env.py depends on
without paying for the import (or its side effects).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The shape env.py's parse assumes: "## [0.11.3] - 2026-08-31".
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\] - (?P<date>.+)$")


@pytest.fixture(scope="module")
def changelog_headings(open_webui_backend: Path) -> list[str]:
    """Every markdown h2 in CHANGELOG.md, code fences excluded."""
    changelog = open_webui_backend.parent / "CHANGELOG.md"
    if not changelog.is_file():
        pytest.skip(f"no CHANGELOG.md at {changelog}")

    headings = []
    in_fence = False
    for line in changelog.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("## "):
            headings.append(line.rstrip())
    assert headings, "CHANGELOG.md contains no version headings"
    return headings


def test_every_version_heading_carries_a_date(changelog_headings: list[str]) -> None:
    """The ` - <date>` half is what env.py indexes into; without it the
    backend raises IndexError on import and the app never starts."""
    malformed = [heading for heading in changelog_headings if not HEADING.match(heading)]
    assert not malformed, (
        f"CHANGELOG.md headings env.py cannot parse: {malformed}. "
        "Expected the form: ## [0.0.0] - YYYY-MM-DD"
    )


def test_no_version_is_released_twice(changelog_headings: list[str]) -> None:
    """env.py keys the changelog dict by version, so a repeated heading
    silently drops one of the two entries from the release notes."""
    versions = [
        match.group("version")
        for match in (HEADING.match(heading) for heading in changelog_headings)
        if match
    ]
    duplicates = sorted({version for version in versions if versions.count(version) > 1})
    assert not duplicates, f"versions appearing more than once in CHANGELOG.md: {duplicates}"


def test_version_headings_are_plain_version_numbers(changelog_headings: list[str]) -> None:
    """The bracketed half is sliced with [1:-1] and used verbatim as a key,
    so any decoration around it ends up in the rendered release notes."""
    odd = []
    for heading in changelog_headings:
        match = HEADING.match(heading)
        if match and not re.fullmatch(r"\d+\.\d+\.\d+[A-Za-z0-9.\-]*", match.group("version")):
            odd.append(heading)
    assert not odd, f"CHANGELOG.md headings whose version is not a plain version number: {odd}"
