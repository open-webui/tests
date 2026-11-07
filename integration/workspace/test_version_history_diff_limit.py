"""Journey: comparing two versions refuses a version longer than 5000 lines, in every history.

Since 77959080e (#33865) the version history of skills, tools, functions and prompts compares two
versions only when neither holds more than 5000 lines; a longer one is answered with 413 and a
message naming the limit, so one oversized version cannot tie up the server with a huge diff. A
version of exactly 5000 lines is still compared, and the oversized version itself stays readable
from the history.

Discriminates: passes on dev f659c0d12; with the line check taken out of `unified_text_diff` in a
backend copy (77959080e undone) every over-the-limit case answers 200 with the full diff. The
5000-line cases pass on both.
"""

from __future__ import annotations

import textwrap
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import pytest

from harness import skill_files
from harness.actors import Actor
from harness.plugins import installed_function

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

LIMIT = 5000
TOOL = textwrap.dedent(
    '''\
    class Tools:
        def shout(self, text: str) -> str:
            """Say the text back.

            :param text: what to say
            """
            return text
    '''
)
PIPE = textwrap.dedent(
    """\
    class Pipe:
        def pipe(self, body):
            return "answered"
    """
)


def _padded(head: str, lines: int, comment: str = "# ") -> str:
    """`head` followed by numbered lines up to `lines` lines in all, the last one marked."""
    head_lines = head.splitlines()
    filler = [f"{comment}line {number}" for number in range(len(head_lines) + 1, lines)]
    return "\n".join([*head_lines, *filler, f"{comment}last line of {lines}"]) + "\n"


@dataclass
class Compared:
    """A diff request between the short first version and the long second one."""

    path: str
    params: dict
    read_long: Callable[[], str]


@contextmanager
def _tool_versions(owner: Actor, lines: int) -> Iterator[Compared]:
    tool_id = f"diff_{uuid.uuid4().hex[:8]}"
    form = {"id": tool_id, "name": "Echo", "meta": {"description": "diff limit"}}
    long_source = _padded(TOOL, lines)
    with owner.client() as client:
        created = client.post("/api/v1/tools/create", json={**form, "content": TOOL})
        assert created.status_code == 200, created.text
        saved = client.post(
            f"/api/v1/tools/id/{tool_id}/update", json={**form, "content": long_source}
        )
        assert saved.status_code == 200, saved.text
    long_id = saved.json()["version_id"]

    def read_long() -> str:
        with owner.client() as client:
            read = client.get(f"/api/v1/tools/id/{tool_id}/history/{long_id}")
        assert read.status_code == 200, read.text
        return read.json()["snapshot"]["content"]

    try:
        yield Compared(
            f"/api/v1/tools/id/{tool_id}/history/diff",
            {"from_id": created.json()["version_id"], "to_id": long_id},
            read_long,
        )
    finally:
        with owner.client() as client:
            client.delete(f"/api/v1/tools/id/{tool_id}/delete")


@contextmanager
def _function_versions(owner: Actor, lines: int) -> Iterator[Compared]:
    long_source = _padded(PIPE, lines)
    with installed_function(owner, PIPE) as pipe_id, owner.client() as client:
        first = client.get(f"/api/v1/functions/id/{pipe_id}")
        assert first.status_code == 200, first.text
        saved = client.post(
            f"/api/v1/functions/id/{pipe_id}/update",
            json={
                "id": pipe_id,
                "name": pipe_id,
                "content": long_source,
                "meta": {"description": "installed by a regression test"},
            },
        )
        assert saved.status_code == 200, saved.text
        long_id = saved.json()["version_id"]

        def read_long() -> str:
            read = client.get(f"/api/v1/functions/id/{pipe_id}/history/{long_id}")
            assert read.status_code == 200, read.text
            return read.json()["snapshot"]["content"]

        yield Compared(
            f"/api/v1/functions/id/{pipe_id}/history/diff",
            {"from_id": first.json()["version_id"], "to_id": long_id},
            read_long,
        )


@contextmanager
def _prompt_versions(owner: Actor, lines: int) -> Iterator[Compared]:
    command = f"diff-{uuid.uuid4().hex[:8]}"
    form = {"command": command, "name": "Long prompt"}
    long_content = _padded("Write it all down.", lines, comment="")
    with owner.client() as client:
        created = client.post(
            "/api/v1/prompts/create", json={**form, "content": "Write it all down."}
        )
        assert created.status_code == 200, created.text
        prompt_id = created.json()["id"]
        saved = client.post(
            f"/api/v1/prompts/id/{prompt_id}/update", json={**form, "content": long_content}
        )
        assert saved.status_code == 200, saved.text
    long_id = saved.json()["version_id"]

    def read_long() -> str:
        with owner.client() as client:
            entries = client.get(f"/api/v1/prompts/id/{prompt_id}/history")
        assert entries.status_code == 200, entries.text
        (entry,) = [entry for entry in entries.json() if entry["id"] == long_id]
        return entry["snapshot"]["content"]

    try:
        yield Compared(
            f"/api/v1/prompts/id/{prompt_id}/history/diff",
            {"from_id": created.json()["version_id"], "to_id": long_id},
            read_long,
        )
    finally:
        with owner.client() as client:
            client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


@contextmanager
def _skill_versions(owner: Actor, lines: int) -> Iterator[Compared]:
    skill_id = skill_files.new_id("diff")
    skill = skill_files.create(
        owner,
        {"SKILL.md": skill_files.skill_md(skill_id, "Keep the log."), "log.md": "Started.\n"},
        skill_id=skill_id,
    )
    long_log = _padded("Started.", lines, comment="")
    saved = skill_files.save(owner, skill, [{"op": "put", "path": "log.md", "content": long_log}])
    assert saved.status_code == 200, saved.text
    long_id = saved.json()["version_id"]
    try:
        yield Compared(
            f"/api/v1/skills/id/{skill_id}/history/diff/file",
            {"from_id": skill["version_id"], "to_id": long_id, "path": "log.md"},
            lambda: skill_files.read(owner, skill_id, "log.md", long_id).decode(),
        )
    finally:
        skill_files.delete_skills_of(owner)


HISTORIES = {
    "tool": _tool_versions,
    "function": _function_versions,
    "prompt": _prompt_versions,
    "skill file": _skill_versions,
}


@pytest.fixture
def owner(make_user) -> Actor:
    """An admin, who may keep all four kinds and read every history."""
    return make_user(role="admin")


def _diff_text(answer: dict) -> str:
    diff = answer.get("content_diff", answer.get("diff"))
    return "\n".join(diff) if isinstance(diff, list) else diff


@pytest.mark.parametrize("kind", HISTORIES)
def test_a_version_longer_than_5000_lines_is_not_compared(owner, kind):
    with HISTORIES[kind](owner, LIMIT + 1) as versions, owner.client() as client:
        compared = client.get(versions.path, params=versions.params)
        long_version = versions.read_long()

    assert compared.status_code == 413, (
        f"a {kind} version of {LIMIT + 1} lines was compared: HTTP {compared.status_code}, "
        f"{len(compared.text)} bytes"
    )
    assert f"{LIMIT} lines" in compared.json()["detail"]
    assert len(long_version.splitlines()) == LIMIT + 1, "the long version itself is unreadable"


@pytest.mark.parametrize("kind", HISTORIES)
def test_a_version_of_exactly_5000_lines_is_still_compared(owner, kind):
    with HISTORIES[kind](owner, LIMIT) as versions, owner.client() as client:
        compared = client.get(versions.path, params=versions.params)

    assert compared.status_code == 200, compared.text
    diff = _diff_text(compared.json())
    assert f"last line of {LIMIT}" in diff, f"the {kind} diff lacks the version's last line"
