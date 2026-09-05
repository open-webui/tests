"""Guard: the alembic revision graph must stay a single unbroken line.

Every startup runs `alembic upgrade head` (open_webui.config.run_migrations),
and since v0.11.3 a failure there aborts the boot instead of being swallowed.
So a malformed revision graph is not a migration inconvenience, it is a dead
instance for everyone who upgrades.

The failure modes here are all merge accidents rather than logic bugs, which
is exactly why nothing else catches them: two branches each add a revision on
top of the same parent and alembic refuses to run at all ("Multiple head
revisions are present"), a rebase rewrites a revision id and leaves the child
pointing at a parent that no longer exists, or a copy-pasted template keeps
the revision id of the file it was copied from.

Parsed with ast, never imported: this must stay runnable without the backend's
dependencies installed, and importing a revision module runs nothing useful.

test_lifecycle.py already drives a real `upgrade head` / `downgrade base`
against SQLite and Postgres. This file is the cheap structural half — it names
the exact broken edge instead of reporting that alembic exited non-zero.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

VERSIONS_DIR = Path("open_webui") / "migrations" / "versions"


class Revision:
    """One alembic revision file, read without importing it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigned: dict[str, object] = {}
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target.id, node.value
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                target, value = node.targets[0].id, node.value
            else:
                continue
            try:
                assigned[target] = ast.literal_eval(value) if value is not None else None
            except ValueError:
                assigned[target] = "<not a literal>"
        self.revision = assigned.get("revision")
        self.down_revision = assigned.get("down_revision")
        self.functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.fixture(scope="module")
def revisions(open_webui_backend: Path) -> list[Revision]:
    versions = open_webui_backend / VERSIONS_DIR
    if not versions.is_dir():
        pytest.skip(f"no alembic versions directory at {versions}")
    files = sorted(p for p in versions.glob("*.py") if p.name != "__init__.py")
    assert files, f"no revision files under {versions}"
    return [Revision(p) for p in files]


def test_every_revision_declares_a_string_revision_id(revisions: list[Revision]) -> None:
    bad = [r.name for r in revisions if not isinstance(r.revision, str) or not r.revision]
    assert not bad, f"revision files with a missing or non-literal `revision`: {bad}"


def test_revision_ids_are_unique(revisions: list[Revision]) -> None:
    """A copy-pasted revision id makes alembic resolve the wrong file."""
    seen: dict[object, list[str]] = {}
    for rev in revisions:
        seen.setdefault(rev.revision, []).append(rev.name)
    duplicates = {rev_id: names for rev_id, names in seen.items() if len(names) > 1}
    assert not duplicates, f"duplicate revision ids: {duplicates}"


def test_the_filename_still_carries_its_revision_id(revisions: list[Revision]) -> None:
    """`alembic revision` names the file after the id; renaming one by hand
    makes the graph unreadable to anyone grepping for a revision."""
    mismatched = [r.name for r in revisions if not r.name.startswith(f"{r.revision}_")]
    assert not mismatched, f"filenames that do not start with their revision id: {mismatched}"


def test_exactly_one_base_revision(revisions: list[Revision]) -> None:
    bases = [r.name for r in revisions if r.down_revision is None]
    assert len(bases) == 1, f"expected one revision with down_revision = None, got: {bases}"


def test_exactly_one_head_revision(revisions: list[Revision]) -> None:
    """Two heads is the merge accident that stops every upgrade dead:
    `alembic upgrade head` raises "Multiple head revisions are present"
    and, since v0.11.3, that aborts startup."""
    parents = {r.down_revision for r in revisions}
    heads = sorted(str(r.revision) for r in revisions if r.revision not in parents)
    assert len(heads) == 1, (
        f"expected a single head, found {len(heads)}: {heads}. "
        "Merge the branches with `alembic merge` or re-point one down_revision."
    )


def test_no_revision_points_at_a_parent_that_does_not_exist(revisions: list[Revision]) -> None:
    known = {r.revision for r in revisions}
    dangling = [
        (r.name, r.down_revision)
        for r in revisions
        if r.down_revision is not None and r.down_revision not in known
    ]
    assert not dangling, f"down_revision values with no matching revision file: {dangling}"


def test_no_revision_is_its_own_parent(revisions: list[Revision]) -> None:
    self_parented = [r.name for r in revisions if r.down_revision == r.revision]
    assert not self_parented, f"revisions pointing at themselves: {self_parented}"


def test_the_chain_reaches_every_revision(revisions: list[Revision]) -> None:
    """Walking head to base must visit all of them. An island of revisions
    that link to each other but not to the main line never runs."""
    parents = {r.down_revision for r in revisions}
    heads = [r for r in revisions if r.revision not in parents]
    if len(heads) != 1:
        pytest.skip("multiple heads — test_exactly_one_head_revision reports this")

    by_id = {r.revision: r for r in revisions}
    walked: list[object] = []
    current: Revision | None = heads[0]
    while current is not None:
        if current.revision in walked:
            pytest.fail(f"cycle in the revision chain at {current.name}")
        walked.append(current.revision)
        current = by_id.get(current.down_revision) if current.down_revision else None

    unreachable = sorted(str(r.revision) for r in revisions if r.revision not in walked)
    assert not unreachable, f"revisions not on the head-to-base chain: {unreachable}"


def test_every_revision_defines_upgrade_and_downgrade(revisions: list[Revision]) -> None:
    """alembic calls both by name; a missing one raises only once that
    revision is reached, i.e. on somebody's production upgrade."""
    incomplete = [
        (r.name, sorted(r.functions))
        for r in revisions
        if not {"upgrade", "downgrade"} <= r.functions
    ]
    assert not incomplete, f"revisions missing upgrade()/downgrade(): {incomplete}"
