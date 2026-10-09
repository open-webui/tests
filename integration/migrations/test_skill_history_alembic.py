"""Journey: the skill history migration steps down and up again without losing a skill.

The skill history migration (`d6a8c3f912ab`, upstream 9bbb95048) adds `version_id` and `data` to
the skill table and a `skill_history` table, filled with one first version per skill. The
operator's manual `alembic` commands run on a copy of each data set under `upgrade_data/`, whose
skills an older release saved, with the 300 more, the 12 MiB one and the orphaned one
`harness.skill_rows` adds. The chain has a single head, and a second `upgrade head` changes
nothing. Stepping down to the revision before it gives back the skill table exactly as the
release left it (every row, column, constraint and index, every grant) with no history table, and
upgrading again starts every skill's history afresh with one first version. On SQLite the step
down rebuilds the skill table, which is where rows or the unique name could be lost.

Discriminates: passes on dev 178de3666 for all six data sets. In backend copies of it, run on the
v0.10.2 SQLite and v0.11.4 Postgres sets: a second migration file branching off `b8e4f0a3c752`
fails the single-head test (and `upgrade head` refuses the two heads); a `downgrade()` that
leaves `skill_history` behind fails every set at the second upgrade (the table already exists);
one that recreates the skill table empty in the old shape fails the step-down and step-up tests;
a backfill that writes no `version_id` fails the step-up test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pytest
import sqlalchemy

from harness.prepared_data import manual_alembic, release_data
from harness.skill_rows import add_long_used_skills, skill_state
from harness.upgraded_release import DATA_SETS, data_set_params

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

REVISION = "d6a8c3f912ab"


def _schema(database_url: str) -> dict:
    """The skill table's columns, keys and indexes, as the database reports them."""
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.connect() as connection:
            inspector = sqlalchemy.inspect(connection)
            return {
                "tables": sorted(inspector.get_table_names()),
                "columns": [
                    (column["name"], str(column["type"]), column["nullable"])
                    for column in inspector.get_columns("skill")
                ],
                "primary_key": inspector.get_pk_constraint("skill")["constrained_columns"],
                "unique": sorted(
                    tuple(unique["column_names"])
                    for unique in inspector.get_unique_constraints("skill")
                ),
                "indexes": sorted(
                    (index["name"], tuple(index["column_names"]), bool(index["unique"]))
                    for index in inspector.get_indexes("skill")
                ),
            }
    finally:
        engine.dispose()


@dataclass
class Steps:
    states: dict[str, dict]  # skill rows, history and grants after each step
    schemas: dict[str, dict]
    added: dict[str, str]


@pytest.fixture(scope="module", params=data_set_params())
def steps(request, tmp_path_factory) -> Iterator[Steps]:
    """Each data set taken up, up again, one step down past the migration and up once more."""
    name = request.param
    root = tmp_path_factory.mktemp(name)
    with release_data(DATA_SETS / f"{name}.tar.gz", root) as release:
        url = release.database_url
        added = add_long_used_skills(url)
        states, schemas = {}, {}

        def record(step: str) -> None:
            states[step], schemas[step] = skill_state(url), _schema(url)

        def alembic(*arguments: str) -> None:
            finished = manual_alembic(release.data_dir, *arguments, database_url=url)
            assert finished.returncode == 0, (
                f"alembic {' '.join(arguments)} failed on {name}:\n{finished.stderr[-3000:]}"
            )

        record("release")
        for step, arguments in (
            ("upgraded", ("upgrade", "head")),
            ("upgraded twice", ("upgrade", "head")),
            ("stepped down", ("downgrade", f"{REVISION}-1")),
            ("upgraded again", ("upgrade", "head")),
        ):
            alembic(*arguments)
            record(step)
        yield Steps(states, schemas, added)


def test_the_migration_chain_has_a_single_head(tmp_path):
    listed = manual_alembic(tmp_path, "heads")
    assert listed.returncode == 0, listed.stderr[-3000:]
    heads = [line for line in listed.stdout.splitlines() if line.strip()]
    assert len(heads) == 1, f"the chain branches into {len(heads)} heads: {heads}"


def test_a_second_upgrade_changes_nothing(steps):
    assert steps.states["upgraded twice"] == steps.states["upgraded"]
    assert steps.schemas["upgraded twice"] == steps.schemas["upgraded"]


def test_stepping_down_gives_back_the_skill_table_the_release_left(steps):
    release, stepped_down = steps.states["release"], steps.states["stepped down"]
    schemas = steps.schemas
    # an older release's set keeps the tables of the migrations between it and this one
    tables = set(schemas["upgraded"]["tables"]) - {"skill_history"}
    assert set(schemas["stepped down"]["tables"]) == tables
    skill_table = {key: value for key, value in schemas["release"].items() if key != "tables"}
    assert {key: schemas["stepped down"][key] for key in skill_table} == skill_table
    lost = [
        skill_id
        for skill_id, row in release["skills"].items()
        if stepped_down["skills"].get(skill_id) != row
    ]
    assert not lost, f"stepping down lost or changed {len(lost)} skills: {lost[:10]}"
    assert set(stepped_down["skills"]) == set(release["skills"])
    assert stepped_down["grants"] == release["grants"]


def test_upgrading_again_starts_every_history_afresh(steps):
    upgraded_again = steps.states["upgraded again"]
    skills, history = upgraded_again["skills"], upgraded_again["history"]
    assert {skill_id: skill["content"] for skill_id, skill in skills.items()} == {
        skill_id: skill["content"] for skill_id, skill in steps.states["release"]["skills"].items()
    }
    assert len(history) == len(skills)
    for skill_id, skill in skills.items():
        entry = history.get(skill["version_id"])
        assert entry is not None, f"{skill_id} has no first version after upgrading again"
        assert (entry["skill_id"], entry["parent_id"]) == (skill_id, None)
        assert skill["data"] == {"files": [{"path": "SKILL.md", "content": skill["content"]}]}
    assert steps.schemas["upgraded again"] == steps.schemas["upgraded"]
