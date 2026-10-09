"""Skill rows read and written straight in a database no backend is serving.

`copy_skill_rows(database_url, source_id, copies)` adds skills cloned from one a release saved,
with its access grants, the way a long-used install holds hundreds of them: each copy is a dict
of the columns it changes (`id` and `name` at least). `skill_state(database_url)` reads every
skill, history entry and skill grant as stored, keyed so two reads compare whole.
`add_long_used_skills(database_url)` adds 300 skills of the data sets' alice, every tenth shared
like her trip planner, one of 12 MiB (above the 10 MiB a skill file may have since the skill
history migration) and a second skill of the deleted account.
"""

from __future__ import annotations

import uuid

import sqlalchemy


def _table(connection, name: str) -> sqlalchemy.Table:
    return sqlalchemy.Table(name, sqlalchemy.MetaData(), autoload_with=connection)


def copy_skill_rows(database_url: str, source_id: str, copies: list[dict]) -> None:
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.begin() as connection:
            skill, grant = _table(connection, "skill"), _table(connection, "access_grant")
            source = connection.execute(skill.select().where(skill.c.id == source_id)).mappings()
            source = dict(source.one())
            grants = connection.execute(
                grant.select().where(
                    grant.c.resource_type == "skill", grant.c.resource_id == source_id
                )
            ).mappings()
            grants = [dict(row) for row in grants]
            connection.execute(skill.insert(), [{**source, **copy} for copy in copies])
            copied_grants = [
                {**row, "id": str(uuid.uuid4()), "resource_id": copy["id"]}
                for copy in copies
                for row in grants
            ]
            if copied_grants:
                connection.execute(grant.insert(), copied_grants)
    finally:
        engine.dispose()


def skill_state(database_url: str) -> dict:
    """`skills` by id, `history` by id and the skill `grants` as sorted tuples."""
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.connect() as connection:
            tables = set(sqlalchemy.inspect(connection).get_table_names())
            skill, grant = _table(connection, "skill"), _table(connection, "access_grant")
            skills = {row["id"]: dict(row) for row in connection.execute(skill.select()).mappings()}
            history = {}
            if "skill_history" in tables:
                entries = connection.execute(_table(connection, "skill_history").select())
                history = {row["id"]: dict(row) for row in entries.mappings()}
            grants = connection.execute(
                sqlalchemy.select(
                    grant.c.resource_id,
                    grant.c.principal_type,
                    grant.c.principal_id,
                    grant.c.permission,
                ).where(grant.c.resource_type == "skill")
            )
            return {
                "skills": skills,
                "history": history,
                "grants": sorted(tuple(row) for row in grants),
            }
    finally:
        engine.dispose()


BULK = 300
LARGE_LINE = "keep this line exactly as it was written\n"
LARGE_BYTES = 12 * 1024 * 1024


def _bulk_content(number: int) -> str:
    return f"# Bulk {number}\n\n" + f"Check tide table {number}.\n" * (number % 17 + 1)


def _large_content() -> str:
    lines = LARGE_BYTES // (len(LARGE_LINE) + 8) + 1
    return "".join(f"{line:07d} {LARGE_LINE}" for line in range(lines))


def add_long_used_skills(database_url: str) -> dict[str, str]:
    """Skills cloned from the saved ones, as `upgrade_data` names them; returns their content."""
    added = {f"bulk-{number:03d}": _bulk_content(number) for number in range(BULK)}
    copies = [
        {"id": skill_id, "name": f"Bulk skill {skill_id[-3:]}", "content": content}
        for skill_id, content in added.items()
    ]
    copy_skill_rows(database_url, "trip-planner", copies[::10])
    copy_skill_rows(database_url, "uebersetzer", [c for c in copies if c not in copies[::10]])
    added["large-handbook"] = _large_content()
    large = {"id": "large-handbook", "name": "Large handbook", "content": added["large-handbook"]}
    copy_skill_rows(database_url, "old-habits", [large])
    copy_skill_rows(database_url, "dave-notes", [{"id": "orphan-copy", "name": "Orphan copy"}])
    added["orphan-copy"] = "Hand over to the next shift at six."
    return added
