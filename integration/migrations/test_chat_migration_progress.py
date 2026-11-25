"""Journey: an admin upgrading a big install sees the chat message move make progress in the log.

The migration that moves chat messages into rows of their own (`5e7a2c9d4b13`, PR
open-webui/open-webui#34689) works through every chat of an install, which takes minutes on a big
one. What an admin expects at the default log level: after the log names that upgrade step, it
keeps reporting how many chats are done, rising until every chat is, so the upgrade does not look
stuck and nobody restarts the container in the middle. The data set is taken to the revision before
the migration with the manual `alembic` command and gets a few thousand more conversations, copies
of one of alice's, then the checkout starts on it with no log settings, as the Docker image does.

Discriminates: on dev f80a4f782 the log shows only `Running upgrade d2f86c0b41ae -> 5e7a2c9d4b13`
and no progress at all, so every test fails (open-webui/open-webui#35052); with the migration
logging configured to keep its info lines, all pass on SQLite and Postgres.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
import sqlalchemy

from harness.prepared_data import ReleaseData, manual_alembic
from harness.upgraded_release import upgraded_release

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

BEFORE = "5e7a2c9d4b13-1"
COPIES = 3000
STEP = re.compile(r"Running upgrade \S+ -> 5e7a2c9d4b13")
NEXT_STEP = re.compile(r"Running upgrade 5e7a2c9d4b13 -> ")


def _add_copies(release: ReleaseData) -> None:
    """Upgrade to the revision before the move and copy alice's longest chat `COPIES` times."""
    finished = manual_alembic(
        release.data_dir, "upgrade", BEFORE, database_url=release.database_url
    )
    assert finished.returncode == 0, f"alembic upgrade {BEFORE} failed:\n{finished.stderr[-3000:]}"
    alice_id = release.manifest["accounts"]["alice"]["id"]
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.begin() as connection:
            chats = sqlalchemy.Table("chat", sqlalchemy.MetaData(), autoload_with=connection)
            message_table = sqlalchemy.Table(
                "chat_message", sqlalchemy.MetaData(), autoload_with=connection
            )
            longest = connection.execute(
                sqlalchemy.select(message_table.c.chat_id)
                .join(chats, chats.c.id == message_table.c.chat_id)
                .where(chats.c.user_id == alice_id)
                .group_by(message_table.c.chat_id)
                .order_by(sqlalchemy.func.count().desc())
                .limit(1)
            ).scalar_one()
            source = dict(
                connection.execute(sqlalchemy.select(chats).where(chats.c.id == longest))
                .mappings()
                .one()
            )
            source_rows = [
                dict(row)
                for row in connection.execute(
                    sqlalchemy.select(message_table).where(message_table.c.chat_id == longest)
                ).mappings()
            ]
            new_chats, new_rows = [], []
            for number in range(COPIES):
                chat_id = str(uuid.uuid4())
                new_chats.append({**source, "id": chat_id, "share_id": None})
                new_rows.extend(
                    {
                        **row,
                        "id": f"{chat_id}-{row['id'].removeprefix(f'{longest}-')}",
                        "chat_id": chat_id,
                    }
                    for row in source_rows
                )
            connection.execute(chats.insert(), new_chats)
            connection.execute(message_table.insert(), new_rows)
    finally:
        engine.dispose()


def _chat_count(release: ReleaseData) -> int:
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.connect() as connection:
            return connection.execute(sqlalchemy.text("SELECT COUNT(*) FROM chat")).scalar_one()
    finally:
        engine.dispose()


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("v0.13.4-sqlite", id="v0.13.4-sqlite"),
        pytest.param(
            "v0.13.4-postgres", marks=pytest.mark.requires_postgres, id="v0.13.4-postgres"
        ),
    ],
)
def upgrade_log(request, tmp_path_factory) -> Iterator[tuple[list[str], int]]:
    """The log lines of the chat message move, and how many chats the install holds."""
    prepared: list[ReleaseData] = []

    def prepare(release: ReleaseData) -> None:
        _add_copies(release)
        prepared.append(release)

    with upgraded_release(
        request.param, tmp_path_factory.mktemp(request.param), prepare=prepare
    ) as upgraded:
        lines = upgraded.backend.log().splitlines()
        total = _chat_count(prepared[0])
    started = [index for index, line in enumerate(lines) if STEP.search(line)]
    assert started, "the log never names the upgrade step that moves chat messages"
    step_lines = lines[started[0] + 1 :]
    ended = [index for index, line in enumerate(step_lines) if NEXT_STEP.search(line)]
    yield (step_lines[: ended[0]] if ended else step_lines), total


def test_the_log_reports_progress_while_chats_are_moved(upgrade_log):
    step_lines, total = upgrade_log
    progress = [line for line in step_lines if "chat" in line.lower() and re.search(r"\d", line)]
    assert len(progress) >= 2, (
        f"moving the messages of {total} chats logged no running progress at the default log "
        f"level, only:\n" + "\n".join(step_lines[:20])
    )


def test_the_reported_progress_rises_until_every_chat_is_done(upgrade_log):
    step_lines, total = upgrade_log
    counts = [
        int(found)
        for line in step_lines
        if "chat" in line.lower()
        for found in re.findall(r"\b(\d+)\b", line)[-1:]
    ]
    assert counts, f"the log reports no count of moved chats for {total} chats"
    assert counts == sorted(counts) and len(set(counts)) == len(counts), (
        f"the reported counts do not rise steadily: {counts[:20]}"
    )
    assert counts[-1] == total, (
        f"the last reported count is {counts[-1]}, the install holds {total} chats"
    )
