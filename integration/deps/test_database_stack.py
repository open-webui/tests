"""Dependency smoke: the database drivers every request reads and writes through.

Every request's query runs on SQLAlchemy's async engine. On SQLite that drives aiosqlite
(`sqlite://` becomes `sqlite+aiosqlite://`); on Postgres it drives psycopg 3
(`postgresql+psycopg://`), while the synchronous engine that migrates and loads the settings at
boot drives psycopg2. A note saved, changed and deleted over the API must read back from the next
request exactly as written, and writes sent at once must all land. A bump that breaks a driver's
commit, parameter binding or row access loses the write or garbles what reads back. Each test runs
on SQLite (skipped when the whole run is on Postgres) and on a Postgres database of its own on the
embedded server, whose URL names SSL the way some ORMs write it (`ssl=disable`): Open WebUI hands
it to psycopg2 and psycopg as libpq's `sslmode`, which neither would take in the bare form.
pgvector's use of psycopg2 is driven in test_vector_stores.py, Alembic's migrations in
integration/migrations. Twin of unit/deps/test_psycopg.py and unit/deps/test_psycopg2_binary.py.

SQLAlchemy builds every one of those queries. A chat is stored whole in a native `JSON` column,
through the engine's JSON codec, and a model preset's settings in a `JSONField` (a
`TypeDecorator` over text); both must come back exactly, nulls and nesting included. The admin's
user search is an `ilike` over name and email, counted with `func.count` over the filtered
statement and cut into pages of 30 with `order_by`, `offset` and `limit`. On SQLite a `connect`
event hook on the sync and the async engine sets each new connection's PRAGMAs, among them the
journal mode `DATABASE_ENABLE_SQLITE_WAL` chooses, which the database file keeps.

`DATABASE_SCHEMA` gives SQLAlchemy's `MetaData` a schema, so every query names its tables as
`<schema>.<table>`, and the Alembic migrations create the tables in that schema too. They used to
take no notice of it and create them in the connection's default schema, so a fresh Postgres
install with `DATABASE_SCHEMA` set stopped at boot looking for `<schema>.config` (PR
open-webui/open-webui#31533, closed), fixed in dev 65f44053d.

Discriminates: passes on dev ef67cc3fa; in a backend copy with `aiosqlite.Connection.commit`
made a no-op every write is lost, down to the admin account the boot signs up, so both SQLite
cases fail. With `psycopg.AsyncConnection.commit` made a no-op both Postgres cases fail, and
with the bare `ssl` key passed on untranslated the Postgres instance no longer boots. A
`JSONField` handing back the stored text undecoded fails the model preset test (an engine JSON
codec that loses nulls stops the first sign-up, so the chat test cannot be singled out), the
user search without its `offset`
(every page the first) fails the search test, each on both databases, `like` in place of `ilike`
fails it on Postgres (Open WebUI's own SQLite `like` folds case anyway), and the connect hooks
left unregistered fail the WAL case. The schema test passes on dev 1711059db and fails on dev
56b6b660a, before 65f44053d.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
import uuid

import pytest

from harness import backends
from harness.actors import Actor, admin_of, create_user
from harness.instance import LaunchedInstance
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

# quotes, a placeholder look-alike and characters beyond ASCII, so binding and decoding show
AWKWARD_TEXT = 'O\'Brien said "?" and :name; Grüße, 日本語, emoji \U0001f99c'
PARALLEL_WRITES = 12


def _create_note(client, title: str, markdown: str) -> dict:
    created = client.post(
        "/api/v1/notes/create", json={"title": title, "data": {"content": {"md": markdown}}}
    )
    assert created.status_code == 200, created.text
    return created.json()


def _stored_markdown(client, note_id: str) -> str:
    stored = client.get(f"/api/v1/notes/{note_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["data"]["content"]["md"]


@pytest.fixture(scope="module")
def postgres_url():
    pytest.importorskip("pgserver", reason="the Postgres case runs on the embedded server")
    with backends.postgres_database() as url:
        yield url


@pytest.fixture(params=["sqlite", "postgres"])
def on_database(request, instance_with) -> LaunchedInstance:
    """An instance keeping its data in the parametrised database."""
    if request.param == "sqlite":
        if backends.DATABASE == "postgres":
            pytest.skip("this run keeps every instance on Postgres")
        return request.getfixturevalue("instance")
    url = request.getfixturevalue("postgres_url")
    separator = "&" if "?" in url else "?"
    return instance_with({"DATABASE_URL": f"{url}{separator}ssl=disable"})


@pytest.fixture
def author(on_database) -> Actor:
    """A fresh account on that instance."""
    return create_user(on_database)


def test_a_note_reads_back_changed_and_deleted_as_written(author):
    with author.client() as client:
        note = _create_note(client, f"awkward {uuid.uuid4().hex[:6]}", AWKWARD_TEXT)
        assert _stored_markdown(client, note["id"]) == AWKWARD_TEXT

        updated = client.post(
            f"/api/v1/notes/{note['id']}/update",
            json={"title": note["title"], "data": {"content": {"md": AWKWARD_TEXT * 2}}},
        )
        assert updated.status_code == 200, updated.text
        assert _stored_markdown(client, note["id"]) == AWKWARD_TEXT * 2

        deleted = client.delete(f"/api/v1/notes/{note['id']}/delete")
        assert deleted.status_code == 200 and deleted.json() is True, deleted.text
        assert client.get(f"/api/v1/notes/{note['id']}").status_code == 404


def test_notes_written_at_once_all_land(author):
    titles = [f"parallel {index} {uuid.uuid4().hex[:6]}" for index in range(PARALLEL_WRITES)]

    with author.client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            created = list(pool.map(lambda title: _create_note(client, title, title), titles))
        listed = client.get("/api/v1/notes/")

    assert listed.status_code == 200, listed.text
    assert {note["title"] for note in listed.json()} == set(titles)
    assert len({note["id"] for note in created}) == PARALLEL_WRITES


# nested objects and lists, null, a float, a boolean and text beyond ASCII
NESTED_CHAT = {
    "title": "Tide table",
    "models": ["mock-model"],
    "params": {"temperature": 0.25, "stop": None, "seed": 7},
    "messages": [
        {"id": "m1", "role": "user", "content": AWKWARD_TEXT, "files": []},
        {"id": "m2", "role": "assistant", "content": "Hochwasser um 4:25", "done": True},
    ],
    "tags": [],
    "meta": {"nested": {"deeper": [1, [2, {"three": None}]]}},
}


def test_a_chat_keeps_its_nested_json_exactly(author):
    with author.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": NESTED_CHAT})
        assert created.status_code == 200, created.text
        stored = client.get(f"/api/v1/chats/{created.json()['id']}")

    assert stored.status_code == 200, stored.text
    assert {key: stored.json()["chat"].get(key) for key in NESTED_CHAT} == NESTED_CHAT


def test_a_model_presets_json_settings_read_back_exactly(on_database):
    model_id = f"tides-{uuid.uuid4().hex[:8]}"
    meta = {"description": AWKWARD_TEXT, "tags": [{"name": "tides"}], "chart": NESTED_CHAT["meta"]}
    params = {"temperature": 0.25, "stop": ["\n\n"], "seed": None}
    with admin_of(on_database).client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": "Tides",
                "meta": meta,
                "params": params,
            },
        )
        assert created.status_code == 200, created.text
        stored = client.get("/api/v1/models/model", params={"id": model_id})
        client.post("/api/v1/models/model/delete", json={"id": model_id})

    assert stored.status_code == 200, stored.text
    assert {key: stored.json()["meta"].get(key) for key in meta} == meta
    assert stored.json()["params"] == params


def test_a_user_search_is_counted_ordered_and_paged(on_database):
    token = uuid.uuid4().hex[:10]
    names = [f"Pier {token} {index:02d}" for index in range(35)]
    for name in reversed(names):
        create_user(on_database, name=name)

    def page(number: int) -> dict:
        found = client.get(
            "/api/v1/users/",
            params={"query": token.upper(), "order_by": "name", "direction": "asc", "page": number},
        )
        assert found.status_code == 200, found.text
        return found.json()

    with admin_of(on_database).client() as client:
        first, second = page(1), page(2)

    assert first["total"] == second["total"] == 35
    assert [user["name"] for user in first["users"] + second["users"]] == names


@pytest.mark.slow
@pytest.mark.parametrize("wal", [True, False], ids=["wal", "rollback-journal"])
def test_every_sqlite_connection_gets_the_configured_journal_mode(instance_with, wal):
    if backends.DATABASE == "postgres":
        pytest.skip("this run keeps every instance on Postgres")
    launched = instance_with({"DATABASE_ENABLE_SQLITE_WAL": "true" if wal else "false"})
    with admin_of(launched).client() as client:
        _create_note(client, "journal", "written")

    with sqlite3.connect(launched.data_dir / "webui.db") as database:
        [(mode,)] = database.execute("PRAGMA journal_mode").fetchall()

    assert mode == ("wal" if wal else "delete")


def _tables_by_schema(database_url: str) -> dict[str, int]:
    import sqlalchemy

    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.begin() as connection:
            rows = connection.execute(
                sqlalchemy.text(
                    "SELECT table_schema, count(*) FROM information_schema.tables"
                    " WHERE table_name IN ('config', 'note', 'user') GROUP BY table_schema"
                )
            ).fetchall()
    finally:
        engine.dispose()
    return dict(rows)


@pytest.mark.slow
def test_a_fresh_postgres_install_keeps_its_tables_in_the_database_schema(instance_with):
    pytest.importorskip("pgserver", reason="the schema case runs on the embedded server")
    with backends.postgres_database() as url:
        import sqlalchemy

        engine = sqlalchemy.create_engine(url)
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE SCHEMA harbour"))
        engine.dispose()
        try:
            launched = instance_with({"DATABASE_URL": url, "DATABASE_SCHEMA": "harbour"})
        except pytest.fail.Exception:
            launched = None
        tables = _tables_by_schema(url)

        assert launched is not None and tables == {"harbour": 3}, (
            "with DATABASE_SCHEMA set, the migrations create the tables in the default schema "
            f"({tables}) while the app reads them from harbour, so a fresh install never boots"
        )
        with create_user(launched).client() as client:
            note = _create_note(client, "schema", "kept in harbour")
            assert _stored_markdown(client, note["id"]) == "kept in harbour"
