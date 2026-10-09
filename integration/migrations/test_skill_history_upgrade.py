"""Journey: skills an older release saved come through the skill history migration whole.

The skill history migration (`d6a8c3f912ab`, upstream 9bbb95048) gives every skill a set of files
and a history of versions: each existing skill gets its instructions as the file `SKILL.md`, and
one first history entry holding the skill as it was, by its owner and dated to its last change,
which becomes the skill's current version. Each data set under `upgrade_data/` holds skills its
release saved through its own API: shared with the group, with a user for writing and with
everyone, switched off, without a description and with an empty one, non-ASCII text with CRLF
line endings, and one whose owner was deleted. Before the checkout first starts on a copy, 300
more skills cloned from saved ones are added (every tenth with the group and user grants), one
of 12 MiB, above the 10 MiB a file may have since, and a second one of the deleted account. The
start that migrates is read straight from the database and compared with the restart the tests
then use.

Discriminates: passes on dev 178de3666 for all six data sets. In backend copies of it, run on the
v0.10.2 SQLite and v0.11.4 Postgres sets: a migration writing `SKILL.md` empty fails the two
history tests, the bulk test, the first-save test and the rename test; one setting no
`version_id` fails the two history tests and the first-save test (the save is refused as a
conflict); one backfilling only the first 100 skills fails the history, access, bulk, group
mention, rename and orphan tests; one dating the first version to the migration fails the two
history tests; one also resetting `updated_at` fails the unchanged-columns and history tests;
one deleting the skills' grants fails the unchanged-columns, read-back, access, both mention and
first-save tests. A copy whose every startup writes another first version for each skill fails
the restart test, along with the history, first-save, rename and orphan tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.prepared_data import ReleaseData, serving
from harness.skill_rows import add_long_used_skills, skill_state
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

# the releases ran with it, so every account could chat with the provider's model
RELEASE_ENVIRONMENT = {"BYPASS_MODEL_ACCESS_CONTROL": "true"}
ORIGINAL_COLUMNS = ("id", "user_id", "name", "description", "content", "is_active")


@dataclass
class Migrated:
    upgraded: Upgraded
    database_url: str
    before: dict  # skill rows, history and grants as the release left them
    first_start: dict  # the same after the start that migrated
    restarted: dict  # the same after the start the tests use
    added: dict[str, str]  # the content of every skill added before the first start


@pytest.fixture(scope="module", params=data_set_params())
def migrated(request, tmp_path_factory) -> Iterator[Migrated]:
    name = request.param
    states: dict = {}

    def first_start(release: ReleaseData) -> None:
        states["added"] = add_long_used_skills(release.database_url)
        states["before"] = skill_state(release.database_url)
        with serving(release.data_dir, {"WEBUI_AUTH": "true", **release.settings}):
            pass
        states["first_start"] = skill_state(release.database_url)
        states["url"] = release.database_url

    root = tmp_path_factory.mktemp(name)
    with upgraded_release(name, root, RELEASE_ENVIRONMENT, prepare=first_start) as upgraded:
        restarted = skill_state(states["url"])
        yield Migrated(
            upgraded,
            states["url"],
            states["before"],
            states["first_start"],
            restarted,
            states["added"],
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_no_column_a_skill_had_changes_and_its_grants_stay(migrated):
    before, after = migrated.before, migrated.first_start
    assert before["history"] == {}, "the release already had skill history"
    assert set(after["skills"]) == set(before["skills"])
    changed = [
        skill_id
        for skill_id, row in before["skills"].items()
        if {column: after["skills"][skill_id][column] for column in row} != row
    ]
    assert not changed, f"the migration changed what these skills held: {changed[:10]}"
    assert after["grants"] == before["grants"], "the skills' access grants changed"


def test_every_skill_starts_its_history_with_one_version_as_it_was(migrated):
    skills, history = migrated.first_start["skills"], migrated.first_start["history"]
    assert len(history) == len(skills), f"{len(history)} history entries for {len(skills)} skills"
    for skill_id, skill in skills.items():
        entry = history.get(skill["version_id"])
        assert entry is not None, f"{skill_id} has no current version in its history"
        assert (entry["skill_id"], entry["parent_id"]) == (skill_id, None), skill_id
        assert entry["user_id"] == skill["user_id"], f"{skill_id}: the first version's author"
        assert entry["created_at"] == skill["updated_at"], f"{skill_id}: dated to the migration"
        files = [{"path": "SKILL.md", "content": skill["content"]}]
        assert skill["data"] == {"files": files}, f"{skill_id}: SKILL.md is not the instructions"
        snapshot = entry["snapshot"]
        kept = {key: snapshot[key] for key in ("name", "description", "content")}
        assert kept == {key: skill[key] for key in kept}, f"{skill_id}: the first version differs"
        assert snapshot["data"] == {"files": files}, skill_id


def test_a_restart_changes_nothing_the_migration_wrote(migrated):
    assert migrated.restarted == migrated.first_start


def test_saved_skills_read_back_as_their_release_stored_them(migrated):
    upgraded = migrated.upgraded
    for skill_id, saved in upgraded.manifest["skills"].items():
        opened = upgraded.get("admin", f"/api/v1/skills/id/{skill_id}")
        assert opened.status_code == 200, f"{skill_id}: {opened.text}"
        skill = opened.json()
        assert {column: skill[column] for column in ORIGINAL_COLUMNS} == {
            column: saved[column] for column in ORIGINAL_COLUMNS
        }, skill_id
        # releases before 0.11 stored no `i18n`, which the response now fills in as None
        assert {key: skill["meta"][key] for key in saved["meta"]} == saved["meta"], skill_id
        grants = sorted(
            (grant["principal_type"], grant["principal_id"], grant["permission"])
            for grant in skill["access_grants"]
        )
        assert grants == [tuple(grant.values()) for grant in saved["grants"]], skill_id


def test_readers_and_writers_keep_their_access(migrated):
    upgraded = migrated.upgraded
    for who in ("alice", "bob", "carol"):
        listed = {skill["id"] for skill in upgraded.get(who, "/api/v1/skills/").json()}
        for skill_id, saved in upgraded.manifest["skills"].items():
            opened = upgraded.get(who, f"/api/v1/skills/id/{skill_id}")
            if who in saved["readers"]:
                assert skill_id in listed, f"{who} lost {skill_id}"
                assert opened.status_code == 200, f"{who}: {opened.text}"
                assert opened.json()["write_access"] is (who in saved["writers"]), (
                    f"{who}'s write access to {skill_id} changed"
                )
            else:
                assert skill_id not in listed, f"{who} now sees {skill_id}"
                assert opened.status_code in (401, 403, 404), f"{who} now opens {skill_id}"


def test_the_first_version_holds_the_instructions_as_skill_md(migrated):
    upgraded = migrated.upgraded
    names = {account["id"]: account["name"] for account in upgraded.manifest["accounts"].values()}
    for skill_id, saved in upgraded.manifest["skills"].items():
        version_id = upgraded.get("admin", f"/api/v1/skills/id/{skill_id}").json()["version_id"]
        history = upgraded.get("admin", f"/api/v1/skills/id/{skill_id}/history").json()
        assert [(entry["id"], entry["parent_id"]) for entry in history] == [(version_id, None)]
        entry = history[0]
        assert (entry["user_id"], entry["created_at"]) == (saved["user_id"], saved["updated_at"])
        author = names.get(saved["user_id"])
        assert entry["user"] == ({"name": author} if author else None), skill_id

        files = upgraded.get("admin", f"/api/v1/skills/id/{skill_id}/files").json()
        size = len(saved["content"].encode("utf-8"))
        assert files["version_id"] == version_id
        assert files["files"] == [{"path": "SKILL.md", "size": size, "encoding": None}]
        content = upgraded.get(
            "admin",
            f"/api/v1/skills/id/{skill_id}/files/content",
            params={"path": "SKILL.md", "version_id": version_id},
        )
        assert content.status_code == 200, content.text
        assert content.content == saved["content"].encode("utf-8"), skill_id


def test_hundreds_of_added_skills_and_a_large_one_read_back_whole(migrated):
    upgraded = migrated.upgraded
    listed = {skill["id"] for skill in upgraded.get("alice", "/api/v1/skills/").json()}
    bulk = {skill_id for skill_id in migrated.added if skill_id.startswith("bulk-")}
    assert bulk <= listed, f"alice lost {len(bulk - listed)} of her {len(bulk)} skills"
    for skill_id in ("bulk-000", "bulk-151", "bulk-299", "large-handbook", "orphan-copy"):
        skill = upgraded.get("admin", f"/api/v1/skills/id/{skill_id}").json()
        content = upgraded.get(
            "admin",
            f"/api/v1/skills/id/{skill_id}/files/content",
            params={"path": "SKILL.md", "version_id": skill["version_id"]},
        )
        assert content.status_code == 200, f"{skill_id}: {content.text[:300]}"
        expected = migrated.added[skill_id]
        assert hashlib.sha256(content.content).hexdigest() == _sha(expected), skill_id
        assert _sha(skill["content"]) == _sha(expected), skill_id


def _system_prompt(upgraded: Upgraded, who: str, mention: str) -> str:
    upgraded.provider.reset()
    upgraded.provider.queue(reply.text("Noted."))
    with upgraded.client(who) as client:
        ask(client, f"{mention} what first?", model=MOCK_MODEL_ID)
    request = upgraded.provider.chat_requests()[-1]
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


@pytest.mark.parametrize(
    ("who", "skill_id"),
    [("bob", "trip-planner"), ("carol", "house-style")],
    ids=["group", "public"],
)
def test_a_reader_mentions_an_old_skill_and_the_model_gets_its_first_version(
    migrated, who, skill_id
):
    upgraded = migrated.upgraded
    saved = upgraded.manifest["skills"][skill_id]
    version_id = upgraded.get("admin", f"/api/v1/skills/id/{skill_id}").json()["version_id"]

    system = _system_prompt(upgraded, who, f"<${skill_id}|{saved['name']}>")

    assert saved["content"] in system, system
    assert f'version_id="{version_id}"' in system, system


def _save(upgraded: Upgraded, who: str, skill_id: str, changes: dict) -> dict:
    skill = upgraded.get(who, f"/api/v1/skills/id/{skill_id}").json()
    form = {
        "id": skill_id,
        "name": skill["name"],
        "description": skill["description"],
        "expected_version_id": skill["version_id"],
        **changes,
    }
    with upgraded.client(who) as client:
        saved = client.post(f"/api/v1/skills/id/{skill_id}/update", json=form)
    assert saved.status_code == 200, f"{who} cannot save {skill_id}: {saved.text[:500]}"
    return skill


def _history(upgraded: Upgraded, skill_id: str) -> list[dict]:
    return upgraded.get("admin", f"/api/v1/skills/id/{skill_id}/history").json()


def test_the_first_save_by_a_writer_adds_a_version_and_keeps_the_instructions(migrated):
    upgraded = migrated.upgraded
    first = _save(upgraded, "bob", "bulk-000", {"description": "Now with a description."})

    skill = upgraded.get("bob", "/api/v1/skills/id/bulk-000").json()
    assert skill["description"] == "Now with a description."
    assert skill["content"] == migrated.added["bulk-000"], "the first save changed the instructions"
    history = _history(upgraded, "bulk-000")
    assert [(entry["id"], entry["parent_id"]) for entry in history] == [
        (skill["version_id"], first["version_id"]),
        (first["version_id"], None),
    ]
    assert history[0]["user_id"] == upgraded.manifest["accounts"]["bob"]["id"]

    restored = upgraded.backend.client(upgraded.tokens["bob"]).post(
        f"/api/v1/skills/id/bulk-000/history/{first['version_id']}/restore",
        json={"expected_version_id": skill["version_id"]},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["description"] == first["description"]
    assert restored.json()["content"] == migrated.added["bulk-000"]


def test_a_skill_above_the_new_file_limit_can_still_be_renamed(migrated):
    upgraded = migrated.upgraded
    _save(upgraded, "bob", "large-handbook", {"name": "Large handbook, renamed"})

    skill = upgraded.get("bob", "/api/v1/skills/id/large-handbook").json()
    assert skill["name"] == "Large handbook, renamed"
    assert _sha(skill["content"]) == _sha(migrated.added["large-handbook"])
    assert len(_history(upgraded, "large-handbook")) == 2


def test_the_admin_edits_a_skill_whose_owner_was_deleted(migrated):
    upgraded = migrated.upgraded
    deleted_id = upgraded.manifest["deleted_account"]["id"]
    _save(upgraded, "admin", "orphan-copy", {"content": "Hand over at seven."})

    skill = upgraded.get("admin", "/api/v1/skills/id/orphan-copy").json()
    assert (skill["user_id"], skill["content"]) == (deleted_id, "Hand over at seven.")
    history = _history(upgraded, "orphan-copy")
    assert [entry["user_id"] for entry in history] == [
        upgraded.manifest["accounts"]["admin"]["id"],
        deleted_id,
    ]
    assert history[1]["user"] is None
