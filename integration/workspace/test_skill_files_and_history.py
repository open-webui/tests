"""Journey: a skill's files, its saved versions, going back to one, and moving skills in and out.

A skill is a tree of files with `SKILL.md` at its root. The editor creates it with every file and
saves each edit as file operations against the version it opened: a put, a move and a delete in
one save leave the untouched files as they were and add one version carrying the commit message.
A save made from a version that is no longer current is refused as a conflict and changes nothing.
Binary files keep their bytes. The history lists every version; comparing an old one to the
current one names the added, deleted and changed files and gives a text diff per file; setting
an old version as production brings its files back without adding a version, and an old version
can be deleted while the current one cannot. The Skills page imports a ZIP
holding skill folders, a folder picked from disk, a single SKILL.md, or a JSON export, with a
preview that names each skill from its front matter and offers to replace one that exists. A
skill's own ZIP export holds every file under the skill's name, Clone copies the files into a new
private skill, and deleting a skill takes its history with it. A chat that mentions a skill gets
its SKILL.md with the list of its other files.

Who may read or change a shared skill's files and history, and the upgrade of skills saved
before files existed, are covered elsewhere.

Discriminates: passes on dev 178de3666. Each of these edits to a backend copy turns its test red: a
file move that keeps the old path (editor save), every save adding a version (save with nothing
changed), no version check on save (conflict), base64 files counted as text (binary), every file
compared as modified (comparison), a version switch that keeps the current files (set production), a
ZIP import that keeps only SKILL.md (ZIP import), a picked folder that keeps only SKILL.md (folder
import), an import that ignores the front matter (Markdown), a preview that never offers Replace
(replace), a copy import that keeps the uploaded id (copy), a ZIP export that writes only SKILL.md
(ZIP export), an old version exported as the current one (old version export), a clone that copies
an empty SKILL.md alone (clone), a delete that keeps the history (delete) and a mentioned skill sent
without its file list (chat mention). Retargeted for 24ee1cb16, where an old version is set as
production in place of being restored as a copy and old versions can be deleted: the set production
and version delete tests pass on that build, and go red in a backend copy whose version switch keeps
the current files or whose version delete answers true without deleting.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.skill_files import (
    create,
    current,
    delete_skills_of,
    delete_version,
    files,
    history,
    import_skills,
    new_id,
    preview,
    read,
    save,
    set_production,
    skill_md,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    delete_skills_of(account)


# the smallest valid PNG, with a zero byte that keeps it from reading as text
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _tide_skill(owner, **fields) -> dict:
    skill_id = new_id("tides")
    return create(
        owner,
        {
            "SKILL.md": skill_md(skill_id, "Read the tide table first."),
            "references/tables.md": "High water at noon.\n",
            "references/charts.md": "Charts live here.\n",
            "scripts/check.py": "print('tide')\n",
        },
        skill_id=skill_id,
        **fields,
    )


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, data in entries.items():
            archive.writestr(path, data)
    return buffer.getvalue()


def test_an_editor_save_applies_put_move_and_delete_and_keeps_the_rest(owner):
    skill = _tide_skill(owner)

    saved = save(
        owner,
        skill,
        [
            {"op": "put", "path": "references/tables.md", "content": "High water at one.\n"},
            {"op": "put", "path": "notes/new.md", "content": "Fresh notes.\n"},
            {"op": "move", "path": "scripts", "destination": "tools"},
            {"op": "delete", "path": "references/charts.md"},
        ],
        commit_message="Move scripts to tools",
    )

    assert saved.status_code == 200, saved.text
    assert saved.json()["version_id"] != skill["version_id"]
    assert set(files(owner, skill["id"])) == {
        "SKILL.md",
        "notes/new.md",
        "references/tables.md",
        "tools/check.py",
    }
    assert read(owner, skill["id"], "references/tables.md") == b"High water at one.\n"
    assert read(owner, skill["id"], "tools/check.py") == b"print('tide')\n"
    assert read(owner, skill["id"], "SKILL.md").decode() == skill_md(
        skill["id"], "Read the tide table first."
    )
    versions = history(owner, skill["id"])
    assert len(versions) == 2
    assert "Move scripts to tools" in {entry["commit_message"] for entry in versions}
    # the version the editor opened still holds the old tree
    assert "scripts/check.py" in files(owner, skill["id"], skill["version_id"])


def test_a_save_with_nothing_changed_adds_no_version(owner):
    skill = _tide_skill(owner)

    saved = save(owner, skill, [])

    assert saved.status_code == 200, saved.text
    assert saved.json()["version_id"] == skill["version_id"]
    assert len(history(owner, skill["id"])) == 1


def test_a_save_from_an_outdated_version_is_refused_and_changes_nothing(owner):
    skill = _tide_skill(owner)
    first = save(owner, skill, [{"op": "put", "path": "a.md", "content": "first tab\n"}])
    assert first.status_code == 200, first.text

    # a second tab still holds the version both tabs opened
    second = save(owner, skill, [{"op": "put", "path": "a.md", "content": "second tab\n"}])

    assert second.status_code == 409, second.text
    assert second.json()["detail"]["code"] == "version_conflict"
    assert read(owner, skill["id"], "a.md") == b"first tab\n"
    assert len(history(owner, skill["id"])) == 2


def test_an_uploaded_binary_file_keeps_its_bytes(owner):
    skill = _tide_skill(owner)
    encoded = base64.b64encode(PNG).decode()

    saved = save(
        owner,
        skill,
        [{"op": "put", "path": "assets/flag.png", "content": encoded, "encoding": "base64"}],
    )

    assert saved.status_code == 200, saved.text
    assert files(owner, skill["id"])["assets/flag.png"] == len(PNG)
    assert read(owner, skill["id"], "assets/flag.png") == PNG


def test_comparing_an_old_version_names_each_changed_file_and_diffs_it(owner):
    skill = _tide_skill(owner)
    old_version = skill["version_id"]
    saved = save(
        owner,
        {**skill, "description": "Tides, revised"},
        [
            {"op": "put", "path": "references/tables.md", "content": "High water at one.\n"},
            {"op": "put", "path": "notes/new.md", "content": "Fresh notes.\n"},
            {"op": "delete", "path": "references/charts.md"},
        ],
    ).json()

    with owner.client() as client:
        compared = client.get(
            f"/api/v1/skills/id/{skill['id']}/history/diff",
            params={"from_id": old_version, "to_id": saved["version_id"]},
        ).json()
        file_diff = client.get(
            f"/api/v1/skills/id/{skill['id']}/history/diff/file",
            params={
                "from_id": old_version,
                "to_id": saved["version_id"],
                "path": "references/tables.md",
            },
        ).json()

    assert {entry["path"]: entry["status"] for entry in compared["files"]} == {
        "notes/new.md": "added",
        "references/charts.md": "deleted",
        "references/tables.md": "modified",
    }
    assert compared["metadata"]["description"]["after"] == "Tides, revised"
    assert file_diff["binary"] is False
    assert "-High water at noon." in file_diff["diff"]
    assert "+High water at one." in file_diff["diff"]


def test_setting_an_old_version_as_production_brings_back_its_files(owner):
    skill = _tide_skill(owner)
    edited = save(
        owner,
        skill,
        [
            {"op": "put", "path": "SKILL.md", "content": skill_md(skill["id"], "Ignore tides.")},
            {"op": "delete", "path": "references"},
        ],
    ).json()

    switched = set_production(owner, skill["id"], skill["version_id"], edited["version_id"])

    assert switched.status_code == 200, switched.text
    assert switched.json()["version_id"] == skill["version_id"]
    assert set(files(owner, skill["id"])) == {
        "SKILL.md",
        "references/charts.md",
        "references/tables.md",
        "scripts/check.py",
    }
    assert b"Read the tide table first." in read(owner, skill["id"], "SKILL.md")
    assert {entry["id"] for entry in history(owner, skill["id"])} == {
        skill["version_id"],
        edited["version_id"],
    }


def test_an_old_version_can_be_deleted_but_not_the_current_one(owner):
    skill = _tide_skill(owner)
    edited = save(
        owner,
        skill,
        [{"op": "put", "path": "SKILL.md", "content": skill_md(skill["id"], "Ignore tides.")}],
    ).json()

    refused = delete_version(owner, skill["id"], edited["version_id"])
    deleted = delete_version(owner, skill["id"], skill["version_id"])

    assert refused.status_code == 400, refused.text
    assert deleted.status_code == 200, deleted.text
    assert [entry["id"] for entry in history(owner, skill["id"])] == [edited["version_id"]]
    assert b"Ignore tides." in read(owner, skill["id"], "SKILL.md")


def test_a_zip_of_skill_folders_imports_each_skill_with_all_its_files(owner):
    first, second = new_id("knots"), new_id("sails")
    archive = _zip(
        {
            f"bundle/{first}/SKILL.md": skill_md(first, "Tie a bowline.").encode(),
            f"bundle/{first}/references/bowline.md": b"Rabbit out of the hole.\n",
            f"bundle/{second}/SKILL.md": skill_md(second, "Reef early.").encode(),
            f"bundle/{second}/assets/flag.png": PNG,
        }
    )

    previewed = preview(owner, {"skills.zip": archive})
    assert [item["id"] for item in previewed] == [first, second]
    assert {entry["path"] for entry in previewed[0]["files"]} == {
        "SKILL.md",
        "references/bowline.md",
    }
    results = import_skills(
        owner,
        {"skills.zip": archive},
        [{"action": "create", "id": item["id"], "name": item["name"]} for item in previewed],
    )

    assert [result["status"] for result in results] == ["saved", "saved"], results
    assert read(owner, first, "references/bowline.md") == b"Rabbit out of the hole.\n"
    assert read(owner, second, "assets/flag.png") == PNG
    assert current(owner, first)["access_grants"] == []


def test_a_folder_picked_from_disk_imports_as_a_skill(owner):
    skill_id = new_id("anchor")
    # the browser names each picked file by its path inside the chosen folder
    uploads = {
        f"my-skills/{skill_id}/SKILL.md": skill_md(skill_id, "Drop the hook.").encode(),
        f"my-skills/{skill_id}/references/depth.md": b"Three times the depth.\n",
    }

    previewed = preview(owner, uploads)
    results = import_skills(
        owner,
        uploads,
        [{"action": "create", "id": previewed[0]["id"], "name": previewed[0]["name"]}],
    )

    assert previewed[0]["id"] == skill_id
    assert results == [{"status": "saved", "id": skill_id}]
    assert set(files(owner, skill_id)) == {"SKILL.md", "references/depth.md"}


def test_a_markdown_file_is_named_from_its_front_matter(owner):
    skill_id = new_id("moor")
    markdown = skill_md(skill_id, "Double the lines.", description="Mooring a boat.")

    previewed = preview(owner, {"SKILL.md": markdown.encode()})

    assert len(previewed) == 1
    assert previewed[0]["id"] == skill_id
    assert previewed[0]["name"] == skill_id
    assert previewed[0]["description"] == "Mooring a boat."
    assert previewed[0]["id_taken"] is False


def test_an_import_of_an_existing_skill_offers_replace_and_saves_a_new_version(owner):
    skill = _tide_skill(owner)
    with owner.client() as client:
        exported = client.get("/api/v1/skills/export", params={"ids": [skill["id"]]}).json()
    exported["files"] = [
        {**entry, "content": "High water at two.\n"}
        if entry["path"] == "references/tables.md"
        else entry
        for entry in exported["files"]
    ]
    upload = {"tides.json": json.dumps(exported).encode()}

    (item,) = preview(owner, upload)
    assert item["id_taken"] is True and item["can_replace"] is True
    assert item["expected_version_id"] == skill["version_id"]
    results = import_skills(
        owner,
        upload,
        [
            {
                "action": "replace",
                "id": item["id"],
                "name": item["name"],
                "expected_version_id": item["expected_version_id"],
            }
        ],
    )

    assert results == [{"status": "saved", "id": skill["id"]}]
    assert read(owner, skill["id"], "references/tables.md") == b"High water at two.\n"
    assert len(history(owner, skill["id"])) == 2


def test_an_import_as_a_copy_under_a_new_id_and_name_leaves_the_original(owner):
    skill = _tide_skill(owner)
    with owner.client() as client:
        exported = client.get("/api/v1/skills/export", params={"ids": [skill["id"]]}).json()
    upload = {"tides.json": json.dumps(exported).encode()}
    copy_id = new_id("tides-copy")

    results = import_skills(
        owner, upload, [{"action": "copy", "id": copy_id, "name": f"Copy {copy_id}"}]
    )

    assert results == [{"status": "saved", "id": copy_id}]
    assert set(files(owner, copy_id)) == set(files(owner, skill["id"]))
    assert len(history(owner, skill["id"])) == 1


def test_a_skills_zip_export_holds_every_file_under_its_name(owner):
    skill = _tide_skill(owner)

    with owner.client() as client:
        exported = client.get(
            "/api/v1/skills/export", params={"format": "zip", "ids": [skill["id"]]}
        )

    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        names = set(archive.namelist())
        tables = archive.read(f"{skill['id']}/references/tables.md")
    assert names == {
        f"{skill['id']}/SKILL.md",
        f"{skill['id']}/references/charts.md",
        f"{skill['id']}/references/tables.md",
        f"{skill['id']}/scripts/check.py",
    }
    assert tables == b"High water at noon.\n"


def test_an_old_version_exports_with_its_own_files(owner):
    skill = _tide_skill(owner)
    save(owner, skill, [{"op": "delete", "path": "scripts"}])

    with owner.client() as client:
        exported = client.get(
            "/api/v1/skills/export",
            params={"ids": [skill["id"]], "version_id": skill["version_id"]},
        ).json()

    assert "scripts/check.py" in {entry["path"] for entry in exported["files"]}


def test_clone_copies_every_file_into_a_new_private_skill(owner):
    skill = _tide_skill(owner)
    clone_id = new_id("tides-clone")

    with owner.client() as client:
        cloned = client.post(
            f"/api/v1/skills/id/{skill['id']}/clone",
            json={"id": clone_id, "name": f"Clone {clone_id}"},
        )

    assert cloned.status_code == 200, cloned.text
    assert set(files(owner, clone_id)) == set(files(owner, skill["id"]))
    assert read(owner, clone_id, "scripts/check.py") == b"print('tide')\n"
    assert current(owner, clone_id)["access_grants"] == []


def test_deleting_a_skill_takes_its_history_with_it(owner):
    skill = _tide_skill(owner)
    save(owner, skill, [{"op": "put", "path": "a.md", "content": "a\n"}])

    with owner.client() as client:
        client.delete(f"/api/v1/skills/id/{skill['id']}/delete").raise_for_status()
    # a skill made again under the same id starts a history of its own
    remade = create(owner, {"SKILL.md": skill_md(skill["id"], "Start over.")}, skill_id=skill["id"])

    assert [entry["id"] for entry in history(owner, remade["id"])] == [remade["version_id"]]


def test_a_mentioned_skill_reaches_the_model_with_the_list_of_its_files(owner, upstream):
    skill = _tide_skill(owner)
    upstream.queue(reply.text("done"))

    with owner.client() as client:
        ask(client, f"<${skill['id']}|Tides> when is high water?")

    request = upstream.chat_requests()[-1]
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    assert f'<skill id="{skill["id"]}" version_id="{skill["version_id"]}"' in system, system
    assert "Read the tide table first." in system
    for path in ("references/tables.md", "references/charts.md", "scripts/check.py"):
        assert path in system, system
    # the supporting files are listed, not pasted in
    assert "High water at noon." not in system
