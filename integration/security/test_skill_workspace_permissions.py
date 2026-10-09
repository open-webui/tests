"""Journey: the workspace permissions behind creating, copying, importing and exporting skills.

Accounts get each permission through a group of their own and lack it otherwise, as the default
user permissions leave all of them off. Creating a skill needs the skills or the import
permission. Copying one needs read access to it and the same permission, and the copy is the
copier's own and private. Import and its preview need the import permission; an imported skill
is private whatever the package says, and replacing an existing skill through an import needs
write access to it. Export needs the export permission and only carries skills the account may
read, old versions included. A public grant is kept only for an account with the public sharing
permission.

Discriminates: passes on dev 178de3666. In a backend copy, dropping the create permission check
turned the create and copy tests red; copying the current version whatever version is asked
turned the old version copy test red; dropping the import permission check turned the import
permission test red; dropping the write checks on an import's replace turned the replace test
red (the reader's package replaced the owner's files); dropping the export permission check
turned the export permission test red; exporting listed skills without an access check turned
the foreign export test red; looking a history entry up by its id alone turned the old version
export test red; letting every public grant through turned the three public grant rows red;
and keeping the source's grants on a copy, or a public grant on an import, turned the private
copy test red.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SKILLS = "/api/v1/skills"
SECRET = "private packing order: lamp before rope"


def _with(admin: Actor, account: Actor, **workspace: bool) -> Actor:
    make_group(admin, [account], {"workspace": workspace})
    return account


def _skill(text: str = SECRET) -> dict:
    suffix = uuid.uuid4().hex[:8]
    return {
        "id": f"packing-{suffix}",
        "name": f"Packing {suffix}",
        "description": "Pack in order.",
        "files": [
            {"path": "SKILL.md", "content": "Pack in the order the checklist gives."},
            {"path": "references/order.md", "content": text},
        ],
    }


def _create(owner: Actor, grants: list[dict] | None = None, text: str = SECRET) -> dict:
    with owner.client() as client:
        created = client.post(f"{SKILLS}/create", json=_skill(text))
        assert created.status_code == 200, created.text
        skill = created.json()
        if grants:
            shared = client.post(
                f"{SKILLS}/id/{skill['id']}/access/update", json={"access_grants": grants}
            )
            assert shared.status_code == 200, shared.text
    return skill


def _owner_view(owner: Actor, skill_id: str) -> tuple:
    with owner.client() as client:
        skill = client.get(f"{SKILLS}/id/{skill_id}").json()
        files = client.get(f"{SKILLS}/id/{skill_id}/files").json()
    return skill, files


def _package(**fields) -> bytes:
    suffix = uuid.uuid4().hex[:8]
    package = {
        "id": f"imported-{suffix}",
        "name": f"Imported {suffix}",
        "description": "Imported.",
        "files": [
            {"path": "SKILL.md", "content": "Imported instructions."},
            {"path": "references/order.md", "content": "imported order"},
        ],
        **fields,
    }
    return json.dumps(package).encode()


def _preview(account: Actor, package: bytes) -> httpx.Response:
    with account.client() as client:
        return client.post(
            f"{SKILLS}/import/preview",
            files=[("files", ("skill.json", package, "application/json"))],
        )


def _import(account: Actor, package: bytes, decision: dict) -> httpx.Response:
    with account.client() as client:
        return client.post(
            f"{SKILLS}/import",
            files=[("files", ("skill.json", package, "application/json"))],
            data={"decisions": json.dumps([decision])},
        )


def _export(account: Actor, **params) -> httpx.Response:
    with account.client() as client:
        return client.get(f"{SKILLS}/export", params=params)


def test_creating_needs_the_skills_or_the_import_permission(admin, make_user):
    without = make_user()
    editor = _with(admin, make_user(), skills=True)
    importer = _with(admin, make_user(), skills_import=True)

    answered = {}
    for name, account in {"without": without, "editor": editor, "importer": importer}.items():
        with account.client() as client:
            answered[name] = client.post(f"{SKILLS}/create", json=_skill()).status_code

    assert answered == {"without": 401, "editor": 200, "importer": 200}


def test_a_copy_needs_read_access_and_the_skills_permission(admin, make_user):
    owner = _with(admin, make_user(), skills=True)
    reader = _with(admin, make_user(), skills=True)
    reader_without = make_user()
    stranger = _with(admin, make_user(), skills=True)
    source = _create(
        owner,
        [grant("user", reader.id, "read"), grant("user", reader_without.id, "read")],
    )

    answered = {}
    for name, account in {
        "reader": reader,
        "without": reader_without,
        "stranger": stranger,
    }.items():
        with account.client() as client:
            suffix = uuid.uuid4().hex[:8]
            copied = client.post(
                f"{SKILLS}/id/{source['id']}/clone",
                json={"id": f"copy-{suffix}", "name": f"Copy {suffix}"},
            )
        answered[name] = copied
    statuses = {name: answer.status_code for name, answer in answered.items()}

    assert statuses == {"reader": 200, "without": 401, "stranger": 403}
    assert SECRET not in answered["stranger"].text + answered["without"].text
    copy = answered["reader"].json()
    assert copy["user_id"] == reader.id and copy["access_grants"] == [], copy
    with reader.client() as client:
        text = client.get(
            f"{SKILLS}/id/{copy['id']}/files/content",
            params={"version_id": copy["version_id"], "path": "references/order.md"},
        ).text
    assert text == SECRET


def test_a_copy_of_an_old_version_carries_the_old_files(admin, make_user):
    owner = _with(admin, make_user(), skills=True)
    reader = _with(admin, make_user(), skills=True)
    source = _create(owner, [grant("user", reader.id, "read")], text="old order")
    with owner.client() as client:
        client.post(
            f"{SKILLS}/id/{source['id']}/update",
            json={
                "id": source["id"],
                "name": source["name"],
                "expected_version_id": source["version_id"],
                "operations": [{"op": "put", "path": "references/order.md", "content": "new"}],
            },
        ).raise_for_status()

    suffix = uuid.uuid4().hex[:8]
    with reader.client() as client:
        copy = client.post(
            f"{SKILLS}/id/{source['id']}/clone",
            json={
                "id": f"copy-{suffix}",
                "name": f"Copy {suffix}",
                "version_id": source["version_id"],
            },
        ).json()
        text = client.get(
            f"{SKILLS}/id/{copy['id']}/files/content",
            params={"version_id": copy["version_id"], "path": "references/order.md"},
        ).text

    assert text == "old order"


def test_import_needs_the_import_permission(admin, make_user):
    without = _with(admin, make_user(), skills=True)
    importer = _with(admin, make_user(), skills_import=True)

    refused_preview = _preview(without, _package())
    refused_import = _import(without, _package(), {"action": "create"})
    preview = _preview(importer, _package())
    package = _package(access_grants=[grant("user", "*", "write")])
    imported = _import(importer, package, {"action": "create"})

    assert (refused_preview.status_code, refused_import.status_code) == (403, 403)
    assert preview.status_code == 200, preview.text
    assert imported.status_code == 200 and imported.json()[0]["status"] == "saved", imported.text
    skill_id = json.loads(package)["id"]
    with importer.client() as client:
        stored = client.get(f"{SKILLS}/id/{skill_id}").json()
    assert stored["user_id"] == importer.id and stored["access_grants"] == [], stored
    with without.client() as client:
        assert client.get(f"{SKILLS}/id/{skill_id}").status_code == 401


def test_replacing_through_an_import_needs_write_access(admin, make_user):
    owner = _with(admin, make_user(), skills=True)
    reader = _with(admin, make_user(), skills_import=True)
    writer = _with(admin, make_user(), skills_import=True)
    stranger = _with(admin, make_user(), skills_import=True)
    target = _create(
        owner,
        [
            grant("user", reader.id, "read"),
            grant("user", writer.id, "read"),
            grant("user", writer.id, "write"),
        ],
    )
    package = _package(id=target["id"], name=target["name"])

    previews = {
        name: _preview(account, package).json()[0]
        for name, account in {"reader": reader, "writer": writer, "stranger": stranger}.items()
    }
    replace = {"action": "replace", "expected_version_id": target["version_id"]}
    before = _owner_view(owner, target["id"])
    refused = [_import(account, package, replace).json()[0] for account in (reader, stranger)]
    after_refused = _owner_view(owner, target["id"])
    replaced = _import(writer, package, replace).json()[0]

    assert {
        name: (entry["can_replace"], entry["expected_version_id"])
        for name, entry in previews.items()
    } == {
        "reader": (False, None),
        "writer": (True, target["version_id"]),
        "stranger": (False, None),
    }
    assert [entry["status"] for entry in refused] == ["error", "error"], refused
    assert SECRET not in json.dumps(refused)
    assert after_refused == before, "a refused import replaced the owner's skill"
    assert replaced["status"] == "saved", replaced
    skill, files = _owner_view(owner, target["id"])
    assert skill["version_id"] != target["version_id"]
    assert len(skill["access_grants"]) == 3, "a replacing import changed the skill's sharing"
    with owner.client() as client:
        text = client.get(
            f"{SKILLS}/id/{target['id']}/files/content",
            params={"version_id": skill["version_id"], "path": "references/order.md"},
        ).text
    assert text == "imported order"


def test_export_needs_the_export_permission(admin, make_user):
    owner = _with(admin, make_user(), skills=True)
    without = make_user()
    exporter = _with(admin, make_user(), skills_export=True)
    skill = _create(owner, [grant("user", without.id, "read"), grant("user", exporter.id, "read")])

    refused = _export(without, ids=[skill["id"]])
    exported = _export(exporter, ids=[skill["id"]])
    archive = _export(exporter, ids=[skill["id"]], format="zip")

    assert refused.status_code == 403 and SECRET not in refused.text
    assert exported.status_code == 200, exported.text
    assert {f["path"]: f["content"] for f in exported.json()["files"]}[
        "references/order.md"
    ] == SECRET
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        names = bundle.namelist()
    assert sorted(name.split("/", 1)[1] for name in names) == ["SKILL.md", "references/order.md"]


def test_an_export_carries_only_skills_the_account_may_read(admin, make_user):
    owner = _with(admin, make_user(), skills=True)
    exporter = _with(admin, make_user(), skills_export=True)
    shared = _create(owner, [grant("user", exporter.id, "read")], text="shared order")
    private = _create(owner)

    named = _export(exporter, ids=[private["id"]])
    named_with_shared = _export(exporter, ids=[shared["id"], private["id"]])
    everything = _export(exporter)
    by_admin = _export(admin, ids=[private["id"]])

    assert (named.status_code, named_with_shared.status_code) == (403, 403)
    assert SECRET not in named.text + named_with_shared.text
    exported_ids = {package["id"] for package in everything.json()}
    assert shared["id"] in exported_ids and private["id"] not in exported_ids
    assert SECRET not in everything.text
    assert by_admin.status_code == 200 and SECRET in by_admin.text


def test_an_old_version_exports_only_from_its_own_skill(admin, make_user):
    owner = _with(admin, make_user(), skills=True)
    exporter = _with(admin, make_user(), skills_export=True)
    shared = _create(owner, [grant("user", exporter.id, "read")], text="shared order")
    private = _create(owner)

    old = _export(exporter, ids=[shared["id"]], version_id=shared["version_id"])
    foreign = _export(exporter, ids=[shared["id"]], version_id=private["version_id"])

    assert old.status_code == 200 and "shared order" in old.text
    assert foreign.status_code == 404 and SECRET not in foreign.text


@pytest.mark.parametrize("route", ["create", "update", "access"])
def test_a_public_grant_needs_the_public_sharing_permission(route, admin, make_user):
    plain = _with(admin, make_user(), skills=True)
    sharer = make_user()
    make_group(admin, [sharer], {"workspace": {"skills": True}, "sharing": {"public_skills": True}})
    public = [grant("user", "*", "read")]

    kept = {}
    for name, account in {"plain": plain, "sharer": sharer}.items():
        body = _skill()
        with account.client() as client:
            if route == "create":
                answer = client.post(f"{SKILLS}/create", json={**body, "access_grants": public})
            else:
                created = client.post(f"{SKILLS}/create", json=body).json()
                if route == "update":
                    answer = client.post(
                        f"{SKILLS}/id/{body['id']}/update",
                        json={"id": body["id"], "name": body["name"], "access_grants": public},
                    )
                else:
                    answer = client.post(
                        f"{SKILLS}/id/{created['id']}/access/update",
                        json={"access_grants": public},
                    )
            assert answer.status_code == 200, answer.text
            stored = client.get(f"{SKILLS}/id/{body['id']}").json()
        kept[name] = [
            (entry["principal_id"], entry["permission"]) for entry in stored["access_grants"]
        ]

    assert kept == {"plain": [], "sharer": [("*", "read")]}


def test_a_copy_and_an_import_of_a_public_skill_start_private(admin, make_user):
    owner = make_user()
    make_group(admin, [owner], {"workspace": {"skills": True}, "sharing": {"public_skills": True}})
    copier = make_user()
    make_group(
        admin,
        [copier],
        {"workspace": {"skills": True, "skills_import": True}, "sharing": {"public_skills": True}},
    )
    source = _create(owner, [grant("user", "*", "read")])

    suffix = uuid.uuid4().hex[:8]
    with copier.client() as client:
        copy = client.post(
            f"{SKILLS}/id/{source['id']}/clone",
            json={"id": f"copy-{suffix}", "name": f"Copy {suffix}"},
        ).json()
    package = _package(access_grants=[grant("user", "*", "read")])
    _import(copier, package, {"action": "create"}).raise_for_status()
    with copier.client() as client:
        imported = client.get(f"{SKILLS}/id/{json.loads(package)['id']}").json()

    assert copy["access_grants"] == [] and imported["access_grants"] == []
