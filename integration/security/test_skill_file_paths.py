"""Journey: the file names a skill accepts, and how its files come back.

Skill files live in the database under paths relative to the skill, so a name may hold spaces,
accents, other scripts, emoji and the characters `# ? % & + = '`, and each such file comes back
byte for byte under its own name. A path that is absolute, uses a backslash, a colon or a NUL,
or has an empty, `.` or `..` part is refused with a 400, whether it is a new file, a move
destination or an entry in an imported archive or folder, and the skill is left as it was. A
download is always an attachment named after the file, sent with `nosniff` and a sandboxing
content security policy, so an uploaded HTML page never runs in the app's origin. An exported
archive keeps every file under the skill's own folder.

Discriminates: passes on dev 178de3666. In a backend copy, letting the path check pass every
path turned every refused name row and the dotdot, absolute and folder import rows red (200 and
a stored path); letting it pass `..` parts turned the `..` rows and the whole list test red;
letting archives hold links turned the symlink import row red; sending the raw file name in the
download header turned the typed names test red; dropping the sandbox header from downloads
turned the HTML download test red; and writing archive entries without the skill folder turned
the export archive test red.
"""

from __future__ import annotations

import io
import json
import stat
import uuid
import zipfile
from urllib.parse import quote

import httpx
import pytest

from harness.access import make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SKILLS = "/api/v1/skills"
TYPED_NAMES = [
    "notes with spaces.md",
    "références/été.md",
    "日本語/ファイル.txt",
    "emoji 🚢.md",
    "hash#tag.md",
    "question?.md",
    "100% done.md",
    "a&b=c+d.md",
    "it's here.md",
    ".hidden",
    "docs/SKILL.md",
]
REFUSED_PATHS = [
    "../escape.md",
    "references/../../escape.md",
    "/etc/passwd",
    "references\\windows.md",
    "C:/drive.md",
    "notes: draft.md",
    "references//double.md",
    "./here.md",
    "references/.",
    "folder/",
    "",
    "nul\x00byte.md",
]


@pytest.fixture
def author(admin, make_user) -> Actor:
    account = make_user()
    make_group(
        admin,
        [account],
        {"workspace": {"skills": True, "skills_import": True, "skills_export": True}},
    )
    return account


def _create(author: Actor) -> dict:
    suffix = uuid.uuid4().hex[:8]
    with author.client() as client:
        created = client.post(
            f"{SKILLS}/create",
            json={
                "id": f"paths-{suffix}",
                "name": f"Paths {suffix}",
                "files": [
                    {"path": "SKILL.md", "content": "Name files freely."},
                    {"path": "references/kept.md", "content": "kept"},
                ],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()


def _edit(client: httpx.Client, skill: dict, *operations: dict) -> httpx.Response:
    return client.post(
        f"{SKILLS}/id/{skill['id']}/update",
        json={
            "id": skill["id"],
            "name": skill["name"],
            "expected_version_id": skill["version_id"],
            "operations": list(operations),
        },
    )


def _state(client: httpx.Client, skill_id: str) -> tuple:
    current = client.get(f"{SKILLS}/id/{skill_id}").json()
    return current["version_id"], client.get(f"{SKILLS}/id/{skill_id}/files").json()


def test_names_a_person_types_are_kept_and_come_back_whole(author):
    skill = _create(author)
    puts = [{"op": "put", "path": name, "content": f"text of {name}"} for name in TYPED_NAMES]

    with author.client() as client:
        saved = _edit(client, skill, *puts)
        assert saved.status_code == 200, saved.text
        version_id = saved.json()["version_id"]
        listed = client.get(f"{SKILLS}/id/{skill['id']}/files").json()["files"]
        downloads = {
            name: client.get(
                f"{SKILLS}/id/{skill['id']}/files/content",
                params={"version_id": version_id, "path": name},
            )
            for name in TYPED_NAMES
        }

    assert set(TYPED_NAMES) <= {entry["path"] for entry in listed}
    for name, answer in downloads.items():
        assert answer.status_code == 200, f"{name}: {answer.text}"
        assert answer.content == f"text of {name}".encode(), name
        basename = name.rsplit("/", 1)[-1]
        assert answer.headers["content-disposition"] == (
            "attachment; filename*=UTF-8''" + quote(basename, safe="")
        ), name


@pytest.mark.parametrize("path", REFUSED_PATHS, ids=repr)
def test_an_unsafe_name_is_refused_and_changes_nothing(path, author):
    skill = _create(author)

    with author.client() as client:
        before = _state(client, skill["id"])
        put = _edit(client, skill, {"op": "put", "path": path, "content": "x"})
        moved = _edit(
            client,
            skill,
            {"op": "move", "path": "references/kept.md", "destination": path},
        )
        after = _state(client, skill["id"])

    assert (put.status_code, moved.status_code) == (400, 400), (put.text, moved.text)
    assert after == before


def test_a_whole_file_list_with_an_unsafe_name_is_refused(author):
    skill = _create(author)
    files = [{"path": "SKILL.md", "content": "x"}, {"path": "../escape.md", "content": "x"}]

    with author.client() as client:
        before = _state(client, skill["id"])
        replaced = client.post(
            f"{SKILLS}/id/{skill['id']}/update",
            json={
                "id": skill["id"],
                "name": skill["name"],
                "expected_version_id": skill["version_id"],
                "files": files,
            },
        )
        created = client.post(
            f"{SKILLS}/create",
            json={"id": f"bad-{uuid.uuid4().hex[:8]}", "name": uuid.uuid4().hex, "files": files},
        )
        after = _state(client, skill["id"])

    assert (replaced.status_code, created.status_code) == (400, 400)
    assert after == before


def _archive(entries: dict[str, bytes], link: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        if link:
            info = zipfile.ZipInfo(link)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "/etc/passwd")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "upload",
    [
        ("kit.zip", _archive({"SKILL.md": b"x", "../escape.md": b"x"})),
        ("kit.zip", _archive({"kit/SKILL.md": b"x", "/abs/SKILL.md": b"x"})),
        ("kit.zip", _archive({"SKILL.md": b"x"}, link="references/passwd")),
        ("../kit/SKILL.md", b"x"),
        ("kit/../../SKILL.md", b"x"),
    ],
    ids=["dotdot entry", "absolute entry", "symlink entry", "folder dotdot", "folder escape"],
)
def test_an_import_with_an_unsafe_path_is_refused(upload, author):
    name, data = upload

    with author.client() as client:
        preview = client.post(f"{SKILLS}/import/preview", files=[("files", (name, data))])
        imported = client.post(
            f"{SKILLS}/import",
            files=[("files", (name, data))],
            data={"decisions": json.dumps([{"action": "create"}])},
        )
        listed = client.get(f"{SKILLS}/").json()
    mine = [skill["id"] for skill in listed if skill["user_id"] == author.id]

    assert (preview.status_code, imported.status_code) == (400, 400), preview.text
    assert mine == [], mine


def test_an_html_file_downloads_sandboxed(author):
    skill = _create(author)
    page = "<script>parent.document.title='ran'</script>"

    with author.client() as client:
        saved = _edit(client, skill, {"op": "put", "path": "assets/page.html", "content": page})
        answer = client.get(
            f"{SKILLS}/id/{skill['id']}/files/content",
            params={"version_id": saved.json()["version_id"], "path": "assets/page.html"},
        )

    assert answer.status_code == 200 and answer.text == page
    assert answer.headers["content-disposition"].startswith("attachment;")
    assert answer.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in answer.headers["content-security-policy"]


def test_an_exported_archive_keeps_every_file_inside_the_skill_folder(author):
    skill = _create(author)
    with author.client() as client:
        _edit(
            client,
            skill,
            *[{"op": "put", "path": name, "content": "x"} for name in TYPED_NAMES],
        ).raise_for_status()
        exported = client.get(f"{SKILLS}/export", params={"ids": [skill["id"]], "format": "zip"})

    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        names = archive.namelist()
    assert all(name.startswith(f"{skill['id']}/") for name in names), names
    assert all(".." not in name.split("/") for name in names), names
    assert {name.split("/", 1)[1] for name in names} == {
        "SKILL.md",
        "references/kept.md",
        *TYPED_NAMES,
    }
