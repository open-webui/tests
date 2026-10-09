"""Skills with files, saved and read the way the skill editor and the import dialog do.

A skill holds a tree of files with `SKILL.md` at its root. The editor creates one with every
file (`create`), saves an edit as file operations against the version it opened (`save`), lists
a version's files and reads one (`files`, `read`) and restores an old version (`restore`). The
import dialog previews an upload and then imports it with one decision per skill (`preview`,
`import_skills`). `delete_skills_of` removes an account's skills after a test.
"""

from __future__ import annotations

import json
import uuid

from harness.actors import Actor


def new_id(prefix: str = "skill") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def skill_md(name: str, body: str, description: str = "How to do it.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def create(owner: Actor, files: dict[str, str], skill_id: str | None = None, **fields) -> dict:
    """Create a skill holding `files` (path to text) as the editor's Save & Create does."""
    skill_id = skill_id or new_id()
    form = {
        "id": skill_id,
        "name": fields.pop("name", f"Skill {skill_id}"),
        "description": fields.pop("description", ""),
        "files": [{"path": path, "content": content} for path, content in files.items()],
        "commit_message": fields.pop("commit_message", ""),
        "meta": {},
        "is_active": True,
        "access_grants": [],
        **fields,
    }
    with owner.client() as client:
        created = client.post("/api/v1/skills/create", json=form)
    assert created.status_code == 200, created.text
    return created.json()


def save(
    owner: Actor,
    skill: dict,
    operations: list[dict],
    expected_version_id: str | None = None,
    commit_message: str = "",
):
    """The editor's Save: file operations against the version it opened; the raw response."""
    form = {
        "id": skill["id"],
        "name": skill["name"],
        "description": skill.get("description") or "",
        "operations": operations,
        "expected_version_id": expected_version_id or skill["version_id"],
        "commit_message": commit_message,
        "is_active": True,
        "meta": skill.get("meta") or {},
    }
    with owner.client() as client:
        return client.post(f"/api/v1/skills/id/{skill['id']}/update", json=form)


def current(owner: Actor, skill_id: str) -> dict:
    with owner.client() as client:
        fetched = client.get(f"/api/v1/skills/id/{skill_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def files(owner: Actor, skill_id: str, version_id: str | None = None) -> dict[str, int]:
    """Path to size for every file of a version (the current one by default)."""
    version_id = version_id or current(owner, skill_id)["version_id"]
    with owner.client() as client:
        listed = client.get(
            f"/api/v1/skills/id/{skill_id}/files", params={"version_id": version_id}
        )
    assert listed.status_code == 200, listed.text
    return {entry["path"]: entry["size"] for entry in listed.json()["files"]}


def read(owner: Actor, skill_id: str, path: str, version_id: str | None = None) -> bytes:
    version_id = version_id or current(owner, skill_id)["version_id"]
    with owner.client() as client:
        fetched = client.get(
            f"/api/v1/skills/id/{skill_id}/files/content",
            params={"version_id": version_id, "path": path},
        )
    assert fetched.status_code == 200, fetched.text
    return fetched.content


def history(owner: Actor, skill_id: str) -> list[dict]:
    with owner.client() as client:
        listed = client.get(f"/api/v1/skills/id/{skill_id}/history")
    assert listed.status_code == 200, listed.text
    return listed.json()


def set_production(owner: Actor, skill_id: str, version_id: str, expected_version_id: str):
    with owner.client() as client:
        return client.post(
            f"/api/v1/skills/id/{skill_id}/update/version",
            json={"version_id": version_id, "expected_version_id": expected_version_id},
        )


def delete_version(owner: Actor, skill_id: str, version_id: str):
    with owner.client() as client:
        return client.delete(f"/api/v1/skills/id/{skill_id}/history/{version_id}")


def _uploads(uploads: dict[str, bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, data, "application/octet-stream")) for name, data in uploads.items()]


def preview(owner: Actor, uploads: dict[str, bytes]) -> list[dict]:
    """What the import dialog lists for the picked files (file name to bytes)."""
    with owner.client() as client:
        previewed = client.post("/api/v1/skills/import/preview", files=_uploads(uploads))
    assert previewed.status_code == 200, previewed.text
    return previewed.json()


def import_skills(owner: Actor, uploads: dict[str, bytes], decisions: list[dict]) -> list[dict]:
    """Import Selected: the same files again with the dialog's decision for each skill."""
    with owner.client() as client:
        imported = client.post(
            "/api/v1/skills/import",
            files=_uploads(uploads),
            data={"decisions": json.dumps(decisions)},
        )
    assert imported.status_code == 200, imported.text
    return imported.json()


def delete_skills_of(account: Actor) -> None:
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")
