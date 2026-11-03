"""Comment threads on the items people can open, driven over `/api/v1/comments`.

`switch_comments(admin, on)` sets the admin's Comments switch (preserve `admin_config` first).
`COMMENTABLE` says, per kind of item, how an owner creates one and shares it with a read grant,
the way the workspace and the share dialog do; `shared_item(kind, owner, admin, readers)` makes
one. `comment`, `reply`, `edit`, `delete`, `react` and `unreact` send what the comment panel
sends and return the response; `thread` and `replies` read back what the panel shows.
`mention(actor)` is the token the mention picker writes into a comment, and `mentions` reads
whom the picker offers.
"""

from __future__ import annotations

import uuid

import httpx

from harness.access import Shareable, grant, make_group
from harness.actors import Actor
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

COMMENTS = "/api/v1/comments"
ADMIN_CONFIG = "/api/v1/auths/admin/config"


def switch_comments(admin: Actor, on: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        assert current.status_code == 200, current.text
        assert "ENABLE_COMMENTS" in current.json(), "the admin settings have no Comments switch"
        saved = client.post(ADMIN_CONFIG, json={**current.json(), "ENABLE_COMMENTS": on})
    assert saved.status_code == 200, saved.text
    assert saved.json()["ENABLE_COMMENTS"] is on, f"the Comments switch did not take: {saved.text}"


def _suffix() -> str:
    return uuid.uuid4().hex[:8]


def _chat_shared_to_reply(client: httpx.Client, grants: list[dict]) -> str:
    """A chat shared with "Allow replies", the share mode that lets its grantees open it."""
    chat_id, _ = seed_chat(
        client,
        [
            {"role": "user", "content": "the owner's question"},
            {"role": "assistant", "content": "the owner's answer"},
        ],
    )
    shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
    assert shared.status_code == 200, shared.text
    saved = client.post(
        f"/api/v1/chats/shared/{chat_id}/access/update",
        json={"access_grants": grants, "share_mode": "continue"},
    )
    assert saved.status_code == 200, saved.text
    return chat_id


TOOL_SOURCE = '''class Tools:
    def echo(self, text: str) -> str:
        """Say the text back.

        :param text: what to say
        """
        return text
'''

COMMENTABLE = {
    "chat": Shareable(create_shared=_chat_shared_to_reply),
    "note": Shareable(
        create_path="/api/v1/notes/create",
        create_body=lambda: {"title": f"commented {_suffix()}", "data": {"content": {}}},
        access_path="/api/v1/notes/{id}/access/update",
    ),
    "model": Shareable(
        create_path="/api/v1/models/create",
        create_body=lambda: {
            "id": f"commented-{_suffix()}",
            "base_model_id": MOCK_MODEL_ID,
            "name": "Commented preset",
            "meta": {"description": "comment threads"},
            "params": {},
        },
        access_path="/api/v1/models/model/access/update",
        access_fields=lambda model_id: {"id": model_id},
        owner_permissions={"workspace": {"models": True}},
    ),
    "knowledge": Shareable(
        create_path="/api/v1/knowledge/create",
        create_body=lambda: {"name": f"commented {_suffix()}", "description": ""},
        access_path="/api/v1/knowledge/{id}/access/update",
        owner_permissions={"workspace": {"knowledge": True}},
    ),
    "prompt": Shareable(
        create_path="/api/v1/prompts/create",
        create_body=lambda: {
            "command": f"commented-{_suffix()}",
            "name": "Commented prompt",
            "content": "Summarise the thread.",
        },
        access_path="/api/v1/prompts/id/{id}/access/update",
        owner_permissions={"workspace": {"prompts": True}},
    ),
    "skill": Shareable(
        create_path="/api/v1/skills/create",
        create_body=lambda: {
            "id": f"skill-{_suffix()}",
            "name": f"Commented skill {_suffix()}",
            "content": "Write release notes.",
        },
        access_path="/api/v1/skills/id/{id}/access/update",
        owner_permissions={"workspace": {"skills": True}},
    ),
    "tool": Shareable(
        create_path="/api/v1/tools/create",
        create_body=lambda: {
            "id": f"tool_{_suffix()}",
            "name": "Echo",
            "content": TOOL_SOURCE,
            "meta": {"description": "comment threads"},
        },
        access_path="/api/v1/tools/id/{id}/access/update",
        owner_permissions={"workspace": {"tools": True}},
    ),
}


def shared_item(kind: str, owner: Actor, admin: Actor, readers: list[Actor]) -> str:
    """A new item of `kind`, the owner's, shared for reading with each of `readers`."""
    shareable = COMMENTABLE[kind]
    if shareable.owner_permissions:
        make_group(admin, [owner], shareable.owner_permissions)
    grants = [grant("user", reader.id, "read") for reader in readers]
    with owner.client() as client:
        if shareable.create_shared:
            return shareable.create_shared(client, grants)
        created = client.post(shareable.create_path, json=shareable.create_body())
        assert created.status_code == 200, f"creating the {kind} failed: {created.text}"
        item_id = created.json()["id"]
        shared = client.post(
            shareable.access_path.format(id=item_id),
            json={**shareable.access_fields(item_id), "access_grants": grants},
        )
    assert shared.status_code == 200, f"sharing the {kind} failed: {shared.text}"
    return item_id


def mention(actor: Actor) -> str:
    return f"<@U:{actor.id}|{actor.name}>"


def comment(actor: Actor, kind: str, item_id: str, content: str) -> httpx.Response:
    with actor.client() as client:
        return client.post(
            f"{COMMENTS}/",
            json={"resource_type": kind, "resource_id": item_id, "content": content},
        )


def reply(
    actor: Actor, comment_id: str, content: str, reply_to: str | None = None
) -> httpx.Response:
    body = {"content": content}
    if reply_to:
        body["reply_to_id"] = reply_to
    with actor.client() as client:
        return client.post(f"{COMMENTS}/{comment_id}/replies", json=body)


def edit(actor: Actor, posted: dict, content: str) -> httpx.Response:
    """Edits the comment as last read (`posted` carries its id and etag)."""
    with actor.client() as client:
        return client.patch(
            f"{COMMENTS}/{posted['id']}", json={"content": content, "etag": posted["etag"]}
        )


def delete(actor: Actor, posted: dict) -> httpx.Response:
    with actor.client() as client:
        return client.delete(f"{COMMENTS}/{posted['id']}", params={"etag": posted["etag"]})


def react(actor: Actor, comment_id: str, name: str) -> httpx.Response:
    with actor.client() as client:
        return client.put(f"{COMMENTS}/{comment_id}/reactions", json={"name": name})


def unreact(actor: Actor, comment_id: str, name: str) -> httpx.Response:
    with actor.client() as client:
        return client.delete(f"{COMMENTS}/{comment_id}/reactions", params={"name": name})


def thread(actor: Actor, kind: str, item_id: str) -> httpx.Response:
    """The item's starting comments, as the comment panel lists them."""
    with actor.client() as client:
        return client.get(f"{COMMENTS}/", params={"resource_type": kind, "resource_id": item_id})


def replies(actor: Actor, comment_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"{COMMENTS}/{comment_id}/replies")


def read(actor: Actor, comment_id: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(f"{COMMENTS}/{comment_id}")


def mentions(actor: Actor, kind: str, item_id: str, query: str) -> httpx.Response:
    """The people the mention picker offers for `query` on this item."""
    with actor.client() as client:
        return client.get(
            f"{COMMENTS}/mentions",
            params={"resource_type": kind, "resource_id": item_id, "query": query},
        )
