"""The workspace search over the API, called the way the search dialog calls it.

`GET /api/v1/search` answers `{"items", "has_more", "scopes", "kinds"}`. Each item id names its
source and the record: `chat:<chat id>` for a live chat, `shared:<share id>` for a shared
snapshot, `note:<note id>` and `message:<message id>`. `import_chat` stores a chat with one
question and answer the way a chat import does, so its text is known to the test.
"""

from __future__ import annotations

import uuid

import httpx

from harness.actors import Actor

SEARCH_PATH = "/api/v1/search"
PAGE_SIZE = 24


def unique_word() -> str:
    return f"numbat{uuid.uuid4().hex[:10]}"


def search(actor: Actor, q: str = "", **filters: str | int) -> dict:
    """The search answer for `actor`; `filters` are `kind`, `sharing`, `scope` and `page`."""
    with actor.client() as client:
        answered = client.get(SEARCH_PATH, params={"q": q, **filters})
    assert answered.status_code == 200, answered.text
    return answered.json()


def found(actor: Actor, q: str = "", **filters: str | int) -> set[str]:
    return {item["id"] for item in search(actor, q, **filters)["items"]}


def item(answer: dict, item_id: str) -> dict:
    [match] = [entry for entry in answer["items"] if entry["id"] == item_id]
    return match


def conversation(question: str, answer: str) -> dict:
    asked = {"id": "q", "parentId": None, "childrenIds": ["a"], "role": "user"}
    answered = {"id": "a", "parentId": "q", "childrenIds": [], "role": "assistant"}
    messages = {"q": {**asked, "content": question}, "a": {**answered, "content": answer}}
    return {"currentId": "a", "messages": messages}


def import_chat(
    client: httpx.Client,
    title: str,
    *,
    question: str = "hello",
    answer: str = "hi there",
    **import_fields,
) -> str:
    """A stored chat with one question (message `q`) and one answer (message `a`)."""
    chat = {"title": title, "models": [], "history": conversation(question, answer)}
    imported = client.post(
        "/api/v1/chats/import", json={"chats": [{"chat": chat, **import_fields}]}
    )
    assert imported.status_code == 200, imported.text
    return imported.json()[0]["id"]


def post(client: httpx.Client, path: str, **body) -> dict:
    response = client.post(path, json=body or None)
    assert response.status_code == 200, f"POST {path}: HTTP {response.status_code} {response.text}"
    return response.json()


def create_folder(client: httpx.Client, name: str, parent_id: str | None = None) -> str:
    return post(client, "/api/v1/folders/", name=name, parent_id=parent_id)["id"]


def create_note(client: httpx.Client, title: str, text: str) -> str:
    return post(client, "/api/v1/notes/create", title=title, data={"content": {"md": text}})["id"]
