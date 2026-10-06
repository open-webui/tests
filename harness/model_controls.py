"""Model controls: parameter presets an admin approves on a model, which people pick in chat.

An admin gives a model `params.model_controls`, each control a label, an optional description
and default, and options that each carry the parameters they send. The chat input offers the
controls of the selected model, and what a person picks is kept in their own settings under
`ui.params.model_controls`, by model id and control key; the web client sends that map with
every message and the server merges the picked options (or each control's default) into the
custom parameters the provider gets.

`THINKING` (a default, an option with two parameters) and `LENGTH` (no default, a nested value)
are the controls the tests share. `preset_with_controls(admin, controls)` is a preset on the
scripted model with those controls that every account may read, deleted again afterwards;
`pick(actor, selections)` saves an account's picks the way the chat input does, and
`ask_with_picks(client, selections, ...)` sends a message carrying them as the web client does.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import httpx

from harness.actors import Actor
from harness.chat import ChatTurn, send_message
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

THINKING = {
    "label": "Thinking",
    "description": "How long the model thinks before it answers",
    "default": "medium",
    "options": {
        "low": {"label": "Low", "params": {"reasoning_effort": "low"}},
        "medium": {"label": "Medium", "params": {"reasoning_effort": "medium"}},
        "high": {"label": "High", "params": {"reasoning_effort": "high", "max_tokens": "4096"}},
    },
}

LENGTH = {
    "label": "Answer length",
    "options": {
        "brief": {"label": "Brief", "params": {"verbosity": "low"}},
        "thorough": {
            "label": "Thorough",
            "params": {"verbosity": "high", "text": {"format": {"type": "text"}}},
        },
    },
}


@contextlib.contextmanager
def preset_with_controls(
    admin: Actor,
    controls: dict,
    *,
    name: str | None = None,
    params: dict | None = None,
    access_grants: list[dict] | None = None,
) -> Iterator[dict]:
    """A preset on the scripted model carrying `controls`; yields `{"id", "name"}`."""
    model = {"id": f"controls-{uuid.uuid4().hex[:8]}", "name": name or "Tuned helper"}
    form = {
        **model,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"description": "a preset with model controls"},
        "params": {**(params or {}), "model_controls": controls},
        "access_grants": [EVERYONE_READS] if access_grants is None else access_grants,
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, f"creating {model['id']} failed: {created.text}"
        try:
            client.get("/api/models").raise_for_status()  # registers the preset for chats
            yield model
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model["id"]})


def pick(actor: Actor, selections: dict) -> None:
    """Save `selections` ({model id: {control: option}}) as the account's picks."""
    with actor.client() as client:
        current = client.get("/api/v1/users/user/settings").json() or {}
        params = {**((current.get("ui") or {}).get("params") or {}), "model_controls": selections}
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": {"params": params}})
    assert saved.status_code == 200, saved.text


def ask_with_picks(
    client: httpx.Client, content: str, selections: dict, *, params: dict | None = None, **options
) -> ChatTurn:
    """Send `content` with the picks and any chat parameters in `params`, as the web client does."""
    return send_message(
        client, content, params={**(params or {}), "model_controls": selections}, **options
    )
