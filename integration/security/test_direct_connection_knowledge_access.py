"""Regression: a direct-connection request could name any knowledge it liked.

open-webui 0.11.0 fix `305880f2e` (#26723): on a direct connection the model object arrives in
the request body as `model_item`. `/api/chat/completions`, `/api/chat/completed` and
`/api/chat/actions/{id}` installed it as sent, so the knowledge it listed (collections, files,
notes) was used without a read check. A file listed there counts as attached to the model, and
the knowledge tools hand an attached file to the model whoever owns it. The fix keeps only the
entries the caller may read before the model is used.

The caller's browser tab answers the completion here (`harness.direct_connection`), plays the
model calling `view_file` on someone else's file and reads what the server sends back. The tab
sends its lines one at a time, since the server can otherwise reorder them and drop the tool call
(open-webui/open-webui#31953). An action is shown the model it runs for, so it reports the
knowledge left on it.

Discriminates: passes on dev ebc6add67; with the filtering removed from `_set_direct_model` the
other account's file comes back through `view_file`, the system prompt names their file and
collection, and the action sees them on the model.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness.chat import send_message, wait_for_reply
from harness.direct_connection import answering, chunk_line, direct_model
from harness.knowledge_bases import knowledge_base
from harness.second_provider import tool_call_delta

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

MODEL = "my-own-model"
THEIR_SECRET = "their payroll figures 4471"
MY_TEXT = "my own shopping list"

REPORTING_ACTION = """
class Action:
    async def action(self, body, __model__=None):
        knowledge = ((__model__ or {}).get("info") or {}).get("meta", {}).get("knowledge") or []
        return {"knowledge_ids": [entry.get("id") for entry in knowledge]}
"""


def _upload(client, filename: str, text: str) -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


@pytest.fixture
def knowledge(admin, make_user):
    """A caller with one file of their own, and a file and collection they may not read."""
    caller, other = make_user(), make_user()
    with caller.client() as client:
        my_file = _upload(client, "mine.txt", MY_TEXT)
    with other.client() as client:
        their_file = _upload(client, "payroll.txt", THEIR_SECRET)
    with admin.client() as client, knowledge_base(client, "Board minutes") as their_collection:
        yield {
            "caller": caller,
            "my_file": my_file,
            "their_file": their_file,
            "their_collection": their_collection,
        }


def _claiming(knowledge: dict) -> dict:
    """The caller's direct model, claiming their own file and two things they cannot read."""
    model = direct_model(MODEL)
    model["info"]["meta"]["knowledge"] = [
        {"type": "file", "id": knowledge["my_file"], "name": "mine.txt"},
        {"type": "file", "id": knowledge["their_file"], "name": "payroll.txt"},
        {"type": "collection", "id": knowledge["their_collection"], "name": "Board minutes"},
    ]
    return model


def _view_file(knowledge: dict, file_id: str) -> tuple[dict, str]:
    """Have the tab's model call `view_file`; returns its first request and the tool result."""
    caller = knowledge["caller"]
    with answering(caller, in_order=True) as tab, caller.client() as client:
        tab.stream(
            chunk_line(tool_call_delta("view_file", {"file_id": file_id})),
            chunk_line({}, "tool_calls"),
        )
        tab.stream(chunk_line({"content": "read it"}), chunk_line({}, "stop"))
        turn = send_message(
            client,
            "read the file",
            model=MODEL,
            model_item=_claiming(knowledge),
            session_id=tab.session_id,
        )
        wait_for_reply(client, turn)

    first, follow_up = tab.requests
    [tool_result] = [
        entry["content"] for entry in follow_up["form_data"]["messages"] if entry["role"] == "tool"
    ]
    return first, tool_result


def _system_prompt(request: dict) -> str:
    messages = request["form_data"]["messages"]
    return "\n".join(entry["content"] for entry in messages if entry["role"] == "system")


# narrow: knowledge the caller cannot read is dropped from the direct model


def test_a_claimed_file_of_another_account_cannot_be_read(knowledge):
    _, tool_result = _view_file(knowledge, knowledge["their_file"])

    assert THEIR_SECRET not in tool_result, (
        "a direct connection listed another account's file as its knowledge and view_file "
        f"handed its content to the model (#26723): {tool_result[:200]}"
    )
    assert "not found" in tool_result.lower(), tool_result


def test_the_model_is_only_told_about_knowledge_the_caller_can_read(knowledge):
    first_request, _ = _view_file(knowledge, knowledge["my_file"])
    system_prompt = _system_prompt(first_request)

    assert knowledge["my_file"] in system_prompt, system_prompt
    assert knowledge["their_file"] not in system_prompt, (
        f"the model was offered another account's file as attached knowledge (#26723): "
        f"{system_prompt}"
    )
    assert knowledge["their_collection"] not in system_prompt, (
        f"the model was offered a collection the caller cannot read (#26723): {system_prompt}"
    )


@pytest.fixture
def reporting_action(admin):
    """A global action that answers with the knowledge ids on the model it runs for."""
    function_id = f"report_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/functions/create",
            json={
                "id": function_id,
                "name": "Report knowledge",
                "content": REPORTING_ACTION,
                "meta": {"description": "reports the model's knowledge"},
            },
        )
        assert created.status_code == 200, created.text
        client.post(f"/api/v1/functions/id/{function_id}/toggle").raise_for_status()
        client.post(f"/api/v1/functions/id/{function_id}/toggle/global").raise_for_status()
    yield function_id
    with admin.client() as client:
        client.delete(f"/api/v1/functions/id/{function_id}/delete")


def test_an_action_on_a_direct_model_only_sees_readable_knowledge(knowledge, reporting_action):
    body = {
        "model": MODEL,
        "model_item": _claiming(knowledge),
        "messages": [{"id": "message-1", "role": "assistant", "content": "a reply"}],
        "chat_id": "local:harness-socket",
        "id": "message-1",
        "session_id": "harness-session",
    }
    with knowledge["caller"].client() as client:
        response = client.post(f"/api/chat/actions/{reporting_action}", json=body)

    assert response.status_code == 200, response.text
    assert response.json()["knowledge_ids"] == [knowledge["my_file"]], (
        f"the action ran on a direct model still listing knowledge the caller cannot read "
        f"(#26723): {json.dumps(response.json())}"
    )


# nearby: the caller's own attached file still reads


def test_the_callers_own_claimed_file_still_reads(knowledge):
    _, tool_result = _view_file(knowledge, knowledge["my_file"])

    assert MY_TEXT in tool_result, tool_result
