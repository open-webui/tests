"""Journey: model controls, the parameter presets an admin approves and people pick in chat.

An admin stores controls on a model (`params.model_controls`); every option carries the
parameters it sends. A chat request names the account's picks by model id and control key, and
the server merges each picked option, or the control's default when nothing is picked, into the
custom parameters the provider gets: they are sent as plain top-level fields, string values read
as JSON. A pick is sent for its parameters even over the chat's or the model's own value, picks
for one model never reach another, and a pick of an option that is gone is refused; the chat
input still offers that control, so the user can pick again. A pick of a removed control is
pinned in the browser suite, where it leaves the user stuck.
Picking needs the Chat Controls and Advanced Params permissions; the defaults apply regardless.
On the model routes, people see the labels of the controls and options but never what an option
sends, only admins read or change the controls, a writer's save without them keeps the admin's,
malformed controls are refused and a model on a pipe cannot carry any.

Discriminates: passes on dev ebc6add67. In a backend copy whose chat route checks the picks but
never merges them, every payload test and the permission test fail; in one that sends each
control's default whatever was picked, with no permission check and no keeping of omitted
controls, the picked-option, per-model, permission and writer's save tests fail; in one that
gives every model the first model's picks and returns the full controls to everyone, the per-model
and both visibility tests fail; in one without the pick, key, label, option and default
validation, the writer and user checks and the pipe check, every refusal test fails.
"""

from __future__ import annotations

import copy
import uuid

import pytest

from harness.chat import wait_for_reply
from harness.model_controls import LENGTH, THINKING, ask_with_picks, preset_with_controls
from harness.plugins import installed_function
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PIPE_SOURCE = """class Pipe:
    def pipe(self, body: dict) -> str:
        return "piped"
"""


def _sent(client, upstream, model_id: str, selections: dict, **options) -> dict:
    """The provider request for one message on `model_id` carrying `selections`."""
    question = f"how warm is the lagoon? {uuid.uuid4().hex[:6]}"
    turn = ask_with_picks(client, question, selections, model=model_id, **options)
    reply = wait_for_reply(client, turn)
    assert not reply.get("error"), reply.get("error")
    return [body for body in upstream.chat_requests() if question in str(body["messages"])][-1]


def _set_chat_permissions(admin, **flags: bool) -> None:
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(flags)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def _stored_controls(admin, model_id: str) -> dict:
    with admin.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model_id})
    assert stored.status_code == 200, stored.text
    return stored.json()["params"].get("model_controls")


def _writer_grants(account) -> list[dict]:
    return [
        {"principal_type": "user", "principal_id": account.id, "permission": permission}
        for permission in ("read", "write")
    ]


# --------------------------------------------------------------------------- what is sent


def test_a_controls_default_is_sent_when_nothing_is_picked(admin, make_user, upstream):
    with preset_with_controls(admin, {"thinking": THINKING}) as model:
        with make_user().client() as client:
            sent = _sent(client, upstream, model["id"], {})

    assert sent.get("reasoning_effort") == "medium"
    assert "max_tokens" not in sent


def test_a_picked_option_is_sent_in_place_of_the_default(admin, make_user, upstream):
    with preset_with_controls(admin, {"thinking": THINKING}) as model:
        with make_user().client() as client:
            sent = _sent(client, upstream, model["id"], {model["id"]: {"thinking": "high"}})

    assert (sent.get("reasoning_effort"), sent.get("max_tokens")) == ("high", 4096)
    assert "model_controls" not in sent


def test_a_control_without_a_default_sends_nothing_until_picked(admin, make_user, upstream):
    with preset_with_controls(admin, {"length": LENGTH}) as model:
        with make_user().client() as client:
            unpicked = _sent(client, upstream, model["id"], {})
            picked = _sent(client, upstream, model["id"], {model["id"]: {"length": "thorough"}})

    assert "verbosity" not in unpicked and "text" not in unpicked
    assert picked.get("verbosity") == "high"
    assert picked.get("text") == {"format": {"type": "text"}}


def test_every_control_of_the_model_applies_at_once(admin, make_user, upstream):
    controls = {"thinking": THINKING, "length": LENGTH}
    with preset_with_controls(admin, controls) as model:
        with make_user().client() as client:
            picks = {model["id"]: {"thinking": "low", "length": "brief"}}
            sent = _sent(client, upstream, model["id"], picks)

    assert (sent.get("reasoning_effort"), sent.get("verbosity")) == ("low", "low")


def test_a_pick_wins_for_its_parameter_and_the_rest_still_apply(admin, make_user, upstream):
    model_params = {"reasoning_effort": "minimal", "top_p": 0.5, "custom_params": {"x_model": "1"}}
    chat_params = {"temperature": 0.2, "reasoning_effort": "none"}
    with preset_with_controls(admin, {"thinking": THINKING}, params=model_params) as model:
        with make_user().client() as client:
            by_default = _sent(client, upstream, model["id"], {}, params=chat_params)
            picks = {model["id"]: {"thinking": "low"}}
            picked = _sent(client, upstream, model["id"], picks, params=chat_params)

    for sent, effort in ((by_default, "medium"), (picked, "low")):
        assert sent.get("reasoning_effort") == effort
        assert (sent.get("temperature"), sent.get("top_p"), sent.get("x_model")) == (0.2, 0.5, 1)


def test_picks_apply_only_to_the_model_they_were_made_for(admin, make_user, upstream):
    with (
        preset_with_controls(admin, {"thinking": THINKING}, name="Harbour pilot") as picked_model,
        preset_with_controls(admin, {"thinking": THINKING}, name="Lighthouse keeper") as other,
    ):
        picks = {picked_model["id"]: {"thinking": "high"}}
        with make_user().client() as client:
            on_picked = _sent(client, upstream, picked_model["id"], picks)
            on_other = _sent(client, upstream, other["id"], picks)
            on_plain = _sent(client, upstream, MOCK_MODEL_ID, picks)

    assert on_picked.get("reasoning_effort") == "high"
    assert on_other.get("reasoning_effort") == "medium"
    assert "reasoning_effort" not in on_plain


# --------------------------------------------------------------------------- refused picks


def _refused(client, model_id: str, selections: dict):
    """A message carrying `selections`, sent the way an API client does (answered inline)."""
    return client.post(
        "/api/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "params": {"model_controls": selections},
        },
    )


def test_a_pick_of_an_option_that_is_gone_is_refused(admin, make_user, upstream):
    with preset_with_controls(admin, {"thinking": THINKING}) as model:
        with make_user().client() as client:
            refused = _refused(client, model["id"], {model["id"]: {"thinking": "extreme"}})

    assert refused.status_code == 400
    expected = "Model control thinking: the selected option is no longer available."
    assert refused.json()["detail"] == expected
    assert upstream.chat_requests() == []


def test_picks_not_keyed_by_model_are_refused(make_user, upstream):
    with make_user().client() as client:
        refused = _refused(client, MOCK_MODEL_ID, ["thinking"])

    assert refused.status_code == 400
    assert refused.json()["detail"] == "Model control options must be keyed by model."


@pytest.mark.parametrize("withheld", ["controls", "params"])
def test_picking_needs_the_chat_parameter_permissions(
    admin, make_user, upstream, preserve, withheld
):
    preserve("permissions")
    _set_chat_permissions(admin, **{withheld: False})
    with preset_with_controls(admin, {"thinking": THINKING}) as model:
        with make_user().client() as client:
            refused = _refused(client, model["id"], {model["id"]: {"thinking": "high"}})
            defaults_only = _sent(client, upstream, model["id"], {model["id"]: {}})
        with admin.client() as client:
            as_admin = _sent(client, upstream, model["id"], {model["id"]: {"thinking": "high"}})

    assert refused.status_code == 403
    assert refused.json()["detail"] == "You cannot change model parameters."
    assert defaults_only.get("reasoning_effort") == "medium"
    assert as_admin.get("reasoning_effort") == "high"


# --------------------------------------------------------------------------- model routes


def test_people_see_the_labels_and_never_what_an_option_sends(admin, make_user):
    with preset_with_controls(admin, {"thinking": THINKING, "length": LENGTH}) as model:
        with make_user().client() as client:
            listed = client.get("/api/models").json()["data"]
    offered = next(entry for entry in listed if entry["id"] == model["id"])

    assert offered["info"]["params"] == {
        "model_controls": {
            "thinking": {
                "label": "Thinking",
                "description": "How long the model thinks before it answers",
                "default": "medium",
                "options": {
                    "low": {"label": "Low"},
                    "medium": {"label": "Medium"},
                    "high": {"label": "High"},
                },
            },
            "length": {
                "label": "Answer length",
                "options": {"brief": {"label": "Brief"}, "thorough": {"label": "Thorough"}},
            },
        }
    }


def test_only_admins_read_the_controls_from_the_model_routes(admin, make_user):
    writer = make_user()
    model_params = {"temperature": 0.4}
    with preset_with_controls(
        admin, {"thinking": THINKING}, params=model_params, access_grants=_writer_grants(writer)
    ) as model:
        with writer.client() as client:
            as_writer = client.get("/api/v1/models/model", params={"id": model["id"]}).json()
            listed = client.get("/api/v1/models/list").json()["items"]
        as_admin = _stored_controls(admin, model["id"])
    writer_listed = next(entry for entry in listed if entry["id"] == model["id"])

    assert as_admin == {"thinking": THINKING}
    assert as_writer["params"] == model_params
    assert "model_controls" not in writer_listed["params"]


def test_a_writer_saving_without_controls_keeps_the_admins(admin, make_user):
    writer = make_user()
    with preset_with_controls(
        admin, {"thinking": THINKING}, access_grants=_writer_grants(writer)
    ) as model:
        with writer.client() as client:
            loaded = client.get("/api/v1/models/model", params={"id": model["id"]}).json()
            saved = client.post(
                "/api/v1/models/model/update", json={**loaded, "name": "Renamed by the writer"}
            )
        assert saved.status_code == 200, saved.text
        kept = _stored_controls(admin, model["id"])

    assert kept == {"thinking": THINKING}


def test_a_writer_cannot_change_the_controls(admin, make_user):
    writer = make_user()
    changed = copy.deepcopy(THINKING)
    changed["options"]["high"]["params"] = {"reasoning_effort": "high", "max_tokens": "999999"}
    with preset_with_controls(
        admin, {"thinking": THINKING}, access_grants=_writer_grants(writer)
    ) as model:
        with writer.client() as client:
            loaded = client.get("/api/v1/models/model", params={"id": model["id"]}).json()
            for controls in ({"thinking": changed}, {}):
                params = {**loaded["params"], "model_controls": controls}
                refused = client.post(
                    "/api/v1/models/model/update", json={**loaded, "params": params}
                )
                assert refused.status_code == 403, refused.text
                assert refused.json()["detail"] == "Only admins can change model controls."
        kept = _stored_controls(admin, model["id"])

    assert kept == {"thinking": THINKING}


def test_a_user_cannot_create_a_model_with_controls(make_user, preserve, admin):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["workspace"]["models"] = True
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()

    with make_user().client() as client:
        refused = client.post(
            "/api/v1/models/create",
            json={
                "id": f"self-tuned-{uuid.uuid4().hex[:8]}",
                "name": "Self tuned",
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {"model_controls": {"thinking": THINKING}},
            },
        )

    assert refused.status_code == 403
    assert refused.json()["detail"] == "Only admins can change model controls."


def _with(control: dict, **changes) -> dict:
    return {**copy.deepcopy(control), **changes}


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": _with(THINKING, default="extreme")},
        {"thinking": _with(THINKING, options={})},
        {"thinking": _with(THINKING, label="  ")},
        {"thinking": _with(THINKING, options={"low": {"label": "", "params": {}}})},
        {"constructor": THINKING},
        {"1st": THINKING},
        {"thinking": _with(THINKING, options={"Bad key!": {"label": "Low", "params": {}}})},
    ],
    ids=[
        "default-not-an-option",
        "no-options",
        "blank-label",
        "blank-option-label",
        "reserved-key",
        "key-starting-with-a-digit",
        "option-key-with-spaces",
    ],
)
def test_malformed_controls_are_refused(admin, controls):
    model_id = f"malformed-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        refused = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": "Malformed",
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {"model_controls": controls},
            },
        )
        stored = client.get("/api/v1/models/model", params={"id": model_id})

    assert refused.status_code == 422, refused.text
    assert stored.status_code != 200


def test_a_model_on_a_pipe_cannot_carry_controls(admin):
    with installed_function(admin, PIPE_SOURCE) as pipe_id, admin.client() as client:
        client.get("/api/models").raise_for_status()  # registers the pipe as a model
        refused = client.post(
            "/api/v1/models/create",
            json={
                "id": f"piped-{uuid.uuid4().hex[:8]}",
                "name": "Piped",
                "base_model_id": pipe_id,
                "meta": {},
                "params": {"model_controls": {"thinking": THINKING}},
            },
        )

    assert refused.status_code == 400
    assert refused.json()["detail"] == "Model controls require a server-managed provider model."
