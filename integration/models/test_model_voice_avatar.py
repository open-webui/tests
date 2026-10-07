"""Journey: a model's voice avatar, its state clips and named gestures, over the model routes.

A model's `meta.voice_avatar` names an uploaded VRM (`file_id`), optional VRMA clips for the
idle, listening and speaking states and up to 16 named gestures with a description each. Saving
a model through create, update or import opens every newly named file and refuses, with
400 and the reason, anything that is not a self-contained rigged VRM up to 25 MiB (or, for a
clip, a VRMA body animation up to 10 MiB and 60 seconds), the avatar's own file reused as a
clip, and with 403 a file the saver cannot read. Malformed settings (a repeated or capitalised
gesture name, an empty description, a 17th gesture, an unknown state) are refused with 422.
An update that leaves the avatar out keeps it, `null` removes it. Whoever may read the model may
read its avatar and clips, so a shared model's call can load them; nobody else can. Files come
from `harness.voice_avatars`.

Discriminates: in a backend copy without the avatar check on save, the refusal and access tests
turn red; with the avatar's files dropped from the model's readable files, the reader test turns
red; with the update no longer carrying the stored avatar over, the keep test turns red.
"""

from __future__ import annotations

import io
import uuid

import pytest

from harness.upstream import MOCK_MODEL_ID
from harness.voice_avatars import RIG, animation, avatar, grown_to, upload

Image = pytest.importorskip("PIL.Image")

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

AVATAR_MIB = 1024 * 1024


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (20, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def model_form(model_id: str, voice_avatar, **fields) -> dict:
    meta = {} if voice_avatar is ... else {"voice_avatar": voice_avatar}
    return {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Harbour pilot",
        "meta": meta,
        "params": {},
        **fields,
    }


def create(actor, voice_avatar, **fields):
    model_id = f"avatar-{uuid.uuid4().hex[:8]}"
    with actor.client() as client:
        response = client.post(
            "/api/v1/models/create", json=model_form(model_id, voice_avatar, **fields)
        )
    return model_id, response


def stored_avatar(actor, model_id: str):
    with actor.client() as client:
        model = client.get("/api/v1/models/model", params={"id": model_id})
    assert model.status_code == 200, model.text
    return model.json()["meta"].get("voice_avatar")


@pytest.fixture
def builder(admin, make_user):
    """A user allowed to create and import workspace models, through a group of their own."""
    member = make_user()
    with admin.client() as client:
        group = client.post(
            "/api/v1/groups/create",
            json={
                "name": f"Avatar builders {uuid.uuid4().hex[:6]}",
                "description": "may create and import workspace models",
                "permissions": {"workspace": {"models": True, "models_import": True}},
            },
        )
        assert group.status_code == 200, group.text
        joined = client.post(
            f"/api/v1/groups/id/{group.json()['id']}/users/add", json={"user_ids": [member.id]}
        )
        assert joined.status_code == 200, joined.text
    return member


@pytest.fixture
def full_avatar(builder) -> dict:
    """An avatar of the builder's with an idle and a speaking clip and a wave gesture."""
    return {
        "file_id": upload(builder, "pilot.vrm", avatar()),
        "states": {
            "idle": {"file_id": upload(builder, "idle.vrma", animation())},
            "speaking": {"file_id": upload(builder, "speaking.vrma", animation())},
        },
        "gestures": [
            {
                "name": "wave",
                "description": "Wave when greeting someone.",
                "file_id": upload(builder, "wave.vrma", animation()),
            }
        ],
    }


def test_a_model_keeps_the_avatar_clips_and_gestures_it_was_saved_with(builder, full_avatar):
    model_id, created = create(builder, full_avatar)

    assert created.status_code == 200, created.text
    assert stored_avatar(builder, model_id) == full_avatar


def test_a_vrm_0_avatar_is_accepted_as_well(builder):
    model_id, created = create(builder, {"file_id": upload(builder, "old.vrm", avatar(version=0))})

    assert created.status_code == 200, created.text
    assert stored_avatar(builder, model_id)["file_id"]


def test_an_update_without_the_avatar_keeps_it_and_null_removes_it(builder, full_avatar):
    model_id, created = create(builder, full_avatar)
    assert created.status_code == 200, created.text

    with builder.client() as client:
        renamed = client.post(
            "/api/v1/models/model/update",
            json=model_form(model_id, ..., name="Harbour pilot, renamed"),
        )
        assert renamed.status_code == 200, renamed.text
        assert stored_avatar(builder, model_id) == full_avatar

        removed = client.post("/api/v1/models/model/update", json=model_form(model_id, None))
        assert removed.status_code == 200, removed.text
    assert stored_avatar(builder, model_id) is None


def test_legacy_movement_settings_are_dropped_on_save(builder):
    file_id = upload(builder, "pilot.vrm", avatar())
    legacy = {"file_id": file_id, "preset": "calm", "movement": 0.4, "mouth": 1, "gaze": True}

    model_id, created = create(builder, legacy)

    assert created.status_code == 200, created.text
    assert stored_avatar(builder, model_id) == {"file_id": file_id, "states": {}, "gestures": []}


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("pilot.vrm", _png(8, 8), "Upload a binary VRM file."),
        (
            "plain.glb",
            avatar(bones=tuple(bone for bone in RIG if bone != "hips")),
            "Avatar is missing its hips bone. Upload a rigged VRM file.",
        ),
        (
            "linked.vrm",
            avatar(buffer_uri="https://cdn.example/pilot.bin"),
            "Embed all buffers in the VRM file.",
        ),
        (
            "huge-texture.vrm",
            avatar(texture=_png(4097, 1)),
            "Avatar textures are too large. Export at 2048px or below.",
        ),
        (
            "gif-texture.vrm",
            avatar(texture=_png(8, 8), texture_type="image/gif"),
            "Embed PNG, JPEG or WebP textures in the VRM file.",
        ),
        ("big.vrm", grown_to(avatar(), 25 * AVATAR_MIB + 4), "Avatar must be at most 25 MiB."),
        ("clip.vrm", animation(), "Avatar is missing its hips bone. Upload a rigged VRM file."),
    ],
    ids=[
        "png",
        "no-rig",
        "external-buffer",
        "texture-too-large",
        "gif-texture",
        "over-25-mib",
        "vrma",
    ],
)
def test_a_file_that_is_no_usable_avatar_is_refused_on_save(builder, name, content, reason):
    model_id, created = create(builder, {"file_id": upload(builder, name, content)})

    assert created.status_code == 400, created.text
    assert created.json()["detail"] == reason
    with builder.client() as client:
        assert client.get("/api/v1/models/model", params={"id": model_id}).status_code == 404


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("pilot.vrm", avatar(), "Use a VRMA 1.0 humanoid animation."),
        ("long.vrma", animation(seconds=61), "Use a body animation between 0 and 60 seconds."),
        (
            "big.vrma",
            grown_to(animation(), 10 * AVATAR_MIB + 4),
            "Animation must be at most 10 MiB.",
        ),
    ],
    ids=["vrm", "over-60-seconds", "over-10-mib"],
)
def test_a_clip_that_is_no_usable_animation_is_refused_on_save(builder, name, content, reason):
    clip_id = upload(builder, name, content)
    avatar_id = upload(builder, "pilot.vrm", avatar())

    _, as_state = create(builder, {"file_id": avatar_id, "states": {"idle": {"file_id": clip_id}}})
    gesture = {"name": "wave", "description": "Wave hello.", "file_id": clip_id}
    _, as_gesture = create(builder, {"file_id": avatar_id, "gestures": [gesture]})

    assert (as_state.status_code, as_state.json()["detail"]) == (400, reason)
    assert (as_gesture.status_code, as_gesture.json()["detail"]) == (400, reason)


def test_the_avatar_file_itself_cannot_be_a_clip(builder):
    avatar_id = upload(builder, "pilot.vrm", avatar())

    _, created = create(builder, {"file_id": avatar_id, "states": {"idle": {"file_id": avatar_id}}})

    assert created.status_code == 400, created.text
    assert created.json()["detail"] == "An animation must be a VRMA file, not the avatar."


def _gesture(name: str, description: str = "Wave hello.", file_id: str | None = None) -> dict:
    return {"name": name, "description": description, "file_id": file_id or str(uuid.uuid4())}


@pytest.mark.parametrize(
    "settings",
    [
        {"gestures": [_gesture("wave"), _gesture("wave")]},
        {"gestures": [_gesture("Wave")]},
        {"gestures": [_gesture("1wave")]},
        {"gestures": [_gesture("wave", "   ")]},
        {"gestures": [_gesture(f"gesture_{number}") for number in range(17)]},
        {"states": {"dancing": {"file_id": str(uuid.uuid4())}}},
        {"scale": 2},
    ],
    ids=[
        "repeated-name",
        "capitalised-name",
        "leading-digit",
        "blank-description",
        "seventeen-gestures",
        "unknown-state",
        "unknown-setting",
    ],
)
def test_malformed_avatar_settings_are_refused(builder, settings):
    _, created = create(builder, {"file_id": str(uuid.uuid4()), **settings})

    assert created.status_code == 422, created.text


def test_a_file_id_that_is_no_upload_id_is_refused(builder):
    _, created = create(builder, {"file_id": "../../etc/passwd"})

    assert created.status_code == 422, created.text


def test_someone_elses_file_or_an_unknown_one_cannot_be_the_avatar(builder, make_user):
    others_file = upload(make_user(), "theirs.vrm", avatar())

    _, with_others = create(builder, {"file_id": others_file})
    _, with_unknown = create(builder, {"file_id": str(uuid.uuid4())})

    refusal = "Avatar or animation file is not accessible. Upload it again."
    assert (with_others.status_code, with_others.json()["detail"]) == (403, refusal)
    assert (with_unknown.status_code, with_unknown.json()["detail"]) == (403, refusal)


def test_someone_elses_clip_cannot_be_a_gesture(builder, make_user):
    others_clip = upload(make_user(), "theirs.vrma", animation())
    gesture = _gesture("wave", file_id=others_clip)

    _, created = create(
        builder, {"file_id": upload(builder, "pilot.vrm", avatar()), "gestures": [gesture]}
    )

    assert created.status_code == 403, created.text


def test_an_admin_may_use_any_users_avatar(admin, make_user):
    users_file = upload(make_user(), "theirs.vrm", avatar())

    model_id, created = create(admin, {"file_id": users_file})

    assert created.status_code == 200, created.text
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_bad_avatar_is_refused_on_update_and_the_old_one_kept(builder, full_avatar):
    model_id, created = create(builder, full_avatar)
    assert created.status_code == 200, created.text
    broken = upload(builder, "broken.vrm", _png(8, 8))

    with builder.client() as client:
        updated = client.post(
            "/api/v1/models/model/update", json=model_form(model_id, {"file_id": broken})
        )

    assert updated.status_code == 400, updated.text
    assert stored_avatar(builder, model_id) == full_avatar


def test_an_import_with_a_broken_avatar_or_someone_elses_is_refused(builder, make_user):
    broken = upload(builder, "broken.vrm", _png(8, 8))
    others_file = upload(make_user(), "theirs.vrm", avatar())

    statuses = []
    for file_id in (broken, others_file):
        model_id = f"imported-{uuid.uuid4().hex[:8]}"
        with builder.client() as client:
            imported = client.post(
                "/api/v1/models/import",
                json={"models": [model_form(model_id, {"file_id": file_id})]},
            )
            found = client.get("/api/v1/models/model", params={"id": model_id})
        statuses.append((imported.status_code, imported.json()["detail"], found.status_code))

    assert statuses == [
        (400, "Upload a binary VRM file.", 404),
        (403, "Avatar or animation file is not accessible. Upload it again.", 404),
    ]


def test_whoever_reads_the_model_reads_its_avatar_and_clips_and_nobody_else(
    builder, full_avatar, make_user
):
    reader, stranger = make_user(), make_user()
    read_grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    model_id, created = create(builder, full_avatar, access_grants=[read_grant])
    assert created.status_code == 200, created.text
    files = [
        full_avatar["file_id"],
        *(clip["file_id"] for clip in full_avatar["states"].values()),
        full_avatar["gestures"][0]["file_id"],
    ]

    with reader.client() as client:
        reader_statuses = [
            client.get(f"/api/v1/files/{file}/content").status_code for file in files
        ]
        deleted = client.delete(f"/api/v1/files/{full_avatar['file_id']}")
    with stranger.client() as client:
        stranger_statuses = [
            client.get(f"/api/v1/files/{file}/content").status_code for file in files
        ]

    assert reader_statuses == [200, 200, 200, 200]
    assert stranger_statuses == [404, 404, 404, 404]
    assert deleted.status_code != 200, "a reader of the model deleted its avatar"
