"""A VRM avatar and VRMA body animations built in memory, small enough to upload in a test.

A VRM is a binary glTF (GLB: a header, a JSON chunk, a binary chunk) whose `VRMC_vrm` extension
(`VRM` in 0.x) names the nodes of a humanoid rig. `avatar()` writes a meshless rig standing on
its feet with every bone the three-vrm loader requires, the `aa` (mouth) and `blink` expressions
and an embedded buffer; its keywords leave bones or expressions out, embed a texture, point the
buffer at a URL or write the 0.x layout. `animation()` writes a VRMA, the same rig under
`VRMC_vrm_animation` with one clip that turns the head and both upper arms over `seconds`.
`glb(document, binary)` wraps any glTF JSON for a hand-made broken file and `grown_to(content,
size)` pads a file's binary chunk out to `size` bytes for the size limits. `upload(actor, name,
content)` uploads one as the model editor does (unprocessed) and returns its file id.
"""

from __future__ import annotations

import json
import math
import struct

from harness.actors import Actor

GLB_MAGIC, JSON_CHUNK, BINARY_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
FLOAT = 5126
LICENSE_URL = "https://vrm.dev/licenses/1.0/"

# (bone, translation from its parent, child bones)
RIG = {
    "hips": ((0.0, 0.9, 0.0), ("spine", "leftUpperLeg", "rightUpperLeg")),
    "spine": ((0.0, 0.1, 0.0), ("chest",)),
    "chest": ((0.0, 0.15, 0.0), ("neck", "leftUpperArm", "rightUpperArm")),
    "neck": ((0.0, 0.15, 0.0), ("head",)),
    "head": ((0.0, 0.1, 0.0), ()),
    "leftUpperArm": ((0.1, 0.1, 0.0), ("leftLowerArm",)),
    "leftLowerArm": ((0.25, 0.0, 0.0), ("leftHand",)),
    "leftHand": ((0.22, 0.0, 0.0), ()),
    "rightUpperArm": ((-0.1, 0.1, 0.0), ("rightLowerArm",)),
    "rightLowerArm": ((-0.25, 0.0, 0.0), ("rightHand",)),
    "rightHand": ((-0.22, 0.0, 0.0), ()),
    "leftUpperLeg": ((0.08, -0.05, 0.0), ("leftLowerLeg",)),
    "leftLowerLeg": ((0.0, -0.4, 0.0), ("leftFoot",)),
    "leftFoot": ((0.0, -0.4, 0.0), ()),
    "rightUpperLeg": ((-0.08, -0.05, 0.0), ("rightLowerLeg",)),
    "rightLowerLeg": ((0.0, -0.4, 0.0), ("rightFoot",)),
    "rightFoot": ((0.0, -0.4, 0.0), ()),
}
# the bones Open WebUI's own check asks for; three-vrm asks for all of RIG but chest and neck
SERVER_CHECKED_BONES = (
    "hips",
    "spine",
    "head",
    "leftUpperArm",
    "rightUpperArm",
    "leftLowerArm",
    "rightLowerArm",
)


def glb(document: dict, binary: bytes = b"\x00" * 4) -> bytes:
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<3I", GLB_MAGIC, 2, total)
        + struct.pack("<2I", len(text), JSON_CHUNK)
        + text
        + struct.pack("<2I", len(binary), BINARY_CHUNK)
        + binary
    )


def grown_to(content: bytes, size: int) -> bytes:
    """`content` with its binary chunk grown to `size` bytes in all, its headers kept consistent."""
    extra = size - len(content)
    grown = bytearray(content + b"\x00" * extra)
    struct.pack_into("<I", grown, 8, size)
    (json_size,) = struct.unpack_from("<I", grown, 12)
    (binary_size,) = struct.unpack_from("<I", grown, 20 + json_size)
    struct.pack_into("<I", grown, 20 + json_size, binary_size + extra)
    return bytes(grown)


def _rig_nodes() -> tuple[list[dict], dict[str, int]]:
    order = list(RIG)
    index = {bone: position for position, bone in enumerate(order)}
    nodes = [
        {
            "name": bone,
            "translation": list(RIG[bone][0]),
            **({"children": [index[child] for child in RIG[bone][1]]} if RIG[bone][1] else {}),
        }
        for bone in order
    ]
    return nodes, index


def avatar(
    *,
    bones: tuple[str, ...] | None = None,
    expressions: tuple[str, ...] = ("aa", "blink"),
    texture: bytes | None = None,
    texture_type: str = "image/png",
    buffer_uri: str | None = None,
    version: int = 1,
) -> bytes:
    nodes, index = _rig_nodes()
    named = bones if bones is not None else tuple(RIG)
    binary = b"\x00" * 4
    document: dict = {
        "asset": {"version": "2.0", "generator": "open-webui tests"},
        "scene": 0,
        "scenes": [{"nodes": [index["hips"]]}],
        "nodes": nodes,
    }
    if texture is not None:
        binary = texture + b"\x00" * (-len(texture) % 4)
        document["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(texture)}]
        document["images"] = [{"bufferView": 0, "mimeType": texture_type}]
    buffer = {"byteLength": len(binary)}
    if buffer_uri:
        buffer["uri"] = buffer_uri
    document["buffers"] = [buffer]
    if version == 1:
        document["extensionsUsed"] = ["VRMC_vrm"]
        document["extensions"] = {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {
                    "name": "Harbour pilot",
                    "authors": ["open-webui tests"],
                    "licenseUrl": LICENSE_URL,
                    "avatarPermission": "everyone",
                },
                "humanoid": {"humanBones": {bone: {"node": index[bone]} for bone in named}},
                "expressions": {"preset": {name: {} for name in expressions}},
            }
        }
    else:
        document["extensionsUsed"] = ["VRM"]
        document["extensions"] = {
            "VRM": {
                "specVersion": "0.0",
                "meta": {"title": "Harbour pilot", "author": "open-webui tests"},
                "humanoid": {"humanBones": [{"bone": bone, "node": index[bone]} for bone in named]},
                "blendShapeMaster": {
                    "blendShapeGroups": [
                        {"name": name, "presetName": name.lower(), "binds": []}
                        for name in expressions
                    ]
                },
            }
        }
    return glb(document, binary)


def _turn(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    half = math.sin(angle / 2)
    return (axis[0] * half, axis[1] * half, axis[2] * half, math.cos(angle / 2))


def animation(seconds: float = 1.0) -> bytes:
    nodes, index = _rig_nodes()
    times = (0.0, seconds / 2, seconds)
    turns = {
        "head": ((0.0, 1.0, 0.0), 0.4),
        "leftUpperArm": ((0.0, 0.0, 1.0), 1.0),
        "rightUpperArm": ((0.0, 0.0, 1.0), -1.0),
    }
    binary = struct.pack("<3f", *times)
    accessors = [{"bufferView": 0, "componentType": FLOAT, "count": 3, "type": "SCALAR"}]
    accessors[0]["min"], accessors[0]["max"] = [times[0]], [times[-1]]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}]
    channels, samplers = [], []
    for bone, (axis, angle) in turns.items():
        keys = (_turn(axis, 0.0), _turn(axis, angle), _turn(axis, 0.0))
        values = b"".join(struct.pack("<4f", *key) for key in keys)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(values)})
        binary += values
        accessors.append(
            {"bufferView": len(views) - 1, "componentType": FLOAT, "count": 3, "type": "VEC4"}
        )
        samplers.append({"input": 0, "output": len(accessors) - 1, "interpolation": "LINEAR"})
        channels.append(
            {"sampler": len(samplers) - 1, "target": {"node": index[bone], "path": "rotation"}}
        )
    document = {
        "asset": {"version": "2.0", "generator": "open-webui tests"},
        "scene": 0,
        "scenes": [{"nodes": [index["hips"]]}],
        "nodes": nodes,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "animations": [{"channels": channels, "samplers": samplers}],
        "extensionsUsed": ["VRMC_vrm_animation"],
        "extensions": {
            "VRMC_vrm_animation": {
                "specVersion": "1.0",
                "humanoid": {"humanBones": {bone: {"node": index[bone]} for bone in RIG}},
            }
        },
    }
    return glb(document, binary)


def upload(actor: Actor, name: str, content: bytes) -> str:
    with actor.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "false"},
            files={"file": (name, content, "application/octet-stream")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]
