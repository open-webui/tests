"""What a stored note holds, read the way its editor saved it.

A note keeps its files and pasted pictures inside its pages' content: each file is an attachment
block naming the uploaded file, each picture an image whose address is the picture itself as a
data URL. `note_files(note)` lists the attachment blocks of every page in order, and
`note_images(note)` the image addresses.
"""

from __future__ import annotations

from typing import Iterator


def _nodes(note: dict) -> Iterator[dict]:
    pages = (note.get("data") or {}).get("pages") or []
    stack = [(page.get("content") or {}).get("json") for page in reversed(pages)]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        yield node
        stack.extend(reversed(node.get("content") or []))


def note_files(note: dict) -> list[dict]:
    """Each attached file as `{"id", "name", "size"}`."""
    return [
        node.get("attrs") or {} for node in _nodes(note) if node.get("type") == "fileAttachment"
    ]


def note_images(note: dict) -> list[str]:
    return [
        (node.get("attrs") or {}).get("src") or ""
        for node in _nodes(note)
        if node.get("type") == "image"
    ]
