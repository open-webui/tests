"""Small files of the kinds a file preview opens, built in memory for a test to put on a terminal.

`pdf` writes one line of text per page and can pin a note with an author on the first page.
`docx` writes paragraphs and can attach a review comment to the first one, `pptx` writes one
slide per title and can leave a classic comment on the first slide. `wav` is a second of a
tone, `webm` a second of video in a codec every Chromium plays, `stl` a single triangle and
`png` a flat picture of the given size. Each returns the file's bytes.
"""

from __future__ import annotations

import io
import math
import struct
import wave
import zlib
from typing import NamedTuple


class Note(NamedTuple):
    """A reviewer's comment on a document."""

    author: str
    text: str


def png(width: int = 64, height: int = 48, colour: bytes = b"\x20\x80\xe0") -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + colour * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def wav(seconds: float = 1.0, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(rate)
        samples = (
            int(12000 * math.sin(2 * math.pi * 440 * index / rate))
            for index in range(int(seconds * rate))
        )
        sound.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return buffer.getvalue()


def webm(seconds: int = 1, width: int = 64, height: int = 48) -> bytes:
    import av
    import numpy

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="webm") as container:
        stream = container.add_stream("libvpx", rate=10)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(seconds * 10):
            picture = numpy.full((height, width, 3), (index * 20) % 255, dtype=numpy.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def stl(name: str = "wedge") -> bytes:
    return (
        f"solid {name}\n"
        "facet normal 0 0 1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 10 0 0\n  vertex 0 10 0\n"
        " endloop\nendfacet\n"
        f"endsolid {name}\n"
    ).encode()


def pdf(pages: list[str], note: Note | None = None) -> bytes:
    """A PDF with one line of text on each page; `note` is pinned beside the first page's line."""

    def text(value: str) -> str:
        return "(" + value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    number = 4
    for index, line in enumerate(pages):
        page_number, content_number = number, number + 1
        number += 2
        stream = f"BT /F1 24 Tf 72 700 Td {text(line)} Tj ET".encode()
        objects[content_number] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        annotations = ""
        if note and index == 0:
            annotations = f" /Annots [{number} 0 R]"
            objects[number] = (
                "<< /Type /Annot /Subtype /Text /Rect [72 660 92 680] /Name /Comment"
                f" /T {text(note.author)} /Contents {text(note.text)} /F 4 >>"
            ).encode()
            number += 1
        objects[page_number] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R"
            f"{annotations} >>"
        ).encode()
        kids.append(f"{page_number} 0 R")
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()

    body = b"%PDF-1.4\n"
    offsets = {}
    for object_number in sorted(objects):
        offsets[object_number] = len(body)
        body += f"{object_number} 0 obj\n".encode() + objects[object_number] + b"\nendobj\n"
    table_at = len(body)
    body += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for object_number in sorted(objects):
        body += f"{offsets[object_number]:010d} 00000 n \n".encode()
    body += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{table_at}\n%%EOF\n"
    ).encode()
    return body


def docx(paragraphs: list[str], note: Note | None = None) -> bytes:
    """A Word document; `note` is a review comment on the first paragraph."""
    from docx import Document

    document = Document()
    written = [document.add_paragraph(paragraph) for paragraph in paragraphs]
    if note:
        document.add_comment(written[0].runs, text=note.text, author=note.author, initials="")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


COMMENT_AUTHORS_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.commentAuthors+xml"
)
COMMENTS_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.comments+xml"
OFFICE_RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PRESENTATION_NAMESPACE = "http://schemas.openxmlformats.org/presentationml/2006/main"


def pptx(titles: list[str], note: Note | None = None) -> bytes:
    """A presentation with one titled slide per entry; `note` is a comment on the first slide."""
    from pptx import Presentation
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI

    deck = Presentation()
    slides = []
    for title in titles:
        slide = deck.slides.add_slide(deck.slide_layouts[5])
        slide.shapes.title.text = title
        slides.append(slide)
    if note:
        package = deck.part.package
        authors = Part(
            PackURI("/ppt/commentAuthors.xml"),
            COMMENT_AUTHORS_TYPE,
            package,
            (
                f'<p:cmAuthorLst xmlns:p="{PRESENTATION_NAMESPACE}">'
                f'<p:cmAuthor id="0" name="{note.author}" initials="" lastIdx="1" clrIdx="0"/>'
                "</p:cmAuthorLst>"
            ).encode(),
        )
        deck.part.relate_to(authors, f"{OFFICE_RELATIONSHIPS}/commentAuthors")
        comments = Part(
            PackURI("/ppt/comments/comment1.xml"),
            COMMENTS_TYPE,
            package,
            (
                f'<p:cmLst xmlns:p="{PRESENTATION_NAMESPACE}">'
                '<p:cm authorId="0" dt="2026-03-04T09:00:00.000" idx="1">'
                f'<p:pos x="600" y="600"/><p:text>{note.text}</p:text></p:cm>'
                "</p:cmLst>"
            ).encode(),
        )
        slides[0].part.relate_to(comments, f"{OFFICE_RELATIONSHIPS}/comments")
    buffer = io.BytesIO()
    deck.save(buffer)
    return buffer.getvalue()
