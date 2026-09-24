"""Dependency contract: chardet.

Open WebUI's one chardet call is in ``Loader._detect_text_encoding``
(``retrieval/loaders/main.py``). When an uploaded text file is not UTF-8 it runs
``chardet.detect(sample)`` on the bytes around the first bad byte and reads
``(detected.get('encoding') or '')`` as a hint: a CJK hint puts that codec family first, any
other hint is tried with ``raw.decode(hint)`` once the CJK codecs fail, and latin-1 is the last
resort. So what the loader needs is narrow: ``detect`` takes bytes, never raises on them, returns
a mapping whose ``'encoding'`` is a codec name or None, and for real text that name is a codec
Python can decode the bytes with, correctly for CJK text.

Which name chardet picks for a codec family is not pinned: chardet 5 said SHIFT_JIS and EUC-KR,
chardet 7 says cp932 and CP949, and each decodes the text. (The loader's own map lacks cp949;
integration/deps/test_document_extraction.py pins that, #31352.) Two earlier
assertions on exact guesses (empty input, pure ASCII) broke on the 5 to 7 bump for nothing the
loader reads, so none are made here.

Discriminates: with ``detect`` answering ``ascii`` for everything (a pytest plugin patching the
installed library), the CJK and Western cases go red.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import codecs

import pytest

pytestmark = pytest.mark.depcheck

CJK_SAMPLES = {
    "gb18030": "港口灯塔的预算已经批准。简体中文编码测试。" * 8,
    "big5": "港口燈塔的預算已經批准。繁體中文編碼測試。" * 8,
    "shift_jis": "港の灯台の予算が承認されました。日本語の文字コード判定。" * 8,
    "euc-kr": "항구 등대 예산이 승인되었습니다. 한국어 인코딩 감지 테스트." * 8,
}

WESTERN = "Smart “quotes” and an ellipsis… from Zürich, Grüße." * 4

ANY_BYTES = {
    "ascii": b"plain 7-bit text",
    "empty": b"",
    "noise": bytes(range(256)) * 4,
    **{codec: text.encode(codec) for codec, text in CJK_SAMPLES.items()},
}


def _hint(depcheck, raw: bytes) -> str | None:
    """The loader's exact read of the result."""
    return depcheck.load("chardet").detect(raw).get("encoding")


def test_detect_is_exported(depcheck):
    depcheck.assert_callable(depcheck.load("chardet"), "detect")


@pytest.mark.parametrize("sample", sorted(ANY_BYTES))
def test_detect_returns_a_codec_name_or_none_and_never_raises(depcheck, sample):
    hint = _hint(depcheck, ANY_BYTES[sample])

    assert hint is None or isinstance(hint, str), f"{sample}: {hint!r}"


@pytest.mark.parametrize("codec", sorted(CJK_SAMPLES))
def test_a_cjk_hint_decodes_the_text_correctly(depcheck, codec):
    text = CJK_SAMPLES[codec]

    hint = _hint(depcheck, text.encode(codec))

    assert hint, f"no hint for {codec} text"
    assert text.encode(codec).decode(hint) == text, f"{codec} text was named {hint!r}"


def test_a_western_hint_is_a_codec_that_decodes_the_bytes(depcheck):
    raw = WESTERN.encode("cp1252")

    hint = _hint(depcheck, raw)

    assert hint, "no hint for Windows-1252 text"
    codecs.lookup(hint)
    raw.decode(hint)
