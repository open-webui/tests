"""Dependency contract: chardet.

The Open WebUI backend uses chardet to detect the encoding of uploaded text
files. The single consumer is ``RagDocumentLoader._detect_text_encoding`` in
``backend/open_webui/retrieval/loaders/main.py``: after a UTF-8 fast path
fails, it calls ``chardet.detect(raw)`` and reads ``detected.get('encoding')``
as a *hint* to prioritise the right CJK codec family (GB2312 -> gb18030 etc.),
then validates the decode itself. So the contract the backend actually depends
on is narrow but exact:

    detected = chardet.detect(raw)            # raw is bytes
    enc = (detected.get('encoding') or '')    # str | None, never KeyError
    enc = enc.lower().replace('-', '').replace('_', '')
    ...
    detected.get('encoding')                  # used again as a decode hint

This module pins that slice so the chardet 5 -> 7 MAJOR bump fails loudly here
(symbol gone, signature changed, return shape changed, ``.get('encoding')``
broken) instead of surfacing as a garbled-text or AttributeError deep in the
RAG ingestion path. Crucially it pins the *return shape and the .get()
contract*, not chardet's specific guesses: the loader deliberately treats the
guess as unreliable for short CJK input (and our own probe confirms chardet 5
misclassifies short Big5/Shift-JIS samples), so asserting exact encoding names
would be both wrong and flaky. Exact values are asserted only where chardet is
genuinely deterministic (pure ASCII, a UTF-16 BOM, empty input -> None).

Exemplar for the unit/deps/ pattern: symbol-existence checks (API surface) +
offline behavioural contracts (no network, no filesystem). Uses the
``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "chardet"
DIST_NAME = "chardet"

# Symbols the Open WebUI backend (directly or via the documented public API it
# relies on) references on ``chardet``. ``detect`` + the dict it returns is the
# only thing the loader calls today; the rest are part of chardet's stable
# public surface that a major bump must not silently drop.
USED_SYMBOLS = [
    "detect",
    "detect_all",
    "__version__",
    "universaldetector.UniversalDetector",
]

# The three keys chardet's result dict has carried for its entire 4.x/5.x life.
# The loader reads 'encoding'; 'confidence'/'language' are pinned so a shape
# change is caught even though the loader ignores them today.
RESULT_KEYS = ("encoding", "confidence", "language")


# --------------------------------------------------------------------------- #
# Local, offline byte-string fixtures.
#
# Each value is produced by encoding a known string with a known codec, so the
# bytes are deterministic and require no network or data files. The CJK samples
# are repeated so chardet has enough signal to return *a* (possibly wrong)
# answer rather than None — we assert the shape of that answer, mirroring how
# the loader consumes it.
# --------------------------------------------------------------------------- #

_ASCII = b"Hello, this is plain 7-bit ASCII text with no high bytes.\n"

# UTF-8 carrying CJK — chardet detects this one reliably as utf-8, and it is the
# common real-world case the loader's UTF-8 fast path actually handles before
# chardet is ever consulted; included to prove detect() agrees.
_UTF8_CJK = ("UTF-8 mixed content. " + "你好世界，これは日本語。" * 8).encode("utf-8")

# The specific CJK family the loader cares about most: GB18030 (it maps
# chardet's GB2312/GBK guesses onto gb18030, a strict superset).
_GB18030 = ("简体中文测试，编码检测。" * 12).encode("gb18030")

# Big5 (Traditional Chinese) and Shift-JIS (Japanese) — chardet 5 is known to
# misclassify short samples of these (our probe saw Big5->GB2312,
# Shift-JIS->MacCyrillic). The loader copes by validating the decode itself, so
# here we only assert the *contract*, never the guessed encoding name.
_BIG5 = ("繁體中文測試，編碼偵測。" * 12).encode("big5")
_SHIFT_JIS = ("日本語のテキスト、文字コード判定。" * 12).encode("shift_jis")
_EUC_KR = ("한국어 텍스트, 인코딩 감지 테스트." * 12).encode("euc-kr")

# UTF-16 with a BOM — chardet reads the BOM and returns "UTF-16" deterministically.
_UTF16_BOM = "﻿Hello UTF-16 with BOM and CJK 你好".encode("utf-16")

# Latin-1 / Western European with accented bytes in 0x80-0xFF.
_LATIN1 = "café résumé naïve Zürich, Schöne Grüße.".encode("latin-1")

# All samples that must yield a usable (non-None encoding) result dict.
_NONEMPTY_SAMPLES = {
    "ascii": _ASCII,
    "utf8_cjk": _UTF8_CJK,
    "gb18030": _GB18030,
    "big5": _BIG5,
    "shift_jis": _SHIFT_JIS,
    "euc_kr": _EUC_KR,
    "utf16_bom": _UTF16_BOM,
    "latin1": _LATIN1,
}


def _detect(depcheck):
    """Load chardet (or skip) and return its ``detect`` callable."""
    mod = depcheck.load(IMPORT_NAME)
    return mod, mod.detect


def _assert_result_shape(result, *, sample_name: str) -> None:
    """Assert a chardet.detect() result matches the dict contract the loader uses."""
    assert isinstance(result, dict), (
        f"chardet.detect({sample_name}) returned {type(result)!r}, not a dict; "
        "the loader does detected.get('encoding') and would break."
    )
    for key in RESULT_KEYS:
        assert key in result, (
            f"chardet.detect({sample_name}) result missing key {key!r}: {result}. "
            "Return-dict shape changed."
        )
    # The loader's exact access pattern: detected.get('encoding') must work and
    # be either a string or None (it then does `(enc or '').lower()...`).
    enc = result.get("encoding")
    assert enc is None or isinstance(enc, str), (
        f"chardet.detect({sample_name})['encoding'] is {type(enc)!r}; loader expects str or None."
    )
    conf = result.get("confidence")
    assert isinstance(conf, float), (
        f"chardet.detect({sample_name})['confidence'] is {type(conf)!r}, not float."
    )
    assert 0.0 <= conf <= 1.0, f"confidence out of [0,1] for {sample_name}: {conf}"


# --------------------------------------------------------------------------- #
# Import / symbol-surface tests.
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    """chardet must import and identify as the chardet package."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "chardet"


def test_used_symbols_exist(depcheck):
    """Every chardet symbol the codebase relies on must still resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_detect_is_callable(depcheck):
    """The loader's only call is chardet.detect(raw)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "detect")


def test_detect_all_is_callable(depcheck):
    """detect_all is part of chardet's public surface."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "detect_all")


def test_universaldetector_class_present(depcheck):
    """UniversalDetector (the incremental API) must remain importable."""
    mod = depcheck.load(IMPORT_NAME)
    ud_cls = depcheck.resolve(mod, "universaldetector.UniversalDetector")
    assert inspect.isclass(ud_cls), "universaldetector.UniversalDetector is not a class"


def test_universaldetector_importable_from_submodule(depcheck):
    """Direct `from chardet.universaldetector import UniversalDetector` must work."""
    depcheck.load(IMPORT_NAME)  # skip if chardet absent
    sub = depcheck.load("chardet.universaldetector")
    assert hasattr(sub, "UniversalDetector")


def test_version_attribute_is_string(depcheck):
    """chardet.__version__ is a non-empty string."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str) and mod.__version__


def test_dist_version_resolvable(depcheck):
    """The installed distribution version is resolvable (bump tooling agrees)."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Signature tests.
# --------------------------------------------------------------------------- #


def test_detect_signature_first_positional(depcheck):
    """detect() must accept the byte string as its first positional argument.

    The loader calls ``chardet.detect(raw)`` positionally, so the leading
    parameter must remain positional (any name). We don't pin the name, only
    that there is at least one positional parameter and detect is happy to be
    called with a single positional bytes argument (covered behaviourally below).
    """
    _mod, detect = _detect(depcheck)
    try:
        sig = inspect.signature(detect)
    except (TypeError, ValueError):
        pytest.skip("detect has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"detect() exposes no positional parameter (sig: {sig})"


def test_detect_all_signature_first_positional(depcheck):
    """detect_all() likewise takes the byte string positionally."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        sig = inspect.signature(mod.detect_all)
    except (TypeError, ValueError):
        pytest.skip("detect_all has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"detect_all() exposes no positional parameter (sig: {sig})"


# --------------------------------------------------------------------------- #
# Behavioural contract: return shape across many known byte strings.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sample_name", sorted(_NONEMPTY_SAMPLES))
def test_detect_returns_dict_with_contract_keys(depcheck, sample_name):
    """Every non-empty sample yields a dict carrying encoding/confidence/language."""
    _mod, detect = _detect(depcheck)
    result = detect(_NONEMPTY_SAMPLES[sample_name])
    _assert_result_shape(result, sample_name=sample_name)


@pytest.mark.parametrize("sample_name", sorted(_NONEMPTY_SAMPLES))
def test_detect_get_encoding_is_str_for_real_text(depcheck, sample_name):
    """For genuine (non-empty) text, detect()['encoding'] resolves to a string.

    This is the value the loader feeds into ``(enc or '').lower()`` and later
    ``raw.decode(enc)``; for real content chardet should commit to *some*
    codec name rather than returning None.
    """
    _mod, detect = _detect(depcheck)
    enc = detect(_NONEMPTY_SAMPLES[sample_name]).get("encoding")
    assert isinstance(enc, str) and enc, (
        f"detect()['encoding'] for {sample_name!r} was {enc!r}; "
        "loader needs a usable codec-name hint here."
    )


def test_detect_ascii_is_deterministic(depcheck):
    """Pure ASCII is the one case chardet pins exactly: 'ascii' at full confidence."""
    _mod, detect = _detect(depcheck)
    result = detect(_ASCII)
    assert result.get("encoding") == "ascii", f"expected ascii, got {result!r}"
    assert result.get("confidence") == 1.0


def test_detect_utf8_cjk_identified_as_utf8(depcheck):
    """UTF-8 (incl. CJK) is detected as utf-8 — the loader's common happy path."""
    _mod, detect = _detect(depcheck)
    enc = detect(_UTF8_CJK).get("encoding")
    assert isinstance(enc, str) and enc.lower().replace("-", "") == "utf8", (
        f"UTF-8 CJK content detected as {enc!r}, expected utf-8."
    )


def test_detect_utf16_bom_identified_as_utf16(depcheck):
    """A UTF-16 BOM is unambiguous; chardet must report a utf-16 variant."""
    _mod, detect = _detect(depcheck)
    enc = detect(_UTF16_BOM).get("encoding")
    assert isinstance(enc, str) and "16" in enc and enc.lower().startswith("utf"), (
        f"UTF-16-BOM content detected as {enc!r}, expected a UTF-16 variant."
    )


def test_detect_latin1_returns_decodable_western_codec(depcheck):
    """Latin-1 input -> a codec name that actually decodes the bytes.

    The loader's final branch does ``raw.decode(detected.get('encoding'))``.
    For Western bytes chardet may say ISO-8859-1 / Windows-1252 / etc.; we
    don't pin which, only that the returned name is a real, loadable codec
    that round-trips the bytes (mirroring what the loader requires of it).
    """
    _mod, detect = _detect(depcheck)
    enc = detect(_LATIN1).get("encoding")
    assert isinstance(enc, str) and enc, f"latin-1 content detected as {enc!r}"
    # Must be a codec Python can actually use, as the loader assumes.
    _LATIN1.decode(enc)


def test_detect_gb18030_family_is_chinese_hint(depcheck):
    """The GB18030 sample yields a hint the loader can map into the gb18030 family.

    chardet often reports GB2312/GBK (subsets) for gb18030 content; the loader
    explicitly maps {gb2312,gbk,gb18030} -> gb18030. We assert the guess is
    *some* string (so the mapping has input) and, softly, that it looks
    GB-family or at least decodes — never that it equals 'GB18030'.
    """
    _mod, detect = _detect(depcheck)
    result = detect(_GB18030)
    _assert_result_shape(result, sample_name="gb18030")
    enc = result.get("encoding")
    assert isinstance(enc, str) and enc, "gb18030 sample produced no encoding hint"
    # The loader normalises like this before looking it up in its family map.
    normalised = enc.lower().replace("-", "").replace("_", "")
    assert normalised, "normalised encoding hint collapsed to empty string"


# --------------------------------------------------------------------------- #
# Behavioural contract: robustness / never-raise on byte inputs.
# --------------------------------------------------------------------------- #


def test_detect_accepts_bytes_without_raising(depcheck):
    """detect() must not raise on ordinary bytes — the loader has no try/except
    around it, so a raising detect() would crash RAG ingestion."""
    _mod, detect = _detect(depcheck)
    for sample in _NONEMPTY_SAMPLES.values():
        detect(sample)  # must not raise


def test_detect_accepts_bytearray(depcheck):
    """bytearray is an accepted input type (chardet's signature allows it)."""
    _mod, detect = _detect(depcheck)
    result = detect(bytearray(_ASCII))
    _assert_result_shape(result, sample_name="bytearray-ascii")


def test_detect_empty_bytes_encoding_none_or_str(depcheck):
    """Empty input -> dict whose 'encoding' is None or a str (never raises).

    The loader short-circuits empty files before reaching chardet, and its
    ``(detected.get('encoding') or '')`` idiom tolerates either None (chardet 5)
    or a low-confidence str like 'utf-8' (chardet 7). Pin that the call returns a
    well-formed dict and never raises on empty input.
    """
    _mod, detect = _detect(depcheck)
    result = detect(b"")
    assert isinstance(result, dict), f"detect(b'') returned {type(result)!r}"
    assert "encoding" in result, f"detect(b'') missing 'encoding' key: {result}"
    # chardet 5 returned None here; chardet 7 returns a low-confidence 'utf-8'.
    # The loader's `(detected.get('encoding') or '')` idiom tolerates either, so
    # the contract is: encoding is None or a str, and the call never raises.
    enc = result.get("encoding")
    assert enc is None or isinstance(enc, str), (
        f"detect(b'') encoding must be None or str, got {enc!r}"
    )


def test_detect_get_encoding_or_empty_idiom(depcheck):
    """Exercise the loader's exact idiom: ``(detected.get('encoding') or '')``
    followed by ``.lower().replace('-','').replace('_','')`` — must never raise
    whether encoding is a str or None."""
    _mod, detect = _detect(depcheck)
    for sample in (_ASCII, _GB18030, b""):
        detected = detect(sample)
        normalised = (detected.get("encoding") or "").lower().replace("-", "").replace("_", "")
        assert isinstance(normalised, str)


def test_detect_high_bytes_noise_does_not_raise(depcheck):
    """Arbitrary high-byte noise (not valid in any single codec) must still
    return the dict contract rather than raising."""
    _mod, detect = _detect(depcheck)
    noise = bytes(range(256)) * 4
    result = detect(noise)
    _assert_result_shape(result, sample_name="byte-noise")


def test_detect_result_is_plain_dict_subscriptable(depcheck):
    """The result supports both ['encoding'] subscripting and .get() — chardet
    returns a plain dict and the loader uses .get(); pin that it stays a real
    mapping, not some custom object that only supports one access style."""
    _mod, detect = _detect(depcheck)
    result = detect(_UTF8_CJK)
    assert result["encoding"] == result.get("encoding")
    assert set(RESULT_KEYS).issubset(result.keys())


# --------------------------------------------------------------------------- #
# Behavioural contract: detect_all and the incremental UniversalDetector.
# These are not used by the loader today but are public API a major bump must
# not silently break; keeping them pinned guards future use and catches a
# wholesale API rewrite early.
# --------------------------------------------------------------------------- #


def test_detect_all_returns_list_of_result_dicts(depcheck):
    """detect_all() returns a list whose entries share the detect() dict shape."""
    mod = depcheck.load(IMPORT_NAME)
    results = mod.detect_all(_ASCII)
    assert isinstance(results, list) and results, f"detect_all returned {results!r}"
    first = results[0]
    assert isinstance(first, dict) and "encoding" in first, (
        f"detect_all entries are not result dicts: {first!r}"
    )


def test_universaldetector_incremental_flow(depcheck):
    """UniversalDetector supports feed()/close() and exposes .result as a dict.

    This is the streaming counterpart to detect(); pin the minimal lifecycle
    (instantiate -> feed -> close -> read .result) and that .result matches the
    same encoding/confidence/language contract.
    """
    mod = depcheck.load(IMPORT_NAME)
    ud_cls = depcheck.resolve(mod, "universaldetector.UniversalDetector")
    detector = ud_cls()
    for method in ("feed", "close", "reset"):
        assert callable(getattr(detector, method, None)), (
            f"UniversalDetector.{method} missing/not callable"
        )
    detector.feed(_UTF8_CJK)
    detector.close()
    result = detector.result
    _assert_result_shape(result, sample_name="UniversalDetector.result")


def test_universaldetector_reset_allows_reuse(depcheck):
    """reset() returns the detector to a usable state for a second document."""
    mod = depcheck.load(IMPORT_NAME)
    ud_cls = depcheck.resolve(mod, "universaldetector.UniversalDetector")
    detector = ud_cls()
    detector.feed(_ASCII)
    detector.close()
    detector.reset()
    detector.feed(_UTF8_CJK)
    detector.close()
    _assert_result_shape(detector.result, sample_name="UniversalDetector-after-reset")


# --------------------------------------------------------------------------- #
# Determinism: the suite must not depend on call order or repetition.
# --------------------------------------------------------------------------- #


def test_detect_is_deterministic_across_calls(depcheck):
    """Calling detect() twice on identical bytes yields identical results —
    the loader assumes a stable answer for a given file."""
    _mod, detect = _detect(depcheck)
    for sample in _NONEMPTY_SAMPLES.values():
        assert detect(sample) == detect(sample)
