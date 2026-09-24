"""Dependency contract: rapidocr (import name ``rapidocr``).

``retrieval/loaders/pdf.py`` reads the images of an uploaded PDF when ``PDF_EXTRACT_IMAGES`` is
on::

    from rapidocr import RapidOCR
    self.ocr = RapidOCR()
    result = self.ocr(pixels)            # an RGB numpy array
    if result and result.txts:
        texts.append('\\n'.join(result.txts).strip())

So the contract is: ``RapidOCR()`` builds with no arguments from the models inside the wheel,
without reaching the network; called on an RGB array it returns an object whose ``txts`` holds
the recognised lines as strings; an image without text gives a falsy result or no ``txts``.
rapidocr replaced rapidocr-onnxruntime, which the backend no longer depends on. The upload path
is covered over HTTP in integration/deps/test_document_extraction.py.

Discriminates: with the wheel's models removed (so ``RapidOCR()`` has to download them) or
``RapidOCR.__call__`` answering with no text, the matching tests go red.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

DEAD_PROXY = "http://127.0.0.1:9"
WORD = "LIGHTHOUSE"


@pytest.fixture(scope="module")
def ocr_engine(depcheck):
    """The engine as the loader builds it, with every download refused."""
    rapidocr = depcheck.load("rapidocr")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            patch.setenv(name, DEAD_PROXY)
        for name in ("NO_PROXY", "no_proxy"):
            patch.delenv(name, raising=False)
        return rapidocr.RapidOCR()


def _rgb_array(depcheck, text: str = ""):
    """What the loader passes: `np.array(image.convert('RGB'))`."""
    numpy = depcheck.load("numpy")
    image_module = depcheck.load("PIL.Image")
    draw_module = depcheck.load("PIL.ImageDraw")
    font_module = depcheck.load("PIL.ImageFont")
    image = image_module.new("RGB", (900, 200), "white")
    if text:
        font = font_module.load_default(size=64)
        draw_module.Draw(image).text((30, 60), text, fill="black", font=font)
    return numpy.array(image.convert("RGB"))


def test_the_engine_is_exported_at_the_top_level(depcheck):
    depcheck.assert_symbols(depcheck.load("rapidocr"), ["RapidOCR"])


def test_the_engine_reads_the_text_of_an_rgb_array(ocr_engine, depcheck):
    result = ocr_engine(_rgb_array(depcheck, WORD))

    assert result and result.txts, f"no text recognised: {result!r}"
    assert all(isinstance(line, str) for line in result.txts)
    assert WORD in "\n".join(result.txts)


def test_an_image_without_text_yields_no_lines(ocr_engine, depcheck):
    result = ocr_engine(_rgb_array(depcheck))

    assert not (result and result.txts)
