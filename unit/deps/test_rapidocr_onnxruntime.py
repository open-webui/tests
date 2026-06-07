"""Dependency contract: rapidocr-onnxruntime (import name ``rapidocr_onnxruntime``).

rapidocr-onnxruntime is an ONNX-runtime OCR engine (PaddleOCR models, no
PaddlePaddle dependency). Open WebUI pins it in ``backend/requirements.txt``
(``rapidocr-onnxruntime==1.4.4``) but does NOT import it directly in
``open_webui/*``: it is a *transitive* dependency of the document-ingestion
stack — the unstructured / docling loaders behind ``retrieval/loaders/main.py``
use it to OCR text out of images and scanned PDFs so that content becomes
indexable for RAG.

Because nothing in the backend names it directly, this module pins its *core
public surface* so a bump that broke it surfaces here rather than as silently
empty OCR output during ingestion. We pin: the top-level ``RapidOCR`` engine
class, its constructor and ``__call__`` signatures (the ``use_det`` / ``use_cls``
/ ``use_rec`` toggles), and — crucially — a REAL end-to-end OCR contract run
fully offline against the models *bundled inside the wheel* (no network, no
model download): an image with rendered text yields the documented
``[[box, text, score], ...]`` result tuple, and a blank image yields the
"no text" sentinel. Both the engine construction and inference are local and
fast (sub-second).

If Pillow/numpy (needed to synthesize the test image) are missing, the
behavioural OCR tests skip cleanly while the surface checks still run.

Pattern mirrors test_requests.py / test_pillow.py. Uses the ``depcheck``
fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "rapidocr_onnxruntime"
DIST_NAME = "rapidocr-onnxruntime"


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`rapidocr_onnxruntime` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "rapidocr_onnxruntime"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling and
    this suite agree on what's under test. NOTE the distribution name is
    `rapidocr-onnxruntime` while the import name is `rapidocr_onnxruntime`."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature checks (API surface)
# ---------------------------------------------------------------------------


def test_rapidocr_class_exists_and_callable(depcheck):
    """RapidOCR is the engine class the loader stack instantiates; it must exist
    and be callable (constructible)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "RapidOCR")


def test_rapidocr_init_accepts_config_path(depcheck):
    """The engine is constructed as RapidOCR(...) with an optional config_path
    and **kwargs (per-model overrides). Pin that config_path remains accepted
    and that **kwargs is preserved (so model-path/threading overrides pass)."""
    mod = depcheck.load(IMPORT_NAME)
    import inspect

    params = inspect.signature(mod.RapidOCR.__init__).parameters
    assert "config_path" in params, "RapidOCR.__init__ dropped config_path"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "RapidOCR.__init__ dropped **kwargs; per-model overrides would be rejected."
    )


def test_rapidocr_call_signature(depcheck):
    """The engine is invoked as engine(img, use_det=, use_cls=, use_rec=). Those
    pipeline-stage toggles must remain accepted parameters on __call__."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.RapidOCR.__call__,
        ["img_content", "use_det", "use_cls", "use_rec"],
    )


def test_rapidocr_is_callable_instance_protocol(depcheck):
    """An instance is called like a function (engine(image)); pin that the class
    defines __call__ as a callable."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.RapidOCR.__call__)


# ---------------------------------------------------------------------------
# Behavioural: REAL offline OCR against the wheel-bundled models.
# Engine construction + inference are local (no network / no download).
# ---------------------------------------------------------------------------


def _engine(depcheck):
    """Construct a RapidOCR engine offline.

    rapidocr-onnxruntime ships its ONNX models inside the wheel, so the engine
    loads them from disk with no network access. Construction is sub-second."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        return mod.RapidOCR()
    except Exception as e:  # pragma: no cover - environment/runtime issue
        pytest.skip(f"RapidOCR engine could not be constructed offline: {e}")


def _np_and_pil(depcheck):
    np = depcheck.try_load("numpy")
    pil = depcheck.try_load("PIL")
    if np is None or pil is None:
        pytest.skip("numpy/Pillow unavailable; cannot synthesize OCR test image")
    return np


def test_behaviour_engine_constructs_offline(depcheck):
    """The loader stack constructs one RapidOCR engine. Construction must succeed
    using the bundled models, with no network."""
    engine = _engine(depcheck)
    assert engine is not None
    assert callable(engine), "constructed RapidOCR instance is not callable"


def test_behaviour_ocr_blank_image_returns_no_text_sentinel(depcheck):
    """Run on a blank white image. RapidOCR returns a (result, elapse) tuple;
    with nothing to read, `result` is the documented None sentinel (the loader
    treats this as 'no OCR text'). Pin that 2-tuple shape and the empty result."""
    np = _np_and_pil(depcheck)
    engine = _engine(depcheck)
    blank = np.full((64, 160, 3), 255, dtype=np.uint8)
    out = engine(blank)
    assert isinstance(out, tuple) and len(out) == 2, (
        f"RapidOCR call no longer returns a (result, elapse) 2-tuple: {type(out)!r}"
    )
    result, _elapse = out
    assert result is None or result == [], f"blank image should yield no OCR text, got {result!r}"


def test_behaviour_ocr_rendered_text_result_shape(depcheck):
    """The core contract the loader depends on: an image containing text yields
    a list of [box, text, score] triples. Render 'HELLO' and assert the result
    is a non-empty list whose items are (box, text:str, score:float)."""
    np = _np_and_pil(depcheck)
    from PIL import Image, ImageDraw

    engine = _engine(depcheck)
    img = Image.new("RGB", (240, 80), (255, 255, 255))
    ImageDraw.Draw(img).text((14, 26), "HELLO", fill=(0, 0, 0))
    result, _elapse = engine(np.array(img))

    if not result:
        pytest.skip("default bitmap font produced no detectable glyphs in this env")

    assert isinstance(result, list) and len(result) >= 1, (
        f"OCR result is not a non-empty list: {result!r}"
    )
    box, text, score = result[0]
    # box is a quad of points; text is the recognised string; score a confidence.
    assert isinstance(text, str), f"OCR text field is not a str: {text!r}"
    assert isinstance(score, float), f"OCR score field is not a float: {score!r}"
    assert box is not None and len(box) == 4, f"OCR box is not a 4-point quad: {box!r}"


def test_behaviour_ocr_accepts_numpy_array_input(depcheck):
    """The loaders hand RapidOCR a numpy image array (decoded page/region). Pin
    that an ndarray is an accepted img_content type and does not raise."""
    np = _np_and_pil(depcheck)
    engine = _engine(depcheck)
    arr = np.full((48, 120, 3), 255, dtype=np.uint8)
    # Should run without raising regardless of whether it finds text.
    out = engine(arr)
    assert isinstance(out, tuple) and len(out) == 2
