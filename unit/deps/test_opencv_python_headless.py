"""Dependency contract: opencv-python-headless (import name ``cv2``).

The backend never imports cv2 by name. Its consumer on Open WebUI's path is rapidocr, which
``retrieval/loaders/pdf.py`` runs on a PDF's images when ``PDF_EXTRACT_IMAGES`` is on: its text
detector finds boxes with ``findContours``, ``minAreaRect`` and ``boxPoints``, scores them with
``fillPoly`` and ``mean``, and crops each box upright with ``getPerspectiveTransform`` and
``warpPerspective`` before the recogniser resizes, pads (``copyMakeBorder``) and rotates it.

requirements.txt pins the headless wheel, while rapidocr requires ``opencv-python``; both install
the same ``cv2`` module, so which wheel and which major version ends up providing it is not
Open WebUI's contract and is not asserted here (two earlier version assertions broke on exactly
that). What is asserted is the behaviour: the codec, colour and resize primitives, and the
detection-and-crop pipeline rapidocr runs, on tiny in-memory arrays. OCR end to end is covered
by unit/deps/test_rapidocr.py and over HTTP by integration/deps/test_document_extraction.py.

cv2 is a C extension, so every check calls the function and asserts its result.

Discriminates: with ``minAreaRect`` answering a zero-size box (a pytest plugin patching cv2),
the pipeline test goes red.

Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "cv2"

# Core functions any image-processing consumer relies on.
USED_FUNCTIONS = [
    "imencode",
    "imdecode",
    "cvtColor",
    "resize",
    "imread",
    "imwrite",
]

# Flag/colour-code constants used by those functions.
USED_CONSTANTS = [
    "IMREAD_COLOR",
    "IMREAD_GRAYSCALE",
    "IMREAD_UNCHANGED",
    "COLOR_BGR2RGB",
    "COLOR_RGB2BGR",
    "COLOR_BGR2GRAY",
    "INTER_LINEAR",
    "INTER_AREA",
]


def _np(depcheck):
    np = depcheck.try_load("numpy")
    if np is None:
        pytest.skip("numpy not installed; cv2 behavioural tests need ndarray inputs")
    return np


def _bgr_image(np, h=8, w=12):
    """A small BGR image with a distinctive pure-red pixel (BGR = [0,0,255])
    so colour-channel order is observable after conversions."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # red channel in BGR ordering
    return img


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "cv2"


def test_core_functions_callable(depcheck):
    """Each core image function must exist and be callable (behavioural — cv2
    is a C-extension, so we assert callability, not hasattr on properties)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in USED_FUNCTIONS:
        fn = getattr(mod, name, None)
        assert callable(fn), f"cv2.{name} missing or not callable"


def test_constants_present_and_int(depcheck):
    """The flag/colour-code constants must exist and be integers (they are
    passed positionally into cvtColor / imdecode / resize)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in USED_CONSTANTS:
        val = getattr(mod, name, None)
        assert val is not None, f"cv2.{name} missing"
        assert isinstance(val, int), f"cv2.{name} is not an int (got {type(val)})"


# ---------------------------------------------------------------------------
# imencode / imdecode — in-memory codec round trip (no disk).
# ---------------------------------------------------------------------------


def test_png_encode_decode_round_trip_lossless(depcheck):
    """imencode('.png', img) -> bytes; imdecode(bytes, IMREAD_COLOR) -> img.
    PNG is lossless, so the decoded array must equal the original exactly."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np)
    ok, buf = mod.imencode(".png", img)
    assert ok is True
    assert buf is not None and len(buf) > 0
    decoded = mod.imdecode(buf, mod.IMREAD_COLOR)
    assert decoded.shape == img.shape
    assert np.array_equal(decoded, img), "PNG round trip was not lossless"


def test_jpg_encode_produces_bytes(depcheck):
    """imencode('.jpg', img) must succeed and yield a non-empty buffer (JPEG
    is lossy, so we don't assert pixel equality — just that the codec runs)."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np)
    ok, buf = mod.imencode(".jpg", img)
    assert ok is True
    assert len(buf) > 0


def test_imdecode_grayscale_drops_channels(depcheck):
    """Decoding with IMREAD_GRAYSCALE yields a single-channel (2-D) array."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np)
    ok, buf = mod.imencode(".png", img)
    assert ok
    gray = mod.imdecode(buf, mod.IMREAD_GRAYSCALE)
    assert gray.ndim == 2
    assert gray.shape == img.shape[:2]


def test_imdecode_garbage_returns_none(depcheck):
    """Decoding non-image bytes must return None (the documented cv2 failure
    mode), not raise — consumers branch on a None result."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    junk = np.frombuffer(b"not an image at all, just text bytes", dtype=np.uint8)
    result = mod.imdecode(junk, mod.IMREAD_COLOR)
    assert result is None


# ---------------------------------------------------------------------------
# cvtColor — colour-space conversions.
# ---------------------------------------------------------------------------


def test_cvtcolor_bgr_to_rgb_swaps_channels(depcheck):
    """BGR2RGB must swap the channel order: a BGR pure-red pixel [0,0,255]
    becomes RGB [255,0,0]. This is the exact gotcha (OpenCV is BGR, most of
    the world is RGB) consumers convert for."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np)
    rgb = mod.cvtColor(img, mod.COLOR_BGR2RGB)
    assert rgb.shape == img.shape
    assert rgb[0, 0].tolist() == [255, 0, 0]


def test_cvtcolor_bgr_to_gray_reduces_dims(depcheck):
    """BGR2GRAY collapses to a single channel (2-D array)."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np)
    gray = mod.cvtColor(img, mod.COLOR_BGR2GRAY)
    assert gray.ndim == 2
    assert gray.shape == img.shape[:2]


def test_cvtcolor_round_trip_bgr_rgb(depcheck):
    """BGR->RGB->BGR must restore the original array (channel swap is its own
    inverse)."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np)
    back = mod.cvtColor(mod.cvtColor(img, mod.COLOR_BGR2RGB), mod.COLOR_RGB2BGR)
    assert np.array_equal(back, img)


# ---------------------------------------------------------------------------
# resize — spatial scaling.
# ---------------------------------------------------------------------------


def test_resize_changes_dimensions(depcheck):
    """resize(img, (w, h)) returns an array with the requested width/height.
    Note cv2's dsize is (width, height) while the array shape is (h, w, c)."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np, h=8, w=12)
    out = mod.resize(img, (24, 16))  # (width=24, height=16)
    assert out.shape == (16, 24, 3)


def test_resize_with_interpolation_flag(depcheck):
    """resize honours an explicit interpolation flag (the kwarg path used when
    down/upsampling)."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    img = _bgr_image(np, h=16, w=16)
    out = mod.resize(img, (8, 8), interpolation=mod.INTER_AREA)
    assert out.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# The detection-and-crop pipeline rapidocr runs on a page image.
# ---------------------------------------------------------------------------


def test_a_text_box_is_found_scored_and_cropped_upright(depcheck):
    """rapidocr's detector and cropper, on a white box drawn into a black bitmap."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    bitmap = np.zeros((60, 100), dtype=np.uint8)
    bitmap[20:40, 10:90] = 255

    found = mod.findContours(bitmap, mod.RETR_LIST, mod.CHAIN_APPROX_SIMPLE)
    contours = found[-2]  # (contours, hierarchy), or (image, contours, hierarchy) on OpenCV 3
    assert len(contours) == 1

    box = mod.boxPoints(mod.minAreaRect(contours[0]))
    (left, top), (right, bottom) = box.min(axis=0), box.max(axis=0)
    assert box.shape == (4, 2)
    assert (round(right - left), round(bottom - top)) == (79, 19)

    mask = np.zeros_like(bitmap)
    mod.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    assert mod.mean(bitmap, mask)[0] > 250  # the box scores as all text

    source = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    target = np.float32([[0, 0], [40, 0], [40, 10], [0, 10]])
    transform = mod.getPerspectiveTransform(source, target)
    crop = mod.warpPerspective(
        bitmap, transform, (40, 10), borderMode=mod.BORDER_REPLICATE, flags=mod.INTER_CUBIC
    )
    assert crop.shape == (10, 40)
    assert crop.min() > 200  # only the white box, nothing of the black page


def test_a_crop_is_padded_and_rotated(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    crop = np.full((10, 40, 3), 255, dtype=np.uint8)

    padded = mod.copyMakeBorder(crop, 0, 0, 0, 8, mod.BORDER_CONSTANT, value=0)
    turned = mod.rotate(crop, mod.ROTATE_180)
    upright = mod.rotate(crop, mod.ROTATE_90_CLOCKWISE)

    assert padded.shape == (10, 48, 3) and padded[:, 40:].max() == 0
    assert turned.shape == crop.shape
    assert upright.shape == (40, 10, 3)
