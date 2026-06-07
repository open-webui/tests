"""Dependency contract: opencv-python-headless (import name ``cv2``).

cv2 is not imported anywhere in the Open WebUI backend by name; it is a
transitive dependency (pinned ``opencv-python-headless==4.13.0.92`` in
requirements.txt, NOT in requirements-min.txt) pulled in by the ML / image
stack. The *headless* build is deliberate: it carries no GUI (highgui)
backend, which matters on a server. Image-bearing model code (and some
document/image preprocessing) relies on cv2's encode/decode/colour-convert/
resize primitives; a bump that broke them, or that swapped in the non-
headless wheel (dragging in GUI libs that fail to load on a headless host),
would break image handling.

Because nothing in the backend names a cv2 symbol of its own, this module
pins cv2's *core image-processing surface* and exercises each operation
OFFLINE against tiny in-memory NumPy arrays — no disk reads, no GUI, no
network, no model download:
  - ``imencode`` / ``imdecode`` round-trip a PNG through a byte buffer
    (lossless), proving the in-memory codec path works;
  - ``cvtColor`` performs the BGR<->RGB and BGR->GRAY conversions;
  - ``resize`` changes spatial dimensions;
  - the colour/flag constants the API uses are present and integer-valued.

cv2 is a C-extension module, so checks on it are behavioural (call the
function and assert the result), never ``hasattr`` probing of executing
properties.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "cv2"
DIST_NAME = "opencv-python-headless"

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


def test_version_reported(depcheck):
    """The *distribution* is opencv-python-headless even though the import is
    cv2; the version must resolve under that dist name (so a swap to the GUI
    wheel `opencv-python` is noticed)."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_cv2_version_attr(depcheck):
    """cv2.__version__ must report the 4.x line the backend pins."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    major = int(mod.__version__.split(".")[0])
    assert major == 4, f"expected OpenCV 4.x, got {mod.__version__}"


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
# Headless build sanity — the GUI surface must be absent/no-op so importing
# cv2 on a server never pulls a display backend.
# ---------------------------------------------------------------------------


def test_headless_distribution_is_installed(depcheck):
    """The backend pins opencv-python-headless (no GUI / system libGL deps). Pin
    that the headless distribution is installed so a regression dropping it is
    caught. (Older headless builds made imshow raise; 4.x no longer guarantees
    that, so we assert the distribution instead of imshow's behaviour, since the
    backend never calls any highgui function. A transitive dep such as rapidocr
    may ALSO pull the GUI `opencv-python` wheel; both can coexist and cv2 still
    works, so absence of the GUI wheel is not asserted.)"""
    depcheck.load(IMPORT_NAME)  # skip cleanly if cv2 is not importable
    assert depcheck.dist_version("opencv-python-headless") is not None, (
        "opencv-python-headless (the pinned OpenCV distribution) is not installed"
    )
