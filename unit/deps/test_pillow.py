"""Dependency contract: Pillow (import name ``PIL``).

Pillow is the image-decoding/transform engine in the Open WebUI backend's
dependency tree. The backend does not ``from PIL import`` anything in its
own first-party code, but Pillow is a pinned, non-optional dependency
(``pyproject.toml`` / ``backend/requirements.txt``: ``pillow==12.2.0``)
that the document-ingestion and image-handling stack pulls in: it is the
image loader behind retrieval/document loaders (``unstructured`` and the
OCR/RapidOCR pre-processing path), image-format detection, and any
thumbnailing/conversion done on uploaded or fetched images. Every one of
those paths feeds Pillow *attacker-influenced bytes* (an uploaded file, a
fetched remote image, a document's embedded image), which is exactly why
image parsers are a recurring CVE area and why a Pillow bump
(12.1.x -> 12.2.0 here) is worth a contract gate.

This module pins the slice of the Pillow API that any consumer in this
stack relies on, so a bump that removed/renamed a symbol, changed a save
keyword, dropped a codec, or weakened the decompression-bomb guard fails
loudly here instead of as a runtime ``AttributeError`` / silent behaviour
change deep in an ingestion path. The pattern mirrors ``test_requests.py``
and ``test_redis.py``: symbol-existence and signature checks for the API
surface, plus fully offline behavioural contracts built on *in-memory*
images (``Image.new`` + ``io.BytesIO`` round-trips — no disk fixtures, no
network).

Uses the ``depcheck`` fixture from ``unit/deps/conftest.py``.
"""

from __future__ import annotations

import inspect
import io

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "PIL"
DIST_NAME = "pillow"


# ---------------------------------------------------------------------------
# Symbol inventory — the Pillow API surface this stack depends on. These are
# the names a consumer (document loaders, OCR pre-processing, image handling)
# resolves on PIL and its submodules. Kept as dotted paths so the depcheck
# resolver imports submodules as needed.
# ---------------------------------------------------------------------------

# Top-level package symbols.
TOP_LEVEL_SYMBOLS = [
    "Image",  # PIL.Image module (the core entry point)
    "ImageOps",  # exif_transpose / fit / contain on loaded images
    "ImageFile",  # ImageFile.ImageFile (open() return type) + tuning flags
    "ExifTags",  # tag-name <-> id mapping for EXIF reads
    "UnidentifiedImageError",  # raised on undecodable/garbage bytes
    "features",  # codec availability probing (check('jpg'/'webp'/'zlib'))
    "__version__",  # version string (bump tooling agreement)
]

# Symbols on PIL.Image the codebase / its consumers rely on.
IMAGE_MODULE_SYMBOLS = [
    "open",  # decode bytes/file -> ImageFile
    "new",  # construct in-memory image
    "frombytes",  # raw-pixel construction
    "merge",  # band recombination
    "Image",  # the image class itself (isinstance checks / type hints)
    "Resampling",  # resize/thumbnail filter enum (LANCZOS, BICUBIC, ...)
    "Transpose",  # transpose/rotate constant enum
    "MAX_IMAGE_PIXELS",  # decompression-bomb pixel ceiling
    "DecompressionBombError",  # raised when an image exceeds 2x the ceiling
    "DecompressionBombWarning",  # warned when an image exceeds the ceiling
    "UnidentifiedImageError",  # re-exported on the Image module too
    "registered_extensions",  # extension -> format map (format detection)
]

# Methods invoked on a decoded/constructed image instance.
IMAGE_INSTANCE_METHODS = [
    "convert",  # mode normalisation (-> RGB / RGBA / L) before downstream use
    "resize",  # explicit resize
    "thumbnail",  # in-place aspect-preserving downscale
    "save",  # re-encode to a BytesIO/file in a target format
    "crop",  # region extraction
    "copy",  # defensive copy
    "rotate",  # rotation
    "transpose",  # orientation normalisation
    "tobytes",  # raw pixel export
    "getexif",  # EXIF metadata read (orientation handling)
    "getdata",  # pixel access
    "load",  # force full decode (where lazy decode must be materialised)
    "paste",  # compositing
]

# Resampling filter members used when resizing/thumbnailing.
RESAMPLING_MEMBERS = ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS", "BOX", "HAMMING"]

# ImageOps helpers consumers use for orientation/size normalisation.
IMAGEOPS_SYMBOLS = ["exif_transpose", "fit", "contain", "grayscale", "pad"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """``import PIL`` must succeed (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "PIL"


def test_image_submodule_imports(depcheck):
    """Consumers do ``from PIL import Image``; it must be a real importable
    submodule, not just a lazy attribute."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    assert image.__name__ == "PIL.Image"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_version_attribute_matches_distribution(depcheck):
    """``PIL.__version__`` should agree with the installed distribution
    metadata (guards against a shadowed/partial install)."""
    mod = depcheck.load(IMPORT_NAME)
    dist = depcheck.dist_version(DIST_NAME)
    if dist is None:
        pytest.skip("pillow distribution metadata not resolvable")
    assert mod.__version__ == dist


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level ``PIL.*`` symbol this stack resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_image_module_symbols_exist(depcheck):
    """Every ``PIL.Image.*`` symbol relied on must exist."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    depcheck.assert_symbols(image, IMAGE_MODULE_SYMBOLS)


def test_imageops_symbols_exist(depcheck):
    """ImageOps helpers used for orientation/size normalisation must exist."""
    depcheck.load(IMPORT_NAME)
    imageops = depcheck.load("PIL.ImageOps")
    depcheck.assert_symbols(imageops, IMAGEOPS_SYMBOLS)


def test_image_instance_methods_exist(depcheck):
    """Every method invoked on a decoded/constructed image must exist on the
    Image class (checked on the class via dir(), so no descriptor executes)."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    names = set(dir(image.Image))
    missing = [m for m in IMAGE_INSTANCE_METHODS if m not in names]
    assert not missing, f"PIL.Image.Image missing method(s) the stack uses: {missing}"


def test_core_callables(depcheck):
    """The decode/construct entry points must be callable."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    for name in ("open", "new", "frombytes", "merge"):
        depcheck.assert_callable(image, name)


def test_resampling_members_exist(depcheck):
    """resize()/thumbnail() pass Image.Resampling.* filters; the enum members
    used must all exist (NEAREST..LANCZOS)."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    resampling = image.Resampling
    missing = [m for m in RESAMPLING_MEMBERS if not hasattr(resampling, m)]
    assert not missing, f"Image.Resampling missing member(s): {missing}"


def test_unidentified_image_error_is_oserror(depcheck):
    """Decoders raise UnidentifiedImageError on garbage; consumers catch it
    (often via ``except OSError``/``except Exception``). It must remain an
    OSError subclass so those broad handlers keep working."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.UnidentifiedImageError, OSError)
    # Re-exported on the Image module too, and the same class.
    image = depcheck.load("PIL.Image")
    assert image.UnidentifiedImageError is mod.UnidentifiedImageError


def test_decompression_bomb_error_is_exception(depcheck):
    """The decompression-bomb guard raises Image.DecompressionBombError; it
    must be a concrete Exception subclass distinct from the (warning) sibling
    so a hardened consumer can catch the hard-fail specifically."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    assert inspect.isclass(image.DecompressionBombError)
    assert issubclass(image.DecompressionBombError, Exception)
    assert issubclass(image.DecompressionBombWarning, Warning)


# ---------------------------------------------------------------------------
# Signature / keyword contracts — pin the call shapes consumers use.
# ---------------------------------------------------------------------------


def test_open_signature(depcheck):
    """Image.open(fp, mode='r', formats=None). The ``formats`` allow-list
    parameter (a hardening lever to restrict decoders) must remain."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    depcheck.assert_params(image.open, ["fp", "mode", "formats"])


def test_new_signature(depcheck):
    """Image.new(mode, size, color=0) is the in-memory construction shape."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    depcheck.assert_params(image.new, ["mode", "size", "color"])


def test_save_signature(depcheck):
    """img.save(fp, format=None, **params) — consumers pass format= plus
    encoder params (quality=, optimize=, exif=). The **params tail must
    remain so arbitrary encoder kwargs keep flowing through."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    sig = inspect.signature(image.Image.save)
    assert "fp" in sig.parameters
    assert "format" in sig.parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), (
        "Image.save no longer accepts **params (encoder kwargs)"
    )


def test_resize_signature(depcheck):
    """img.resize(size, resample=None, box=None, reducing_gap=None)."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    depcheck.assert_params(image.Image.resize, ["size", "resample"])


def test_thumbnail_signature(depcheck):
    """img.thumbnail(size, resample=..., reducing_gap=2.0) — in-place."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    depcheck.assert_params(image.Image.thumbnail, ["size", "resample"])


def test_convert_signature(depcheck):
    """img.convert(mode=None, ...) — mode normalisation entry point."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    depcheck.assert_params(image.Image.convert, ["mode"])


# ---------------------------------------------------------------------------
# Behavioural contracts — fully offline, in-memory images only.
# ---------------------------------------------------------------------------


def _make_rgb(depcheck, size=(16, 12), color=(10, 20, 30)):
    """Construct a small in-memory RGB image (no disk, no network)."""
    image = depcheck.load("PIL.Image")
    return image, image.new("RGB", size, color)


def test_new_image_basic_properties(depcheck):
    """Image.new yields an object exposing the size/mode/getpixel surface
    consumers read before deciding how to process an image."""
    image, im = _make_rgb(depcheck, size=(16, 12), color=(10, 20, 30))
    assert im.size == (16, 12)
    assert im.width == 16
    assert im.height == 12
    assert im.mode == "RGB"
    assert im.getpixel((0, 0)) == (10, 20, 30)
    assert isinstance(im, image.Image)


def test_convert_rgb_and_rgba(depcheck):
    """convert('RGB')/convert('RGBA') are the mode-normalisation calls
    consumers make so downstream encoders/models get a known channel count."""
    _, im = _make_rgb(depcheck)
    rgba = im.convert("RGBA")
    assert rgba.mode == "RGBA"
    assert rgba.size == im.size
    back = rgba.convert("RGB")
    assert back.mode == "RGB"
    grey = im.convert("L")
    assert grey.mode == "L"


def test_resize_changes_size_deterministically(depcheck):
    """resize() with an explicit Resampling filter returns a new image at the
    requested dimensions (the original is untouched)."""
    image, im = _make_rgb(depcheck, size=(16, 12))
    out = im.resize((8, 6), image.Resampling.LANCZOS)
    assert out.size == (8, 6)
    assert out.mode == "RGB"
    assert im.size == (16, 12)  # resize is non-mutating


def test_thumbnail_is_in_place_and_aspect_preserving(depcheck):
    """thumbnail() mutates in place and never upscales past the box while
    preserving aspect ratio — the contract image-handling code relies on."""
    _, im = _make_rgb(depcheck, size=(40, 20))
    assert im.thumbnail((10, 10)) is None  # returns None, mutates self
    assert im.width <= 10 and im.height <= 10
    # 40x20 (2:1) into a 10x10 box -> 10x5.
    assert im.size == (10, 5)


def test_crop_region(depcheck):
    """crop((l, t, r, b)) returns the requested sub-region size."""
    _, im = _make_rgb(depcheck, size=(20, 20))
    cropped = im.crop((2, 4, 12, 14))
    assert cropped.size == (10, 10)


def test_png_roundtrip_in_memory(depcheck):
    """Construct -> save(PNG) to BytesIO -> reopen: size/mode/format must
    survive the encode/decode round trip with no disk involved."""
    image, im = _make_rgb(depcheck, size=(16, 12), color=(200, 100, 50))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    assert buf.tell() > 0
    buf.seek(0)
    reopened = image.open(buf)
    assert reopened.format == "PNG"
    assert reopened.size == (16, 12)
    assert reopened.mode in ("RGB", "RGBA", "P")
    reopened.load()


def test_jpeg_roundtrip_in_memory(depcheck):
    """save(JPEG, quality=) to BytesIO -> reopen. JPEG is the most common
    uploaded/fetched format and a frequent decoder-CVE target; pin that the
    encoder accepts ``quality`` and the decoder reads the result back."""
    if not _codec_ok(depcheck, "jpg"):
        pytest.skip("JPEG codec not built into this Pillow")
    image, im = _make_rgb(depcheck, size=(24, 18))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    reopened = image.open(buf)
    assert reopened.format == "JPEG"
    assert reopened.size == (24, 18)
    assert reopened.mode == "RGB"
    reopened.load()


def test_webp_roundtrip_in_memory(depcheck):
    """save(WEBP) to BytesIO -> reopen. WEBP support is build-dependent, so
    skip cleanly if the codec is absent rather than failing the contract."""
    if not _codec_ok(depcheck, "webp"):
        pytest.skip("WEBP codec not built into this Pillow")
    image, im = _make_rgb(depcheck, size=(20, 10))
    buf = io.BytesIO()
    im.save(buf, format="WEBP")
    buf.seek(0)
    reopened = image.open(buf)
    assert reopened.format == "WEBP"
    assert reopened.size == (20, 10)
    reopened.load()


def test_rgba_png_roundtrip_preserves_alpha(depcheck):
    """An RGBA image saved as PNG must reopen with an alpha-capable mode —
    consumers that composite/flatten depend on the alpha surviving."""
    image, im = _make_rgb(depcheck, size=(10, 10))
    rgba = im.convert("RGBA")
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    buf.seek(0)
    reopened = image.open(buf)
    assert reopened.mode in ("RGBA", "LA", "P")
    reopened.load()


def test_format_detected_from_content_not_name(depcheck):
    """Image.open identifies the format from magic bytes (a BytesIO has no
    filename), which is what format-detection in the ingestion path relies on.
    Encode as PNG, reopen, and confirm the format is read back as PNG."""
    image, im = _make_rgb(depcheck)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    with image.open(buf) as reopened:
        assert reopened.format == "PNG"


def test_open_garbage_bytes_raises_unidentified(depcheck):
    """The security-critical contract: undecodable/garbage bytes must raise
    UnidentifiedImageError, not silently return a usable image. This is the
    failure mode every attacker-supplied-bytes path leans on."""
    mod = depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    garbage = io.BytesIO(b"this is definitely not an image \x00\x01\x02\x03")
    with pytest.raises(mod.UnidentifiedImageError):
        image.open(garbage)


def test_open_empty_bytes_raises(depcheck):
    """Empty input must also raise (UnidentifiedImageError), never produce a
    zero-sized image silently."""
    mod = depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    with pytest.raises(mod.UnidentifiedImageError):
        image.open(io.BytesIO(b""))


def test_truncated_png_header_raises(depcheck):
    """A valid PNG magic prefix followed by truncated/garbage data must still
    fail to decode (raise), rather than yielding a partial image — guards the
    'looks-like-PNG but isn't' confusion an attacker could exploit."""
    image = depcheck.load("PIL.Image")
    png_magic = b"\x89PNG\r\n\x1a\n"
    truncated = io.BytesIO(png_magic + b"\x00\x00\x00\x00garbagegarbage")
    with pytest.raises(Exception):  # UnidentifiedImageError or decode OSError
        img = image.open(truncated)
        img.load()


def test_exif_transpose_available_and_noop_on_plain_image(depcheck):
    """ImageOps.exif_transpose normalises orientation from EXIF before any
    downstream use. On an image with no EXIF it must return an equivalent
    image (same size), not raise."""
    image, im = _make_rgb(depcheck, size=(16, 12))
    imageops = depcheck.load("PIL.ImageOps")
    result = imageops.exif_transpose(im)
    assert result is not None
    assert result.size == (16, 12)


def test_getexif_roundtrip_orientation(depcheck):
    """getexif() returns a mutable Exif mapping; writing the Orientation tag
    and saving as JPEG with exif=... must round-trip the value. Orientation
    handling is the most common EXIF read in image pre-processing."""
    if not _codec_ok(depcheck, "jpg"):
        pytest.skip("JPEG codec not built into this Pillow")
    mod = depcheck.load(IMPORT_NAME)
    image, im = _make_rgb(depcheck, size=(12, 8))
    orientation_tag = mod.ExifTags.Base.Orientation.value
    assert orientation_tag == 274  # stable EXIF tag id
    exif = im.getexif()
    exif[orientation_tag] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    reopened = image.open(buf)
    read_back = reopened.getexif()
    assert read_back.get(orientation_tag) == 6


def test_exiftags_mappings_present(depcheck):
    """Consumers map EXIF tag ids <-> names via ExifTags.TAGS / ExifTags.Base.
    Both must exist and contain the Orientation entry."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod.ExifTags, "TAGS")
    assert hasattr(mod.ExifTags, "Base")
    assert mod.ExifTags.TAGS.get(274) == "Orientation"


def test_features_check_callable(depcheck):
    """PIL.features.check(codec) is how a consumer probes optional codec
    availability before attempting a format-specific encode/decode. It must
    be callable and return a bool for the core codecs."""
    depcheck.load(IMPORT_NAME)
    features = depcheck.load("PIL.features")
    assert callable(features.check)
    for codec in ("jpg", "zlib"):
        assert isinstance(features.check(codec), bool)


def test_max_image_pixels_guard_is_active(depcheck):
    """Image.MAX_IMAGE_PIXELS is the decompression-bomb ceiling (default
    ~89.5M px). It must be a positive number so the guard is armed; a bump
    that set it to None would silently disable bomb protection for every
    attacker-supplied image in the stack."""
    depcheck.load(IMPORT_NAME)
    image = depcheck.load("PIL.Image")
    assert image.MAX_IMAGE_PIXELS is not None
    assert isinstance(image.MAX_IMAGE_PIXELS, int)
    assert image.MAX_IMAGE_PIXELS > 0


def test_context_manager_protocol(depcheck):
    """`with Image.open(...) as im:` is the idiomatic decode pattern; the
    Image class must remain a context manager so consumers close decoders
    deterministically."""
    image, im = _make_rgb(depcheck)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    with image.open(buf) as opened:
        assert opened.size == im.size
    # Class-level protocol presence (no descriptor execution).
    names = set(dir(image.Image))
    assert "__enter__" in names and "__exit__" in names


def test_imagefile_load_truncated_flag_exists(depcheck):
    """ImageFile.LOAD_TRUNCATED_IMAGES is the module-level toggle that decides
    whether truncated images decode leniently. The flag must remain (consumers
    may read/set it) and default to falsy so truncated input fails closed."""
    depcheck.load(IMPORT_NAME)
    imagefile = depcheck.load("PIL.ImageFile")
    names = set(dir(imagefile))
    assert "LOAD_TRUNCATED_IMAGES" in names
    assert imagefile.LOAD_TRUNCATED_IMAGES is False


# ---------------------------------------------------------------------------
# Local helpers (no cross-file imports — conftest exposes only fixtures).
# ---------------------------------------------------------------------------


def _codec_ok(depcheck, codec: str) -> bool:
    """True if Pillow was built with the given codec (offline probe)."""
    features = depcheck.try_load("PIL.features")
    if features is None:
        return False
    try:
        return bool(features.check(codec))
    except Exception:
        return False
