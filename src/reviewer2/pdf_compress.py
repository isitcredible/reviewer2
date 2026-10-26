"""
pdf_compress.py — shrink oversized PDFs to fit Gemini's 50 MB document limit
by stripping photographic figure images, while never touching tables or text.

Why this is safe for Reviewer 2: the pipeline reads PDFs through Gemini's
vision rendering of each page, and the finders are explicitly instructed to
ignore figures/images. Removing a photographic figure is therefore lossless
for the review. Tables — INCLUDING tables pasted as raster images — and body
text must be preserved exactly, so the stripper is deliberately conservative
and fails safe (keeps the image) whenever it is unsure.

An image is stripped only when ALL of these hold:
  * it does not overlap a detected (vector) table region;
  * it is not a page-filling image on a text-sparse (scanned) page;
  * it is confidently "photographic" (colourful / continuous-tone, not
    white-dominated) — a table-as-image is mostly white with sharp text and
    rules, so it reads as NOT photographic and is kept.

Everything else (line art, charts, anything unreadable, anything table-like)
is left untouched. Vector content — body text, table rules, equations — is
never an image XObject and is never affected.

The entry point, compress_pdf_for_gemini(), is a no-op for any PDF already
under the target size, so the normal (small) paper path pays nothing.
"""

import io
import os
import tempfile

import fitz  # PyMuPDF
from PIL import Image

GEMINI_PDF_LIMIT = 50 * 1024 * 1024   # hard Gemini per-PDF cap
DEFAULT_TARGET   = 48 * 1024 * 1024   # leave headroom under the cap
MIN_IMAGE_BYTES  = 200 * 1024         # ignore images smaller than this
SCANNED_TEXT_CHARS = 120              # page text below this + page-wide image => scanned
PAGEWIDE_FRAC      = 0.55             # image covering >55% of page area
TABLE_OVERLAP      = 0.30             # image/table intersection fraction to count as "in a table"
WHITE_FRAC_PROTECT = 0.78             # >=78% near-white pixels => text/table-like => protect
MAX_COLORS_PROTECT = 256             # <=256 distinct colours => text/table-like => protect


def _overlap_frac(img_rect, tbl_rect):
    """Fraction of the image rectangle that lies inside the table rectangle."""
    inter = img_rect & tbl_rect
    if inter.is_empty:
        return 0.0
    area = abs(img_rect.get_area())
    return abs(inter.get_area()) / area if area else 0.0


def _looks_photographic(image_bytes):
    """True only for confidently photographic images (safe to strip).

    Fails safe: anything unreadable or white/line-art-dominated returns False
    so it is protected.
    """
    if not image_bytes:
        return False
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False  # e.g. JBIG2 scanned text, exotic codec -> protect
    im.thumbnail((160, 160))
    px = list(im.getdata())
    n = len(px) or 1
    white = sum(1 for (r, g, b) in px if r >= 235 and g >= 235 and b >= 235)
    if white / n >= WHITE_FRAC_PROTECT:
        return False  # white-dominated -> text/table-like -> protect
    return len(set(px)) > MAX_COLORS_PROTECT


def compress_pdf_for_gemini(input_path, target_bytes=DEFAULT_TARGET):
    """Return (path_to_use, info).

    If the PDF is already under target_bytes, returns the original path
    unchanged. Otherwise strips photographic figures into a temp file and
    returns that. On ANY error, returns the original path so a job is never
    blocked worse than the status quo.
    """
    info = {"action": "none", "orig_bytes": 0, "final_bytes": 0,
            "stripped": 0, "protected": 0, "over_limit": False}
    try:
        orig = os.path.getsize(input_path)
    except OSError:
        return input_path, info
    info["orig_bytes"] = info["final_bytes"] = orig
    if orig <= target_bytes:
        return input_path, info

    try:
        doc = fitz.open(input_path)
    except Exception:
        return input_path, info

    try:
        xref_pages = {}      # xref -> set of page numbers referencing it
        sizes = {}           # xref -> stored byte size
        bytes_cache = {}     # xref -> raw image bytes (for classification)
        protected = set()    # xrefs we must not strip

        for pno in range(doc.page_count):
            page = doc[pno]
            text_len = len(page.get_text("text").strip())
            try:
                table_rects = [fitz.Rect(t.bbox) for t in page.find_tables().tables]
            except Exception:
                table_rects = []
            page_area = abs(page.rect.get_area()) or 1.0

            for img in page.get_images(full=True):
                xref = img[0]
                xref_pages.setdefault(xref, set()).add(pno)
                if xref not in sizes:
                    try:
                        ex = doc.extract_image(xref)
                        bytes_cache[xref] = ex.get("image", b"")
                        sizes[xref] = len(bytes_cache[xref])
                    except Exception:
                        bytes_cache[xref] = b""
                        sizes[xref] = 0
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                for r in rects:
                    if abs(r.get_area()) / page_area > PAGEWIDE_FRAC and text_len < SCANNED_TEXT_CHARS:
                        protected.add(xref)   # scanned page: image IS the content
                    if any(_overlap_frac(r, tr) > TABLE_OVERLAP for tr in table_rects):
                        protected.add(xref)   # sits inside a table region

        info["protected"] = len(protected)

        to_strip = [
            xref for xref, sz in sizes.items()
            if xref not in protected and sz >= MIN_IMAGE_BYTES
            and _looks_photographic(bytes_cache.get(xref))
        ]

        if not to_strip:
            doc.close()
            info["over_limit"] = orig > GEMINI_PDF_LIMIT
            return input_path, info

        for xref in to_strip:
            for pno in xref_pages.get(xref, ()):
                try:
                    doc[pno].delete_image(xref)
                except Exception:
                    pass

        fd, out = tempfile.mkstemp(suffix=".pdf", prefix="gemini_min_")
        os.close(fd)
        doc.save(out, garbage=4, deflate=True, clean=True)
        doc.close()

        final = os.path.getsize(out)
        if final >= orig:            # compression didn't help; keep original
            try:
                os.remove(out)
            except OSError:
                pass
            info["over_limit"] = orig > GEMINI_PDF_LIMIT
            return input_path, info

        info.update(action="stripped", final_bytes=final,
                    stripped=len(to_strip), over_limit=final > GEMINI_PDF_LIMIT)
        return out, info

    except Exception:
        try:
            doc.close()
        except Exception:
            pass
        return input_path, info
