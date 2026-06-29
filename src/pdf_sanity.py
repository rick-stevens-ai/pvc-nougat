#!/usr/bin/env python
"""
pdf_sanity.py -- cheap pre-flight PDF validity check for the Nougat/Marker pipeline.

Rationale (2026-06-29, Rick): before we spend a model round-trip rasterizing a PDF,
reject obviously-bad inputs cheaply. The Aurora prod run's dominant failure bucket was
~100 "rasterize returned 0 pages" -- corrupt/empty/truncated PDFs that can't be rendered
by ANY extractor. Catching these in the WORKER (before dispatch) means:
  * no server round-trip wasted on a doc that can't possibly succeed
  * no risk of a malformed PDF triggering a native crash inside rasterize -> server respawn
  * a DISTINCT error tag ('pdf_sanity:<reason>') so these route cleanly to the recovery
    pipeline (re-fetch a clean PDF / marker fallback) instead of being mixed in with
    real Nougat failures.

Design:
  * STDLIB-ONLY layer always runs (no deps): exists, size floor, %PDF- magic,
    %%EOF trailer, /Encrypt scan. Fast (reads head + tail only, never the whole file).
  * DEEP layer via PyMuPDF (fitz) if importable: open, page_count in (1, ceiling],
    is_encrypted/needs_pass, and a render-probe of page 0 at low DPI to catch PDFs that
    parse structurally but explode on rasterize. fitz is already in the nougat env;
    if it's absent the stdlib layer still gates the common corruption modes.

Public API:
    ok, reason = pdf_sanity(path)            # bool, str|None
    ok, reason = pdf_sanity(path, deep=True) # also run the fitz render-probe

`reason` is a short stable token suitable for the queue 'err' column, e.g.:
    "missing" "empty" "too_small:512" "no_pdf_magic" "no_eof_trailer"
    "encrypted" "fitz_open_failed:..." "zero_pages" "too_many_pages:N"
    "render_probe_failed:..."
On success reason is None.
"""
import os

# size floor: a real paper PDF is essentially never < 2KB. Catches 0-byte stubs and
# truncated downloads (the Dropbox online-only-stub failure mode, partial fetches).
MIN_BYTES = int(os.environ.get("PDF_SANITY_MIN_BYTES", "2048"))
# page-count ceiling: guards against pathological/abuse PDFs that would never finish.
# 0 disables the ceiling. Default generous (theses/proceedings can be long).
MAX_PAGES = int(os.environ.get("PDF_SANITY_MAX_PAGES", "2000"))
# how many bytes of the tail to scan for the %%EOF trailer
_TAIL = 4096
# how many bytes of the head to scan for the /Encrypt token (cheap heuristic;
# the authoritative encryption check is fitz.is_encrypted in the deep layer)
_HEAD = 65536


def _stdlib_checks(path):
    """No-dependency structural checks. Returns reason str or None."""
    try:
        sz = os.path.getsize(path)
    except OSError:
        return "missing"
    if sz == 0:
        return "empty"
    if sz < MIN_BYTES:
        return f"too_small:{sz}"
    try:
        with open(path, "rb") as f:
            head = f.read(_HEAD)
            # %PDF magic must appear within the first 1KB (spec allows leading junk,
            # but real files have it at offset 0; be lenient to 1KB).
            if b"%PDF-" not in head[:1024]:
                return "no_pdf_magic"
            # tail scan for %%EOF (a truncated download usually lacks it)
            if sz <= _TAIL:
                tail = head
            else:
                f.seek(-_TAIL, os.SEEK_END)
                tail = f.read(_TAIL)
            if b"%%EOF" not in tail:
                return "no_eof_trailer"
    except OSError as e:
        return f"read_error:{type(e).__name__}"
    return None


def _pdfium_checks(path, render_probe):
    """Deep checks via pypdfium2 -- the SAME backend nougat's rasterize_paper uses,
    so this mirrors the real rasterize path: if pdfium can open + count + render page 0,
    rasterize will too. Returns (handled: bool, reason: str|None).
    handled=False means pypdfium2 isn't importable (caller should try the next backend).
    """
    try:
        import pypdfium2 as pdfium
    except Exception:
        return False, None
    pdf = None
    try:
        try:
            pdf = pdfium.PdfDocument(path)
        except Exception as e:
            # pdfium raises PdfiumError on encrypted/corrupt; surface the type
            msg = type(e).__name__
            low = str(e).lower()
            if "password" in low or "encrypt" in low:
                return True, "encrypted"
            return True, f"pdfium_open_failed:{msg}"
        n = len(pdf)
        if n <= 0:
            return True, "zero_pages"
        if MAX_PAGES and n > MAX_PAGES:
            return True, f"too_many_pages:{n}"
        if render_probe:
            try:
                page = pdf[0]
                # render at low scale (~26 DPI: 72*0.36) -- catches pages that parse but
                # bomb on raster (the exact "rasterize returned 0 pages" / crash class)
                bmp = page.render(scale=0.36)
                pil = bmp.to_pil()
                if pil.width <= 0 or pil.height <= 0:
                    return True, "render_probe_empty"
                try:
                    page.close()
                except Exception:
                    pass
            except Exception as e:
                return True, f"render_probe_failed:{type(e).__name__}"
        return True, None
    finally:
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass


def _fitz_checks(path, render_probe):
    """Deep checks via PyMuPDF. Returns (handled: bool, reason: str|None).
    handled=False means fitz isn't importable.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return False, None
    doc = None
    try:
        try:
            doc = fitz.open(path)
        except Exception as e:
            return True, f"fitz_open_failed:{type(e).__name__}"
        # encryption: nougat/marker can't rasterize a locked doc
        if getattr(doc, "needs_pass", False):
            return True, "encrypted"
        n = doc.page_count
        if n <= 0:
            return True, "zero_pages"
        if MAX_PAGES and n > MAX_PAGES:
            return True, f"too_many_pages:{n}"
        if render_probe:
            try:
                # low-DPI render of page 0 -- catches docs that parse but explode on
                # rasterize (the exact "rasterize returned 0 pages" / native-crash class).
                pg = doc.load_page(0)
                pix = pg.get_pixmap(dpi=36)
                if pix.width <= 0 or pix.height <= 0:
                    return True, "render_probe_empty"
            except Exception as e:
                return True, f"render_probe_failed:{type(e).__name__}"
        return True, None
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def pdf_sanity(path, deep=False, render_probe=False):
    """Return (ok: bool, reason: str|None).

    deep=True       -> also run a deep open/page-count/encryption check via the first
                       available backend: pypdfium2 (preferred -- same engine nougat's
                       rasterize uses) then PyMuPDF (fitz). If neither is importable,
                       only the stdlib structural layer applies.
    render_probe=True implies deep; also low-scale renders page 0 to catch rasterize bombs.
    """
    if render_probe:
        deep = True
    r = _stdlib_checks(path)
    if r is not None:
        return False, r
    if deep:
        # Prefer pdfium (mirrors the real rasterize path); fall back to fitz.
        handled, r = _pdfium_checks(path, render_probe)
        if not handled:
            handled, r = _fitz_checks(path, render_probe)
        if handled and r is not None:
            return False, r
    return True, None


if __name__ == "__main__":
    import sys
    deep = "--deep" in sys.argv or "--render-probe" in sys.argv
    rp = "--render-probe" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        print("usage: pdf_sanity.py [--deep] [--render-probe] file.pdf [file.pdf ...]")
        sys.exit(2)
    bad = 0
    for p in paths:
        ok, reason = pdf_sanity(p, deep=deep, render_probe=rp)
        print(f"{'OK  ' if ok else 'BAD '} {p}" + ("" if ok else f"  [{reason}]"))
        if not ok:
            bad += 1
    sys.exit(1 if bad else 0)
