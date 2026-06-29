#!/usr/bin/env python
"""
nougat_convert_server.py  --  persistent per-tile Nougat conversion server.

Loads the Nougat model ONCE on the tile selected by ZE_AFFINITY_MASK (visible as xpu:0).
Then reads newline-delimited JSON requests on stdin:  {"pdf": "<path>", "out": "<path>"}
For each, converts and writes one newline-delimited JSON status on stdout:
    {"pdf":..., "ok":bool, "pages":int, "chars":int, "sec":float, "chunks":int, "err":str|None}

PAGE CHUNKING:
  Large PDFs are processed in windows of CHUNK_PAGES (default 25, override via NOUGAT_CHUNK_PAGES).
  This bounds per-window wall time / device memory, keeps tiles balanced near end-of-queue,
  and emits per-window progress on stderr. Output is concatenated IN ORDER -> one .mmd per paper.

Design contract (fault isolation):
  - If a PDF triggers a NATIVE crash (SIGSEGV/SIGABRT in C/CUDA/L0 code), THIS process dies.
    The parent worker detects EOF on our stdout, marks that pdf failed, and respawns a fresh server.
    A poison PDF can therefore never wedge the tile or the worker permanently.
  - All Python-level exceptions are caught and returned as {"ok":false,"err":...} -- the server keeps serving.
"""
import sys, os, json, time, traceback
from pathlib import Path
from functools import partial

CHUNK_PAGES = int(os.environ.get("NOUGAT_CHUNK_PAGES", "12"))
PAGE_SEP = os.environ.get("NOUGAT_PAGE_SEP", "\n\n")  # how page-markdowns are glued

# --- PROTOCOL CHANNEL ISOLATION (critical) ---------------------------------
# nougat's postprocessing prints diagnostic lines like
#   "INFO: likely hallucinated title at the end of the page: ..."
# directly to stdout. Our worker<->server protocol ALSO uses stdout for the
# newline-delimited JSON responses. If those collide, the worker does
# json.loads() on an "INFO:" line, gets "Expecting value", and wrongly
# declares the server dead -- discarding a fully-completed document.
# (Root-caused 2026-06-25: 1023146 finished all 3 windows then "failed" on the
#  trailing INFO line.) Fix: dup the REAL stdout fd for protocol use, then
# point Python-level sys.stdout at stderr so ALL library prints become harmless
# stderr logging and stdout carries ONLY our JSON frames.
_PROTO_FD = os.dup(1)            # real stdout, reserved for JSON protocol
os.dup2(2, 1)                    # fd 1 now points at stderr (kills stdout noise at fd level)
sys.stdout = sys.stderr          # python-level prints also go to stderr
_proto = os.fdopen(_PROTO_FD, "w", buffering=1)

def emit(obj):
    """Write one JSON frame on the reserved protocol channel."""
    _proto.write(json.dumps(obj) + "\n")
    _proto.flush()

def log(msg):
    print(f"[server tile={os.environ.get('ZE_AFFINITY_MASK','?')} pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)

import torch
if not torch.xpu.is_available():
    log("FATAL: xpu not available")
    sys.exit(2)

from nougat import NougatModel
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import ImageDataset
from nougat.utils.device import move_to_device
from nougat.postprocessing import markdown_compatible
from nougat.dataset.rasterize import rasterize_paper
from torch.utils.data import DataLoader

DEV = "xpu:0"  # under ZE_AFFINITY_MASK the pinned tile is always index 0

t0 = time.time()
ckpt = get_checkpoint(model_tag="0.1.0-small")
model = NougatModel.from_pretrained(ckpt)
model = model.to(DEV).to(torch.bfloat16)
model.eval()
log(f"model loaded on {DEV} in {time.time()-t0:.1f}s (chunk={CHUNK_PAGES}p) -- READY")
# signal readiness on stdout so parent knows model is up
emit({"ready": True, "load_sec": round(time.time()-t0, 2)})

def _clean_page(md):
    """Per-page cleanup before gluing.
    - Returns "" for empty/whitespace-only pages (blank or unreadable) so they
      don't create dead gaps in the joined output.
    - Strips a TRAILING lone heading: nougat's known page-boundary hallucination
      is to emit the *next* section's title at the very end of a page (logged as
      'likely hallucinated title at the end of the page'). The real heading then
      reappears at the top of the following page, so dropping the trailing one
      removes a duplicate without losing content. Only strips if the heading is
      the final non-empty line AND there is body text above it (so single-heading
      pages -- e.g. section dividers -- are preserved).
    """
    if not md or not md.strip():
        return ""
    lines = md.rstrip().split("\n")
    # find last non-empty line
    j = len(lines) - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j > 0 and lines[j].lstrip().startswith("#"):
        # is there real body content above it?
        if any(l.strip() and not l.lstrip().startswith("#") for l in lines[:j]):
            lines = lines[:j]
    return "\n".join(lines).rstrip()

def _glue(page_mds):
    """Join per-page markdown into one document, dropping empties."""
    cleaned = [c for c in (_clean_page(p) for p in page_mds) if c]
    return PAGE_SEP.join(cleaned)

def _infer_window(window_pages):
    """Run nougat inference over a list of PIL pages (a window), return list of md strings in order."""
    prepare = partial(model.encoder.prepare_input, random_padding=False)
    dataset = ImageDataset(window_pages, prepare)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    out = []
    for sample in loader:
        if sample is None:
            continue
        image_tensors = sample.to(DEV).to(torch.bfloat16)
        with torch.no_grad():
            # early_stopping=True is CRITICAL: it stops generation when a page is done AND
            # engages nougat's repetition guard so runaway/hallucinating pages get cut off.
            # With early_stopping=False, every page generates to max_length and runaway pages
            # never terminate -> docs hit the watchdog wall at exactly the timeout (root-caused
            # 2026-06-25: a 9-page doc went 400s->25.5s just by flipping this flag).
            mo = model.inference(image_tensors=image_tensors, early_stopping=True)
        for pred in mo["predictions"]:
            out.append(markdown_compatible(pred))
    return out

def convert(pdf_path, out_path, pg_lo=None, pg_hi=None):
    st = time.time()
    pages = rasterize_paper(pdf=pdf_path, return_pil=True)
    if not pages:
        return {"ok": False, "pages": 0, "chars": 0, "chunks": 0,
                "sec": round(time.time()-st, 2), "err": "rasterize returned 0 pages"}
    # virtual split: process only [pg_lo, pg_hi) if requested (option B)
    total_pg = len(pages)
    if pg_lo is not None or pg_hi is not None:
        lo0 = pg_lo or 0
        hi0 = pg_hi if pg_hi is not None else total_pg
        pages = pages[lo0:hi0]
        if not pages:
            return {"ok": False, "pages": 0, "chars": 0, "chunks": 0,
                    "sec": round(time.time()-st, 2),
                    "err": f"page range [{lo0}:{hi0}) empty of {total_pg}"}
    npg = len(pages)
    chunks, npages = [], 0
    n_windows = (npg + CHUNK_PAGES - 1) // CHUNK_PAGES
    for wi in range(n_windows):
        lo = wi * CHUNK_PAGES
        hi = min(lo + CHUNK_PAGES, npg)
        ws = time.time()
        win_md = _infer_window(pages[lo:hi])
        chunks.extend(win_md)
        npages += len(win_md)
        if n_windows > 1:
            log(f"  {Path(pdf_path).stem}: window {wi+1}/{n_windows} pages[{lo}:{hi}] {time.time()-ws:.1f}s")
    text = _glue(chunks)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # atomic write
    tmp = out_path + ".part"
    Path(tmp).write_text(text)
    os.replace(tmp, out_path)
    return {"ok": len(text) > 0, "pages": npages, "chars": len(text), "chunks": n_windows,
            "sec": round(time.time()-st, 2), "err": None}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception as e:
        emit({"pdf": None, "ok": False, "err": f"bad request json: {e}"})
        continue
    pdf, out = req.get("pdf"), req.get("out")
    pg_lo, pg_hi = req.get("pg_lo"), req.get("pg_hi")
    resp = {"pdf": pdf, "pg_lo": pg_lo, "pg_hi": pg_hi}
    try:
        resp.update(convert(pdf, out, pg_lo=pg_lo, pg_hi=pg_hi))
    except Exception as e:
        resp.update({"ok": False, "pages": 0, "chars": 0, "chunks": 0, "sec": 0, "err": f"{type(e).__name__}: {e}"})
        log("convert exception:\n" + traceback.format_exc())
    emit(resp)
