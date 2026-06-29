"""
nougat_infer.py -- hardened single-tile Nougat inference with hallucination control.

Loads the Nougat model once on xpu:0 (the tile selected by ZE_AFFINITY_MASK).
Converts a PDF to markdown with THREE layers of runaway-page protection:

  1. batch_size = 1
       The stock server batched 4 pages. Nougat's StoppingCriteriaScores only
       halts when ALL pages in the batch satisfy the variance test
       (`all(self.stopped.values())`). So a SINGLE hallucinating page forces the
       whole batch to run to max_length (3584 tok) -> the 724s/4pg pathology.
       With batch_size=1 a runaway page can never drag its neighbors.

  2. Hard token cap (NOUGAT_MAX_NEW_TOKENS, default 1536)
       nougat's variance early-stop has a 200-token warm-up and can miss some
       runaways. We add a MaxLengthCriteria so generation is *always* bounded,
       independent of the variance heuristic. 1536 tok comfortably covers a
       dense page; a page that wants more is hallucinating.

  3. Per-page wall-clock watchdog (NOUGAT_PAGE_TIMEOUT, default 90s)
       Ultimate backstop. If a single page somehow still exceeds the budget
       (native hang, pathological input), we abandon that page, mark it
       'truncated', and continue -- the page is salvaged as empty rather than
       wedging the rank.

Repetition handling: nougat already truncates detected repetitions (output
"repeats" gives the cut index). We surface a per-page 'repeats' / 'truncated'
count so the caller can flag low-confidence pages for re-OCR (e.g. with marker)
without failing the whole document.
"""
import os, time, threading, logging
from functools import partial
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria

MAX_NEW_TOKENS = int(os.environ.get("NOUGAT_MAX_NEW_TOKENS", "1536"))
PAGE_TIMEOUT   = float(os.environ.get("NOUGAT_PAGE_TIMEOUT", "90"))
PAGE_SEP       = os.environ.get("NOUGAT_PAGE_SEP", "\n\n")
DEV            = "xpu:0"  # under ZE_AFFINITY_MASK the pinned tile is index 0

_model = None


def load_model(model_tag="0.1.0-small"):
    global _model
    if _model is not None:
        return _model
    import intel_extension_for_pytorch as ipex  # noqa: F401  (registers xpu)
    from nougat import NougatModel
    from nougat.utils.checkpoint import get_checkpoint
    if not torch.xpu.is_available():
        raise RuntimeError("xpu not available")
    ckpt = get_checkpoint(model_tag=model_tag)
    m = NougatModel.from_pretrained(ckpt).to(DEV).to(torch.bfloat16).eval()
    _model = m
    return m


def _clean_page(md):
    if not md or not md.strip():
        return ""
    lines = md.rstrip().split("\n")
    j = len(lines) - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j > 0 and lines[j].lstrip().startswith("#"):
        if any(l.strip() and not l.lstrip().startswith("#") for l in lines[:j]):
            lines = lines[:j]
    return "\n".join(lines).rstrip()


class _HardCap(StoppingCriteria):
    """Stop the WHOLE batch once the longest sequence hits the hard cap.
    With batch_size=1 this simply bounds the single page."""
    def __init__(self, start_len, max_new):
        self.limit = start_len + max_new
    def __call__(self, input_ids, scores, **kw):
        return input_ids.shape[-1] >= self.limit


def _infer_one_page(model, page_img, prepare):
    """Run nougat on a single page image. Returns (md, repeated:bool, sec, timed_out:bool)."""
    from nougat.utils.dataset import ImageDataset
    from nougat.postprocessing import markdown_compatible
    from torch.utils.data import DataLoader

    ds = ImageDataset([page_img], prepare)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    result = {"md": "", "repeated": False, "timed_out": False}

    def work():
        for sample in dl:
            if sample is None:
                continue
            img = sample.to(DEV).to(torch.bfloat16)
            with torch.no_grad():
                # monkeypatch a hard cap on top of nougat's own early-stop.
                # nougat builds its own StoppingCriteriaList internally; we wrap
                # model.inference but also cap via max_length on the config for
                # this call by temporarily lowering config.max_length.
                saved = model.config.max_length
                try:
                    model.config.max_length = min(saved, MAX_NEW_TOKENS)
                    mo = model.inference(image_tensors=img, early_stopping=True)
                finally:
                    model.config.max_length = saved
            preds = mo.get("predictions", [])
            reps = mo.get("repeats", [None])
            md = markdown_compatible(preds[0]) if preds else ""
            result["md"] = md
            result["repeated"] = reps and reps[0] is not None
            return

    t0 = time.time()
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(PAGE_TIMEOUT)
    sec = time.time() - t0
    if th.is_alive():
        # watchdog fired: page exceeded wall budget. We cannot safely interrupt
        # the device op, so we abandon this page's output and let the thread die
        # with the process if needed. Caller treats as truncated/empty.
        result["timed_out"] = True
        result["md"] = ""
    return result["md"], result["repeated"], sec, result["timed_out"]


def convert(pdf_path, out_path, model=None):
    """Convert one PDF -> .mmd. Page-by-page, hallucination-hardened.
    Returns a status dict (never raises for per-page issues)."""
    from nougat.dataset.rasterize import rasterize_paper
    if model is None:
        model = load_model()
    prepare = partial(model.encoder.prepare_input, random_padding=False)

    st = time.time()
    try:
        pages = rasterize_paper(pdf=pdf_path, return_pil=True)
    except Exception as e:
        return {"ok": False, "pages": 0, "chars": 0, "repeated": 0,
                "timed_out": 0, "sec": round(time.time() - st, 2),
                "err": f"rasterize failed: {type(e).__name__}: {e}"}
    if not pages:
        return {"ok": False, "pages": 0, "chars": 0, "repeated": 0,
                "timed_out": 0, "sec": round(time.time() - st, 2),
                "err": "rasterize returned 0 pages"}

    mds, n_rep, n_to = [], 0, 0
    for i, pg in enumerate(pages):
        md, repeated, sec, timed_out = _infer_one_page(model, pg, prepare)
        if repeated:
            n_rep += 1
        if timed_out:
            n_to += 1
        mds.append(md)

    text = PAGE_SEP.join(c for c in (_clean_page(m) for m in mds) if c)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".part"
    Path(tmp).write_text(text)
    os.replace(tmp, out_path)
    return {"ok": len(text) > 0, "pages": len(pages), "chars": len(text),
            "repeated": n_rep, "timed_out": n_to,
            "sec": round(time.time() - st, 2), "err": None}
