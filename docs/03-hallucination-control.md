# 03 — Hallucination / repetition control (the key engineering finding)

## The symptom

A **4-page** PDF took **724 seconds** through the persistent server
(`reference_server_pipeline/nougat_convert_server.py`). That's ~181s/page — about
**100× too slow**. First suspicion: CPU fallback.

## It was NOT CPU fallback

Measured on the same tile, same model:

| Test | Time | Rate |
|---|---|---|
| bf16 matmul 4096³ ×50 | 0.054s | **126 TFLOP/s** (GPU is fast) |
| nougat, ONE page, batch=1 | 3.0s | normal |
| nougat, full 4-page PDF, **batch=1** | 17.4s | 4.3s/page — normal |
| nougat, same 4-page PDF, **batch=4 server** | **724s** | pathological |

Same hardware, same model, **40× difference** purely from batching. The GPU was
never the problem.

## Root cause: batch-level hallucination contagion

Nougat decodes autoregressively up to `config.max_length` (**3584 tokens**) and
relies on a variance-based early-stop, `StoppingCriteriaScores`. The critical
detail, from its `__call__`:

```python
return all(self.stopped.values()) and len(self.stopped) > 0
```

**Generation only halts when EVERY page in the batch has settled.** So with
`batch_size=4`, a single hallucinating/repeating page keeps the whole batch
generating to `max_length`. The 3 good pages finish early but are dragged along
to 3584 tokens. That's the 724s.

Secondary factors:
- The variance early-stop has a `window_size=200` warm-up — it cannot stop
  before 200 tokens regardless.
- Nougat's post-hoc repetition detector truncates output but only *after* the
  expensive generation already happened.

## The fix: three independent layers

Implemented in [`../src/nougat_infer.py`](../src/nougat_infer.py).

### Layer 1 — `batch_size = 1`  (eliminates contagion)
One runaway page can never drag neighbors. This alone took the 4-page doc from
**724s → 18.8s**.

### Layer 2 — hard token cap (`NOUGAT_MAX_NEW_TOKENS`, default 1536)
The variance heuristic can miss some runaways. We temporarily lower
`model.config.max_length` to the cap for each page so generation is *always*
bounded, regardless of the heuristic. 1536 tokens comfortably covers a dense
page; a page wanting more is hallucinating.

### Layer 3 — per-page wall-clock watchdog (`NOUGAT_PAGE_TIMEOUT`, default 90s)
Ultimate backstop for a native hang/pathological input. Generation runs in a
thread; if it exceeds the budget we abandon that page (emit empty / mark
`timed_out`) and keep the rank alive instead of wedging it.

### Repetitions are flagged, not fatal
Nougat's `repeats` field tells us which pages were truncated due to detected
repetition. We surface per-document counts:

```json
{ "pages": 4, "repeated": 1, "timed_out": 0, "chars": 14384, "ok": true }
```

This lets a caller route `repeated`/`timed_out` pages to a fallback (e.g.
marker) **without failing the whole document**.

## Measured effect

| | old server (batch=4) | new (batch=1 + caps) |
|---|---|---|
| 4-page PDF | 724s | 18.8s |
| 16-tile run, 32 docs / 268 pages | — | 110.7s, **0 timeouts**, 30 pages cleanly truncated |

## Tuning

- Throughput-critical, repetition-heavy corpus: lower `NOUGAT_MAX_NEW_TOKENS`
  (e.g. 1024) — trades a little recall for speed.
- Very dense pages getting clipped: raise it (e.g. 2048) at some throughput cost.
- Adjust `NOUGAT_PAGE_TIMEOUT` to your slowest legitimate page × ~1.5.
