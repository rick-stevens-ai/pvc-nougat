#!/usr/bin/env python
"""
nougat_mpi.py -- 1 MPI rank per GPU tile Nougat conversion.

LAUNCH (16 ranks = 16 tiles on a chiatta/PVC node):
    mpiexec -n 16 python nougat_mpi.py --pdfdir <DIR> --outdir <DIR> [--db <q.sqlite>]

TILE PINNING
    Each rank pins itself to ONE tile via ZE_AFFINITY_MASK = local_rank, set
    BEFORE torch/ipex import so Level Zero only ever sees that single tile as
    xpu:0. We derive local_rank from MPI_LOCALRANKID / PMI_LOCAL_RANK (Intel MPI)
    with a global-rank-mod-tiles fallback.

WORK DISTRIBUTION (lock-free, Lustre-safe -- same contract as the known-good run)
    - Rank 0 scans pdfdir and writes a sorted manifest (once).
    - Every rank reads the manifest read-only, then processes the disjoint shard
      where (index % world_size) == rank.  No shared writes, no SQLite locking.
    - Each rank appends results to its own JSONL: <outdir>/_results/rank_<r>.jsonl
    - A separate reconcile step (or rank 0 at the end) folds JSONL -> summary.

HALLUCINATION CONTROL is in nougat_infer.convert (batch=1 + hard token cap +
per-page watchdog + repetition flagging).  See that module's docstring.

RESUME: a pdf whose .mmd already exists (size>0) is skipped.
"""
import os, sys, json, time, glob, argparse, socket


def detect_local_rank(world_rank):
    for k in ("MPI_LOCALRANKID", "PMI_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
              "SLURM_LOCALID"):
        v = os.environ.get(k)
        if v is not None:
            return int(v)
    # fallback: assume single node, tiles == world size
    return world_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tiles-per-node", type=int,
                    default=int(os.environ.get("TILES_PER_NODE", "16")))
    ap.add_argument("--model-tag", default="0.1.0-small")
    ap.add_argument("--max", type=int, default=0, help="cap docs per rank (debug)")
    args = ap.parse_args()

    # ---- pin tile BEFORE importing torch/ipex ----
    from mpi4py import MPI  # mpi4py import is torch-independent
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    local = detect_local_rank(rank) % args.tiles_per_node
    os.environ["ZE_AFFINITY_MASK"] = str(local)
    host = socket.gethostname()

    # now safe to import the heavy stack (sees only the pinned tile)
    import torch  # noqa
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nougat_infer

    outdir = os.path.abspath(args.outdir)
    resdir = os.path.join(outdir, "_results")
    manifest_path = os.path.join(outdir, "_manifest.json")
    os.makedirs(resdir, exist_ok=True)

    # ---- rank 0 builds manifest, others wait ----
    if rank == 0:
        pdfs = sorted(
            p for p in glob.glob(os.path.join(args.pdfdir, "**", "*.pdf"),
                                 recursive=True)
        )
        json.dump(pdfs, open(manifest_path, "w"))
        print(f"[rank0 {host}] manifest: {len(pdfs)} pdfs", flush=True)
    comm.Barrier()
    pdfs = json.load(open(manifest_path))

    # ---- model load (one per rank/tile) ----
    t0 = time.time()
    model = nougat_infer.load_model(args.model_tag)
    print(f"[r{rank:02d} {host} tile{local}] model ready {time.time()-t0:.1f}s "
          f"(xpu:0 = {torch.xpu.get_device_name(0)})", flush=True)
    comm.Barrier()

    # ---- process disjoint shard ----
    mine = [(i, p) for i, p in enumerate(pdfs) if i % world == rank]
    res_path = os.path.join(resdir, f"rank_{rank}.jsonl")
    fh = open(res_path, "a", buffering=1)
    done = 0
    t_run = time.time()
    for idx, pdf in mine:
        oid = os.path.splitext(os.path.basename(pdf))[0]
        out = os.path.join(outdir, oid + ".mmd")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue  # resume
        r = nougat_infer.convert(pdf, out, model=model)
        r.update({"rank": rank, "tile": local, "host": host,
                  "osti_id": oid, "pdf": pdf, "ts": time.time()})
        fh.write(json.dumps(r) + "\n")
        done += 1
        flag = "REP" if r.get("repeated") else ("TO" if r.get("timed_out") else "ok")
        print(f"[r{rank:02d} t{local}] {oid[:24]:26} {r['pages']:>2}pg "
              f"{r['sec']:6.1f}s {flag} chars={r['chars']}", flush=True)
        if args.max and done >= args.max:
            break
    fh.close()
    elapsed = time.time() - t_run
    pages = 0
    # local tally
    counts = {"docs": done, "elapsed": round(elapsed, 1)}
    allc = comm.gather(counts, root=0)

    if rank == 0:
        # ---- reconcile ----
        tot_docs = tot_pages = tot_rep = tot_to = tot_chars = 0
        per_rank = []
        for r in range(world):
            rp = os.path.join(resdir, f"rank_{r}.jsonl")
            d = p = rep = to = ch = 0
            if os.path.exists(rp):
                for line in open(rp):
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    d += 1; p += j.get("pages", 0); ch += j.get("chars", 0)
                    rep += j.get("repeated", 0); to += j.get("timed_out", 0)
            per_rank.append((r, d, p))
            tot_docs += d; tot_pages += p; tot_rep += rep
            tot_to += to; tot_chars += ch
        wall = max((c["elapsed"] for c in allc), default=elapsed)
        print("=" * 72)
        print(f"NOUGAT MPI DONE | {world} ranks/tiles on {host}")
        print(f"  docs={tot_docs}  pages={tot_pages}  chars={tot_chars}")
        print(f"  repetition-flagged pages={tot_rep}  watchdog-timeouts={tot_to}")
        print(f"  wall (slowest rank)={wall:.1f}s")
        if wall > 0:
            print(f"  THROUGHPUT: {tot_pages/wall:.2f} pages/s  "
                  f"({tot_pages/wall*60:.1f} pages/min)  "
                  f"{tot_docs/wall*60:.1f} docs/min  across {world} tiles")
        json.dump({"docs": tot_docs, "pages": tot_pages, "chars": tot_chars,
                   "repeated": tot_rep, "timed_out": tot_to, "wall": wall,
                   "ranks": world},
                  open(os.path.join(outdir, "_summary.json"), "w"), indent=2)
        print("=" * 72)


if __name__ == "__main__":
    main()
