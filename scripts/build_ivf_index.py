#!/usr/bin/env python
"""Build an IVF(-PQ) index from an existing flat FAISS index.

The flat 2M index (e5_Flat_2M.index) does exact search: ~0.54s/query on CPU,
and the batched path in retrieval_server.py is even slower (~260s for 32
queries). An IVF index coarse-quantizes the corpus so each query only scans a
few cells, dropping search to ~1-5 ms/query with negligible recall loss.

Run inside the `retriever` conda env:

    python scripts/build_ivf_index.py \
        --flat_index retrieval/e5_Flat_2M.index \
        --out_index  retrieval/e5_IVF4096_Flat_2M.index \
        --nlist 4096

    # optional: add PQ compression (much smaller + faster, lower recall)
    python scripts/build_ivf_index.py ... --pq 64

Then restart the retriever pointing at the new index (and keep `--nprobe` in
the retriever >= the value used here).
"""
import argparse
import time

import faiss
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flat_index", required=True)
    p.add_argument("--out_index", required=True)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--pq", type=int, default=0,
                   help="PQ subquantizers (e.g. 64); 0 = IVF,Flat")
    p.add_argument("--nprobe", type=int, default=32,
                   help="nprobe to set on the built index")
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="fraction of vectors to train on (1.0 = all)")
    args = p.parse_args()

    src = faiss.read_index(args.flat_index)
    d = src.d
    n = src.ntotal
    metric = src.metric_type
    metric_name = {faiss.METRIC_INNER_PRODUCT: "IP",
                   faiss.METRIC_L2: "L2"}.get(metric, metric)
    print(f"source: ntotal={n} dim={d} metric={metric_name}")

    # Reconstruct all vectors (2M x 768 x 4B = 6.1 GB)
    print(f"reconstructing {n} vectors ...")
    vecs = np.empty((n, d), dtype=np.float32)
    src.reconstruct_n(0, n, vecs)

    # Build IVF index (matching the flat index's metric, usually IP for e5)
    if args.pq:
        desc = f"IVF{args.nlist},PQ{args.pq}"
    else:
        desc = f"IVF{args.nlist},Flat"
    index = faiss.index_factory(d, desc, metric)

    # Train on a subsample (IVF needs >= nlist*~39 training vectors)
    n_train = int(n * args.train_frac)
    n_train = max(n_train, args.nlist * 40)
    n_train = min(n_train, n)
    rs = np.random.RandomState(0)
    train_idx = rs.choice(n, n_train, replace=False)
    print(f"training on {n_train} vectors ...")
    index.train(vecs[train_idx])

    print(f"adding {n} vectors ...")
    index.add(vecs)

    try:
        index.nprobe = args.nprobe
    except AttributeError:
        pass

    faiss.write_index(index, args.out_index)
    print(f"wrote {args.out_index}")

    # Quick sanity timing on 32 queries (flat was ~260s)
    xq = vecs[:32].copy()
    t = time.time()
    index.search(xq, 3)
    dt = time.time() - t
    print(f"test: 32-query search = {dt*1000:.1f} ms  (flat ~260000 ms)")


if __name__ == "__main__":
    main()
