#!/usr/bin/env python
"""Shrink the wiki-18 FAISS index + corpus to the first N passages.

The full e5_Flat.index is ~20M x 768 (60 GB); exact search on CPU is the
training bottleneck (~6 s per single query). Keeping only the first N passages
shrinks the index and corpus by the same amount while preserving row alignment
(index row i == corpus line i), so the retriever's `load_docs(corpus, idxs)`
keeps working unchanged.

Run inside the `retriever` conda env (it has faiss):

    python scripts/shrink_index.py \
        --index_path retrieval/e5_Flat.index \
        --corpus_path retrieval/wiki-18.jsonl \
        --n 2000000 \
        --out_index retrieval/e5_Flat_2M.index \
        --out_corpus retrieval/wiki-18-2M.jsonl

Then restart the retriever pointing at the _2M files.
"""
import argparse

import faiss
import numpy as np


def shrink_index(index_path, n, out_index):
    print(f"reading {index_path} ...")
    src = faiss.read_index(index_path)
    d = src.d
    metric = src.metric_type
    total = src.ntotal
    keep = min(n, total)
    print(f"source: ntotal={total} dim={d} metric={metric} -> keeping first {keep}")

    vecs = np.empty((keep, d), dtype=np.float32)
    print(f"reconstructing first {keep} vectors ({keep * d * 4 / 1e9:.1f} GB) ...")
    src.reconstruct_n(0, keep, vecs)

    if metric == faiss.METRIC_INNER_PRODUCT:
        dst = faiss.IndexFlatIP(d)
    else:
        dst = faiss.IndexFlatL2(d)
    print("building new index ...")
    dst.add(vecs)
    faiss.write_index(dst, out_index)
    print(f"wrote {out_index} ({keep} x {d})")


def shrink_corpus(corpus_path, n, out_corpus):
    print(f"truncating {corpus_path} to first {n} lines ...")
    written = 0
    # Copy raw bytes (no decode/re-encode): the corpus may not be pure UTF-8, and
    # we only need the first N lines verbatim so index row i == corpus line i.
    with open(corpus_path, "rb") as fin, \
            open(out_corpus, "wb") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
            written = i + 1
    print(f"wrote {out_corpus} ({written} lines)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index_path", required=True)
    p.add_argument("--corpus_path", required=True)
    p.add_argument("--n", type=int, required=True,
                   help="number of passages (and index rows) to keep")
    p.add_argument("--out_index", required=True)
    p.add_argument("--out_corpus", required=True)
    args = p.parse_args()

    shrink_index(args.index_path, args.n, args.out_index)
    shrink_corpus(args.corpus_path, args.n, args.out_corpus)
    print("done.")


if __name__ == "__main__":
    main()
