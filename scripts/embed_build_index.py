#!/usr/bin/env python3
"""Embed a corpus jsonl with e5-base-v2 and build an exact FAISS Flat (inner-product) index.

Embeds the `contents` field of each line (the same text the dense retriever returns), in the
SAME vector space the retrieval server's query encoder expects: e5-base-v2, "passage: "
prefix, mean pooling over last_hidden_state, L2-normalized, float32, searched by inner
product (cosine). This mirrors search_r1/search/retrieval_server.py Encoder.encode(is_query=False).

Run inside `retriever` (has faiss + transformers + torch), on a GPU:

    python scripts/embed_build_index.py \
        --corpus retrieval/wiki-18-keep.jsonl \
        --out_index retrieval/e5_Flat_keep.index \
        --model /workspace/Tree-GRPO/models/e5-base-v2 \
        --batch_size 512
"""
import argparse
import json

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_encoder(model_path):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tok


@torch.no_grad()
def encode_passages(model, tok, texts, max_length=512):
    """e5 passage encoding: 'passage: ' prefix, mean pool, L2-normalize -> float32."""
    texts = [f"passage: {t}" for t in texts]
    enc = tok(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}
    out = model(**enc, return_dict=True)
    last_hidden = out.last_hidden_state                       # [B, L, H]
    mask = enc["attention_mask"].unsqueeze(-1).bool()         # [B, L, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1)                     # [B, 1]
    emb = summed / counts
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.detach().cpu().numpy().astype(np.float32)


def iter_contents(corpus_path):
    with open(corpus_path, "rb") as f:
        for line in f:
            raw = line.rstrip(b"\r\n")
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            c = obj.get("contents")
            if isinstance(c, str):
                yield c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out_index", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_passages", type=int, default=0,
                   help="cap passages embedded (0 = all); use for a smoke test")
    args = p.parse_args()

    model, tok = load_encoder(args.model)

    batch = []
    index = None
    n = 0
    for contents in iter_contents(args.corpus):
        batch.append(contents)
        if len(batch) == args.batch_size:
            emb = encode_passages(model, tok, batch, max_length=args.max_length)
            if index is None:
                index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            n += len(batch)
            batch = []
            if n % 200000 == 0:
                print(f"[embed] {n} passages")
        if args.max_passages and n >= args.max_passages:
            break
    if batch:
        emb = encode_passages(model, tok, batch, max_length=args.max_length)
        if index is None:
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        n += len(batch)

    faiss.write_index(index, args.out_index)
    print(f"[done] wrote {args.out_index} ({n} x {index.d})")


if __name__ == "__main__":
    main()
