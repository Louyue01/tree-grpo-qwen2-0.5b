#!/usr/bin/env python3
"""Filter the full wiki-18 corpus to the passages the benchmarks actually reference, plus filler.

Why: the full 20M-passage corpus + 60 GB flat index is too big/slow for a single 24 GB GPU
box, but the old "first 2M lines" shrink dropped ~80% of the gold evidence (see
check_coverage.py). This instead keeps every passage whose Wikipedia title any benchmark
references — gold supporting facts AND context/distractor titles — which restores ~100%
gold coverage, plus a configurable amount of unrelated filler so retrieval still has
realistic distractors.

Two phases:
  1. Build the referenced-title set from the four FlashRAG benchmarks (all splits, so
     HotpotQA's train split is covered too).
  2. Stream the corpus; keep every referenced line, plus every K-th unreferenced line as filler.

Writes `--out_corpus` (kept lines, in stream order, same JSON shape as wiki-18.jsonl). The
kept corpus is then embedded by embed_build_index.py — no need to touch the 60 GB index.

Run inside `retriever` (has `datasets`) or `treegrpo`:

    python scripts/filter_corpus_by_title.py \
        --corpus retrieval/wiki-18.jsonl \
        --out_corpus retrieval/wiki-18-keep.jsonl \
        --filler_every 8
"""
import argparse
import json
import unicodedata

import datasets

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"
BENCHMARKS = ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    for ch in ("–", "—", "‐", "−"):  # en/em dash, hyphen, minus
        s = s.replace(ch, "-")
    s = s.replace("_", " ")
    return " ".join(s.split())


def corpus_title(contents):
    """wiki-18 'contents' = '"Title"\\nTitle text ...'; title is the first line, quoted."""
    if not isinstance(contents, str):
        return None
    first = contents.split("\n", 1)[0].strip().strip('"').strip()
    return first or None


def collect_titles(obj):
    """Recursively grab every string under a key literally named 'title'.

    Each benchmark stores evidence differently (supporting_facts.title, context.title,
    paragraphs[].title, ...); grabbing every 'title' key makes this benchmark-agnostic.
    In FlashRAG only Wikipedia article titles are keyed 'title', so this is safe.
    """
    out = []
    if isinstance(obj, dict):
        tv = obj.get("title")
        if isinstance(tv, str):
            out.append(tv)
        elif isinstance(tv, list):
            out += [t for t in tv if isinstance(t, str)]
        for v in obj.values():
            out += collect_titles(v)
    elif isinstance(obj, list):
        for v in obj:
            out += collect_titles(v)
    return out


def build_referenced_titles(max_q_per_bench=0):
    keep = set()
    for bench in BENCHMARKS:
        try:
            ds = datasets.load_dataset(FLASHRAG_REPO, bench)
        except Exception as e:
            print(f"[{bench}] load failed: {e}")
            continue
        for split_name, d in ds.items():
            n = len(d)
            if max_q_per_bench and n > max_q_per_bench:
                d = d.select(range(max_q_per_bench))
            titles = set()
            for ex in d:
                for t in collect_titles(ex):
                    if isinstance(t, str) and t.strip():
                        titles.add(norm(t))
            print(f"[{bench}/{split_name}] {len(d)}/{n} examples -> {len(titles)} titles")
            keep |= titles
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out_corpus", required=True)
    p.add_argument("--filler_every", type=int, default=8,
                   help="keep every K-th unreferenced line as filler (distractors); 0 = none")
    p.add_argument("--max_q_per_bench", type=int, default=0,
                   help="cap questions per benchmark split (0 = all); use for a smoke test")
    args = p.parse_args()

    keep = build_referenced_titles(args.max_q_per_bench)
    print(f"[titles] total referenced titles: {len(keep)}")

    n_total = n_kept = n_ref = n_filler = 0
    filler_seen = 0
    with open(args.corpus, "rb") as fin, open(args.out_corpus, "wb") as fout:
        for line in fin:
            keep_this = False
            is_ref = False
            raw = line.rstrip(b"\r\n")
            if raw:
                try:
                    obj = json.loads(raw.decode("utf-8", errors="ignore"))
                    t = corpus_title(obj.get("contents")) if isinstance(obj.get("contents"), str) else None
                    if t and norm(t) in keep:
                        keep_this = True
                        is_ref = True
                except json.JSONDecodeError:
                    pass
            if not keep_this and args.filler_every:
                if filler_seen % args.filler_every == 0:
                    keep_this = True
                filler_seen += 1
            if keep_this:
                fout.write(line)  # verbatim bytes (corpus may not be pure UTF-8)
                n_kept += 1
                if is_ref:
                    n_ref += 1
                else:
                    n_filler += 1
            n_total += 1
            if n_total % 2000000 == 0:
                print(f"[corpus] {n_total} lines, kept {n_kept} (ref {n_ref} + filler {n_filler})")

    print(f"[done] {n_total} lines -> kept {n_kept} (referenced {n_ref} + filler {n_filler})")
    print(f"wrote {args.out_corpus}")


if __name__ == "__main__":
    main()
