#!/usr/bin/env python3
"""Check how much of each benchmark's gold evidence is inside the retrieval corpus.

For each multi-hop QA benchmark (HotpotQA / 2WikiMultiHopQA / MuSiQue / Bamboogle),
load its FlashRAG test/dev split (the same split that produced test.parquet), pull
the Wikipedia titles of its gold passages, and report what fraction of those titles
exist in the retrieval corpus (wiki-18-2M.jsonl).

A title missing from the corpus means that evidence is unreachable no matter how good
the retriever is — it sets a hard ceiling on the reward the RL can ever observe.

No GPU needed. The only slow part is one streaming pass over the 2M-line corpus file.

    python scripts/check_coverage.py --corpus_path retrieval/wiki-18-2M.jsonl [--max_n 200]
"""
import argparse
import json
import sys

import datasets

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"
BENCHMARKS = ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def norm(s):
    return " ".join(str(s).lower().replace("_", " ").split())


def detect_title_key(obj):
    for cand in ("title", "id", "contents"):
        if cand in obj:
            return cand
    return None


def load_corpus_titles(path):
    titles = set()
    title_key = None
    samples = []
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if title_key is None:
                title_key = detect_title_key(obj)
                if title_key is None:
                    print("ERROR: no title/id/contents key; first line keys =",
                          list(obj.keys()))
                    sys.exit(1)
                print(f"[corpus] title field = {title_key!r}  "
                      f"(first line keys = {list(obj.keys())})")
            t = obj.get(title_key)
            if isinstance(t, str) and t.strip():
                titles.add(norm(t))
                if len(samples) < 3:
                    samples.append(t)
            n += 1
            if n % 500000 == 0:
                print(f"[corpus] read {n} lines, {len(titles)} unique titles")
    print(f"[corpus] done: {n} lines, {len(titles)} unique titles")
    print(f"[corpus] sample titles: {samples}")
    return titles


def golden_titles(example):
    for field in ("golden_passages", "golden_articles", "golden_docs"):
        gps = example.get(field)
        if not gps:
            continue
        out = []
        for gp in gps:
            if isinstance(gp, dict) and gp.get("title"):
                out.append(norm(gp["title"]))
        if out:
            return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--max_n", type=int, default=0,
                    help="max questions per benchmark, 0 = all")
    args = ap.parse_args()

    corpus_titles = load_corpus_titles(args.corpus_path)
    print()

    print(f"{'benchmark':<16} {'n_q':>5} {'title_cov':>10} {'q_full':>7} {'q_any':>7}")
    for bench in BENCHMARKS:
        try:
            ds = datasets.load_dataset(FLASHRAG_REPO, bench)
        except Exception as e:
            print(f"[{bench}] load failed: {e}")
            continue
        split = "test" if "test" in ds else ("dev" if "dev" in ds else "train")
        d = ds[split]
        n_total = len(d)
        if args.max_n and n_total > args.max_n:
            d = d.select(range(args.max_n))

        if bench == BENCHMARKS[0]:
            print(f"[{bench}] example keys = {list(d[0].keys())}")

        n_q = title_hit = title_total = 0
        q_full = q_any = 0
        gold_samples = []
        for ex in d:
            ts = golden_titles(ex)
            if not ts:
                continue
            n_q += 1
            in_corpus = [t in corpus_titles for t in ts]
            title_hit += sum(in_corpus)
            title_total += len(ts)
            if all(in_corpus):
                q_full += 1
            if any(in_corpus):
                q_any += 1
            if not gold_samples:
                gold_samples = ts[:3]

        if n_q == 0:
            print(f"[{bench}] split={split} n={n_total} -> no gold titles found")
            continue

        print(f"{bench:<16} {n_q:>5} {title_hit / title_total:>9.1%} "
              f"{q_full / n_q:>6.1%} {q_any / n_q:>6.1%}")
        print(f"           (split={split}, total={n_total}, "
              f"sample gold titles={gold_samples})")

    print()
    print("title_cov = fraction of gold (question,title) pairs found in the corpus")
    print("q_full    = fraction of questions with ALL gold titles present")
    print("q_any     = fraction of questions with at least ONE gold title present")


if __name__ == "__main__":
    main()
