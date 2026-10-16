"""Generate SFT format-warmup data for the ReAct (search-agent) format.

A strong teacher model runs a ReAct loop over sampled HotpotQA questions: it emits
`<think>...</think><search>q</search>` (then the retriever injects `<information>...</information>`)
or `<think>...</think><answer>X</answer>`. Every (context, next-action) pair is stored as one
SFT row, so the small policy model learns the *turn-level* format without ever being asked to
generate the `<information>` block (that stays in the prompt, i.e. masked from the SFT loss).

Usage:
    python scripts/make_sft_data.py --teacher Qwen/Qwen2.5-7B-Instruct \
        --n_samples 1000 --out sft.parquet --retriever_url http://127.0.0.1:8000/retrieve
"""

import argparse
import re

import datasets
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def make_prefix(question: str) -> str:
    """Identical instruction template to scripts/data_process/qa_search_train_merge.py."""
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )


def build_initial_prompt(tokenizer, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def truncate_action(text: str) -> str:
    """Cut the generation at the first closing </search> or </answer> tag (inclusive)."""
    for tag in ("</search>", "</answer>"):
        idx = text.find(tag)
        if idx != -1:
            return text[: idx + len(tag)]
    return text


def extract_action(text: str):
    match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, ""


def passages2string(retrieval_result) -> str:
    if not retrieval_result:
        return ""
    parts = []
    for idx, doc_item in enumerate(retrieval_result):
        if isinstance(doc_item, dict) and "document" in doc_item:
            content = doc_item["document"]["contents"]
        elif isinstance(doc_item, dict) and "contents" in doc_item:
            content = doc_item["contents"]
        else:
            content = str(doc_item)
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        parts.append(f"Doc {idx+1}(Title: {title}) {text}")
    return "\n".join(parts)


def batch_retrieve(queries, url: str, topk: int):
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    resp = requests.post(url, json=payload, timeout=60).json()
    return [passages2string(r) for r in resp["result"]]


@torch.no_grad()
def generate(model, tokenizer, context: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--flashrag_dir", default="RUC-NLPIR/FlashRAG_datasets")
    parser.add_argument("--data_source", default="hotpotqa")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--out", default="sft.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    ds = datasets.load_dataset(args.flashrag_dir, args.data_source)["train"]
    ds = ds.shuffle(seed=args.seed).select(range(args.n_samples))

    rows = []
    for qi, example in enumerate(tqdm(ds, desc="teacher rollout", unit="q")):
        question = example["question"].strip()
        if not question.endswith("?"):
            question += "?"
        instruction = make_prefix(question)
        context = build_initial_prompt(tokenizer, instruction)

        for _ in range(args.max_turns):
            raw = generate(model, tokenizer, context, args.max_new_tokens, args.temperature, args.top_p)
            action_block = truncate_action(raw)
            action, content = extract_action(action_block)
            if action is None:
                # Invalid format from the teacher; drop the rest of this trajectory.
                break

            rows.append({"prompt": context, "response": action_block})

            if action == "answer":
                break

            # search -> inject the information block (environment side).
            retrieved = batch_retrieve([content], args.retriever_url, args.topk)[0]
            context = context + action_block + f"\n\n<information>{retrieved.strip()}</information>\n\n"

        if (qi + 1) % 50 == 0:
            print(f"  [{qi + 1}/{args.n_samples}] {len(rows)} turn samples so far", flush=True)

    if not rows:
        raise RuntimeError("No valid SFT samples generated. Check teacher format and retriever.")

    pd.DataFrame(rows).to_parquet(args.out)
    print(f"Saved {len(rows)} SFT turn-level samples to {args.out}")


if __name__ == "__main__":
    main()
