"""Copy the ChatML chat_template from the teacher (Instruct) tokenizer to the base model.

Why this is needed:

- The multi-hop QA reward function (`verl/utils/reward_score/qa_em_format.py`,
  `is_valid_sequence`) requires the decoded `prompt + response` to contain the
  `<|im_start|>assistant` marker, i.e. the prompt must be ChatML-wrapped.
- `verl/utils/dataset/rl_dataset.py` only wraps the prompt in ChatML when
  `tokenizer.chat_template` is set; otherwise it uses the raw instruction text.
- Qwen2.5 **base** models ship without a `chat_template`, so RL would feed a raw
  (non-ChatML) prompt -> the format reward would always fail.
- The SFT warmup demos (`scripts/make_sft_data.py`) were generated with the
  teacher's `apply_chat_template`, i.e. they ARE ChatML-wrapped and end in the
  assistant marker. Copying the teacher's template makes RL apply the exact same
  wrapper, so the SFT format transfers and the reward parses correctly.

Run once after downloading the models (this is also wired into setup/download_data.sh):

    python scripts/add_chat_template.py \
        --base models/Qwen2.5-0.5B \
        --teacher models/Qwen2.5-7B-Instruct
"""

import argparse

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to the base model dir (e.g. models/Qwen2.5-0.5B).")
    parser.add_argument("--teacher", required=True, help="Path to the teacher model dir (e.g. models/Qwen2.5-7B-Instruct).")
    args = parser.parse_args()

    teacher = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if not getattr(teacher, "chat_template", None):
        raise SystemExit(f"No chat_template found in teacher {args.teacher}; nothing to copy.")

    base = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base.chat_template = teacher.chat_template
    base.save_pretrained(args.base)

    # Sanity check: the template must emit the ChatML assistant marker the reward parser expects.
    probe = base.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False
    )
    ok = "<|im_start|>assistant" in probe
    print(f"Copied chat_template from {args.teacher} -> {args.base}")
    print(f"Probe ends with assistant marker: {ok}")
    print(f"Probe: {probe!r}")
    if not ok:
        raise SystemExit("Probe did not contain <|im_start|>assistant — reward format check will fail.")


if __name__ == "__main__":
    main()
