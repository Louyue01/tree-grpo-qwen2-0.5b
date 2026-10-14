# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A minimal SFT dataset for the FSDP SFT trainer (verl/trainer/fsdp_sft_trainer.py).

It reads a parquet file with a `prompt` column and a `response` column and produces
`input_ids` / `attention_mask` / `position_ids` / `loss_mask`, where the loss is only
computed on the response tokens (the prompt is masked out).

This fills a gap in this fork: `fsdp_sft_trainer.py` imports `SFTDataset` but the class
was never shipped here.
"""

import os
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """Prompt -> response supervised fine-tuning dataset.

    Args:
        parquet_files: path(s) to the parquet file(s).
        tokenizer: huggingface tokenizer.
        prompt_key: column name holding the prompt string.
        response_key: column name holding the response string.
        max_length: every sample is padded/truncated to this length.
        truncation: 'error' to raise on over-long samples, otherwise truncate.
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key: str = 'prompt',
                 response_key: str = 'response',
                 prompt_dict_keys: List[str] = None,
                 response_dict_keys: List[str] = None,
                 max_length: int = 1024,
                 truncation: str = 'error'):
        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.prompt_dict_keys = prompt_dict_keys
        self.response_dict_keys = response_dict_keys
        self.max_length = max_length
        self.truncation = truncation

        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        print(f'SFT dataset loaded: {len(self.dataframe)} samples')

    def __len__(self):
        return len(self.dataframe)

    def _resolve(self, value, dict_keys):
        # Allow either a plain string or a dict (chat message list) column.
        if dict_keys is not None and isinstance(value, dict):
            return {k: value.get(k) for k in dict_keys}
        return value

    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()

        prompt = self._resolve(row_dict.pop(self.prompt_key), self.prompt_dict_keys)
        response = self._resolve(row_dict.pop(self.response_key), self.response_dict_keys)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # Append EOS to the response so the model learns to stop.
        if self.tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != self.tokenizer.eos_token_id):
            response_ids.append(self.tokenizer.eos_token_id)

        # Concatenate and optionally truncate.
        input_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        if len(input_ids) > self.max_length:
            if self.truncation == 'error':
                raise ValueError(f'Sample {item} has length {len(input_ids)} > max_length {self.max_length}. '
                                 f'Set truncation to "left"/"right" or increase max_length.')
            # Truncate from the left so the tail (recent response tokens) is kept.
            input_ids = input_ids[-self.max_length:]
            loss_mask = loss_mask[-self.max_length:]

        attention_mask = [1] * len(input_ids)

        # Pad to exactly max_length so batches are rectangular.
        pad_len = self.max_length - len(input_ids)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = input_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        loss_mask = loss_mask + [0] * pad_len

        # position ids: cumsum of attention mask minus 1 (0 for padding).
        position_ids = torch.clamp(torch.cumsum(torch.tensor(attention_mask, dtype=torch.long), dim=0) - 1, min=0)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'position_ids': position_ids,
            'loss_mask': torch.tensor(loss_mask, dtype=torch.long),
        }
