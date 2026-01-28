# dataset.py

import torch
from torch.utils.data import IterableDataset
import tiktoken


class StreamingLanguageModelDataset(IterableDataset):
    def __init__(self, iterable_ds, seq_len, tokenizer_name="cl100k_base"):
        super().__init__()
        self.iterable_ds = iterable_ds
        self.seq_len = seq_len
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def __iter__(self):
        token_buffer = []

        for item in self.iterable_ds:
            text = item["text"]
            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1 :]

                chunk = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": chunk[:-1],
                    "targets": chunk[1:]
                }
