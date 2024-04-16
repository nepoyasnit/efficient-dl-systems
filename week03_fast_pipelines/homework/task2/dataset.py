from typing import Optional
from collections import defaultdict
import pardata
import torchtext
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


MAX_LENGTH = 640
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def yield_tokens(data_iter):
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    for text, label in data_iter:
        yield tokenizer(text)


class BrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        self.data_path = data_path
        self.max_length = max_length
        self.lines = self._read_lines(data_path, tokenizer, max_length)

    def __getitem__(self, idx: int):
        return self.lines[idx]
    
    def __len__(self):
        return len(self.lines)
    
    @staticmethod
    def _read_lines(data_path, tokenizer, max_length):
        with open(data_path, encoding='utf-8') as f:
            lines = list(filter(len, (tokenizer(line.rstrip())[:max_length] for line in f.readlines())))

        return lines


print(BrainDataset(data_path='data/wikitext-103/wiki.train.tokens', tokenizer=get_tokenizer('basic_english'))[3])

class BigBrainDataset(BrainDataset):
    pass


class UltraDuperBigBrainDataset(BigBrainDataset):
    pass


def collate_fn(
    batch: list[tuple[str, torch.Tensor]], token_to_id: dict[str, int],  max_length: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    result = []
    batch_max_len = max(map(len, batch))
    for line in batch:
        pad_idx = token_to_id[PAD_TOKEN]
        if max_length: 
            padding = [pad_idx] * (max_length - len(line))
        else:
            pading = [pad_idx] * (batch_max_len - len(line))

        unkn_idx = token_to_id[UNK_TOKEN]
        ids = [token_to_id.get(token, unkn_idx) for token in line] + padding
        result.append(ids)

    batch = torch.tensor(result)
    target = batch[:, 1:]

    return batch, target



class UltraDuperBigBrainBatchSampler(Sampler):
    batch_size: int
    bin_size: int
    _dataset_size: int
    _indices: list

    def __init__(self, batch_size: int, dataset: UltraDuperBigBrainDataset, bin_size: int):
        self.batch_size = batch_size
        self.bin_size = bin_size
        self._dataset_size = len(dataset)
        self._indices = list(range(self._dataset_size))

        self._idx_to_sample_len = dict(enumerate(map(len, dataset.lines)))
        self._len_to_indices = self._get_len_to_indices(self._idx_to_sample_len)
        random.shuffle(self._indices)

    def __iter__(self):
        for sample_idx in self._indices:
            sample_len = self._idx_to_sample_len[sample_idx]
            yield self._collect_batch_indices(sample_idx, sample_len)

    def _collect_batch_indices(self, sample_idx, sample_len):
        batch_indices = [sample_idx]
        possible_lens = list(range(sample_len - self.bin_size // 2, sample_len + self.bin_size // 2 + 1))
        random.shuffle(possible_lens)

        for len in possible_lens:
            texts = self._len_to_indices.get(len, [])
            batch_indices += random.sample(texts, k=min(self.batch_size, len(texts)))

            if len(batch_indices) >= self.batch_size:
                return batch_indices[:self.batch_size]
            
        return batch_indices

    @staticmethod
    def _get_len_to_indices(idx_to_sample_len):
        len_to_indices = defaultdict(list)
        for sample_idx, sample_len in idx_to_sample_len.items():
            len_to_indices[sample_len].append(sample_idx)

    def __len__(self):
        return self._dataset_size // self.batch_size
    
