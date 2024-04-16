from typing import Optional
import pardata
import torchtext

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

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
