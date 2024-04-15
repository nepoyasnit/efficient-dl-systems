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

def yield_tokens(data_iter):
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    for text, label in data_iter:
        yield tokenizer(text)


class BrainDataset(Dataset):
    dataset: tuple
    def __init__(self, max_length: int = MAX_LENGTH):
        data = pardata.load_dataset('wikitext103')['train'].split('.')
        labels = torch.randint(2, (len(data),))
        self.dataset = list(zip(data, labels))

        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(yield_tokens(iter(self.dataset)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        text_pipeline = lambda x: vocab(tokenizer(x))

        torch.tensor(text_pipeline(x), dtype=torch.int64)
        print(len(self.data))

    def __getitem__(self, idx: int):
        return self.data[idx]

print(BrainDataset()[4])

class BigBrainDataset(Dataset):
    dataset: tuple

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        data = pardata.load_dataset('wikitext103')['train'].split('.')
        labels = torch.randint(2, (len(data),))
        self.dataset = list(zip(data, labels))
        
        vocab = build_vocab_from_iterator(yield_tokens(iter(self.dataset)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        


    def __getitem__(self, idx: int):
        return self.dataset[idx]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


def collate_fn(
    batch: list[tuple[str, torch.Tensor]], vocab,  max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    text_list, label_list = [], []
    tokenizer = get_tokenizer("basic_english")
    text_pipeline = lambda x: vocab(tokenizer(x))

    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)

    return text_list, label_list



class UltraDuperBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
