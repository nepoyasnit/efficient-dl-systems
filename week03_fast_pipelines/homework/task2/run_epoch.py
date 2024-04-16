from enum import Enum
import itertools
import statistics
import json
from collections import Counter
from enum import Enum
from functools import partial
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

from dataset import (
    BrainDataset,
    BigBrainDataset,
    UltraDuperBigBrainDataset,
    UltraDuperBigBrainBatchSampler,
    collate_fn,
    PAD_TOKEN,
    UNK_TOKEN,
)
from transformer import TransformerModel, generate_square_subsequent_mask


N_BATCHES = 800


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model(ntoken: int) -> torch.nn.Module:
    return TransformerModel(
        ntoken=ntoken,
        d_model=1024, 
        nhead=8,
        d_hid=2048,
        nlayers=1,
        dropout=0.1,
    )

def get_vocab(dataset, vocab_size: int):
    vocab_path = Path().cwd() / 'vocabulary.json'
    if vocab_path.exists():
        return json.loads(vocab_path.read_text())
    
    vocab = Counter()
    for line in dataset:
        vocab.update(line)

    word_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, (word, _) in enumerate(vocab.most_common(vocab_size), start=2):
        word_to_id[word] = i

    vocab_path.write_text(json.dumps(word_to_id))

    return word_to_id


def get_batch_time(batch_time):
    return (
        f'Min time: {min(batch_time):.3f}'
        f'Max time: {max(batch_time):.3f}'
        f'Mean time: {statistics.mean(batch_time):.3f}'
        f'Median time: {statistics.median(batch_time):.3f}'
    )


def run_warmup_batch(model, dataloader, device):
    warmup_batch, _ = next(iter(dataloader))
    warmup_batch = warmup_batch.to(device)
    mask = generate_square_subsequent_mask(warmup_batch.shape[0]).to(device)
    model(warmup_batch, mask)


def measure_batch_time(model: any, dataloader: DataLoader, device: torch.device, n_batches: int = N_BATCHES):
    pbar = tqdm(itertools.islice(dataloader, n_batches), total=n_batches)
    batch_time = []
    batch_lens = []

    for batch, target in pbar:
        start_time = time.perf_counter()
        
        batch = batch.to(device).transpose(1, 0)
        batch_lens.append(batch.shape[0])
        target = target.to(device)

        mask = generate_square_subsequent_mask(batch.shape[0]).to(device)
        model(batch, mask)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        batch_time.append(end_time - start_time)

        pbar.set_description(get_batch_time(batch_time))

    print(statistics.mean(batch_lens), min(batch_lens), max(batch_lens))
    return batch_time

def run_epoch(data_mode: DataMode) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer('basic_english')

    match data_mode:
        case DataMode.BRAIN:
            dataset = BrainDataset(dataset='data/wikitext-103/wiki.train.tokens', tokenizer=tokenizer)
            token_to_id = get_vocab(dataset, vocab_size=2048)
            collate_fn_partial = partial(collate_fn, max_length=dataset.max_length, token_to_id=token_to_id)
            
            dataloader = DataLoader(dataset=dataset, batch_size=128, collate_fn=collate_fn_partial, pin_memory=True)

        case DataMode.BIG_BRAIN:
            dataset = BigBrainDataset(dataset='data/wikitext-103/wiki.train.tokens', tokenizer=tokenizer)
            token_to_id = get_vocab(dataset, vocab_size=2048)
            collate_fn_partial = partial(collate_fn, token_to_id=token_to_id)

            dataloader = DataLoader(dataset=dataset, batch_size=128, collate_fn=collate_fn_partial, pin_memory=True)

        case DataMode.ULTRA_DUPER_BIG_BRAIN:
            dataset = UltraDuperBigBrainDataset(
                data_path='data/wikitext-103/wiki.train.tokens', tokenizer=tokenizer
            )
            token_to_id = get_vocab(dataset, vocab_size=2048)
            collate_fn_partial = partial(collate_fn, token_to_id=token_to_id)

            sampler = UltraDuperBigBrainBatchSampler(batch_size=128, dataset=dataset, bin_size=5)
            dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn_partial, pin_memory=True, batch_sampler=sampler)
        case _:
            raise NotImplementedError(f'Data mode "{data_mode}" is not supported!')
        
    model = get_gpt2_model(ntoken=len(token_to_id)).to(device)
    model.train()

    run_warmup_batch(model, dataloader, device)
    measure_batch_time(model, dataloader, device)


if __name__ == '__main__':
    torch.cuda.manual_seed(0)

    run_epoch(DataMode.ULTRA_DUPER_BIG_BRAIN)
