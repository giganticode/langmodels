import logging
import os
import typing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from typing import Collection, List, Tuple

import numpy as np
from dataprep.util import merge_dicts_
from fastai.core import partition_by_cores
from fastai.data_block import PreProcessor
from fastai.text import Vocab
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Numericalizer(PreProcessor):
    def __init__(self, vocab: Vocab, n_cpus: int = os.cpu_count(),
                 allow_unks: bool = False, large_databunch: bool = False):
        super().__init__()
        self.vocab = vocab
        self.n_cpus = n_cpus
        self.allow_unks = allow_unks
        self.large_databunch = large_databunch

    def process_one(self, item: str):
        with open(item, 'r') as f:
            prep_tokens = [token for line in f for token in line.rstrip('\n').split(' ')]
            unks = Counter()
            for prep_token in prep_tokens:
                if prep_token not in self.vocab.itos:
                    if self.allow_unks:
                        unks.update([prep_token])
                    else:
                        raise ValueError(f'{[prep_token]} is not present in the vocabulary.\n'
                                         f'Vocab size is {len(self.vocab.itos)}. Vocab is {self.vocab.itos}')
            numericalized_tokens = np.array(self.vocab.numericalize(prep_tokens), dtype=np.int64)
            return numericalized_tokens, unks

    def process(self, ds: Collection) -> None:
        ds.vocab = self.vocab
        tokens, unks = self._process_in_parallel(ds.items)
        ds.items = np.array(tokens)
        if self.allow_unks:
            logger.warning(f"Encountered the following unknown tokens "
                           f"{sorted(unks.items(), reverse=True, key=lambda x:x[1])}")

    def _process_on_one_core(self, items: Collection[str]) -> List[Tuple[np.ndarray, typing.Mapping[str, int]]]:
        items = tqdm(items) if self.large_databunch else items
        return [self.process_one(item) for item in items]

    def _process_in_parallel(self, texts: Collection[str]) -> Tuple[List[np.ndarray], typing.Mapping[str, int]]:
        if self.n_cpus <= 1:
            if len(texts) == 0:
                return [[]], Counter()
            all_tokens, unk_list = zip(*self._process_on_one_core(texts))
            return list(all_tokens), reduce(lambda x, y: merge_dicts_(x, y)[0], unk_list, {})
        with ProcessPoolExecutor(self.n_cpus) as e:
            all_tokens = []
            all_unks = {}
            for from_one_core in e.map(self._process_on_one_core, partition_by_cores(texts, self.n_cpus)):
                for tokens, unks in from_one_core:
                    all_tokens.append(tokens)
                    merge_dicts_(all_unks, unks)
            return all_tokens, all_unks
