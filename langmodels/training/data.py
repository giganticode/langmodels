import os
import typing
from collections import Counter
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from functools import reduce
from pprint import pformat

import logging

import numpy as np
from codeprep.util import merge_dicts_
from fastai.basic_data import DataBunch
from fastai.core import partition_by_cores
from fastai.data_block import PreProcessor
from fastai.layers import FlattenedLoss, CrossEntropyFlat
from fastai.text import LMLabelList, Vocab, TextList
from pathlib import Path
from typing import Sequence, Collection, List, Tuple

from tqdm import tqdm

from langmodels.profiling import get_cpu_memory_used_mb
from langmodels.tensor_ops import contains_no_value


logger = logging.getLogger(__name__)


UNKNOWN_TOKEN_INDEX = 0


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


@dataclass
class EmptyDataBunch(object):
    class EmptyTrainDS(object):
        y: LMLabelList = None

    class EmptyTrainDL(object):
        dataset = None

        def __len__(self):
            return 1
    vocab: Vocab
    path: str
    device: str
    backwards: bool = False
    loss_func: FlattenedLoss = CrossEntropyFlat()
    train_ds: EmptyTrainDS = EmptyTrainDS()
    train_dl: EmptyTrainDL = EmptyTrainDL()


def create_databunch(path_to_prep_dataset: str, file_paths: Sequence[Path], vocab: Vocab,
                     bs: int, bptt: int, device: str,
                     only_validation_files: bool = False, allow_unks: bool = False, verbose: bool = True) -> DataBunch:
    if verbose:
        logger.info(f'Getting preprocessed corpus from {path_to_prep_dataset}')
    numericalizer = Numericalizer(vocab, allow_unks=allow_unks, large_databunch=verbose)
    text_list = TextList(file_paths, path=path_to_prep_dataset, processor=numericalizer)

    if verbose:
        logger.info("Splitting into training/validation sets")
    if only_validation_files:
        split_list = text_list.split_by_valid_func(lambda f: True)
    else:
        split_list = text_list.split_by_folder()

    if verbose:
        logger.info("Labeling for langmodeling")
    labelled_list = split_list.label_for_lm()

    if verbose:
        cpu_memory_used_mb = get_cpu_memory_used_mb()
        logger.debug(f"Cpu memory used: {cpu_memory_used_mb} MB")

    if verbose:
        logger.info("Creating data bunches")
    data_bunched = labelled_list.databunch(bs=bs, bptt=bptt, device=device)
    return data_bunched


def check_data(data_bunch: DataBunch, vocab: Vocab, verbose: bool, allow_unks: bool) -> None:
    first_batch = data_bunch.one_batch()[0]

    if not allow_unks and not contains_no_value(first_batch, UNKNOWN_TOKEN_INDEX):
        raise ValueError(f"Unknown is found : {[vocab.textify(seq) for seq in first_batch]}")
    if verbose:
        logger.info(f'Displaying the first batch:\n{first_batch}')
        token_seqs = [vocab.textify(seq) for seq in first_batch]
        logger.info(pformat(token_seqs))
