import logging
from dataclasses import dataclass
from pprint import pprint, pformat

from fastai.basic_data import DataBunch
from fastai.layers import FlattenedLoss, CrossEntropyFlat
from fastai.text import LMLabelList, Vocab, TextList
from pathlib import Path
from typing import Sequence

from langmodels.profiling import get_cpu_memory_used_mb
from langmodels.tensor_ops import contains_no_value
from langmodels.training.numericalize import Numericalizer

logger = logging.getLogger(__name__)


UNKNOWN_TOKEN_INDEX = 0


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


def check_data(data_bunch: DataBunch, vocab: Vocab, verbose: bool) -> None:
    first_batch = data_bunch.one_batch()[0]

    if not contains_no_value(first_batch, UNKNOWN_TOKEN_INDEX):
        raise ValueError(f"Unknown is found : {[vocab.textify(seq) for seq in first_batch]}")
    if verbose:
        logger.info(f'Displaying the first batch:\n{first_batch}')
        token_seqs = [vocab.textify(seq) for seq in first_batch]
        logger.info(pformat(token_seqs))
