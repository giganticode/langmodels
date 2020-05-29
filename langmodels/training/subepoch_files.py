import logging
import os
from pathlib import Path
from typing import Any, Optional, List, Tuple

from fastai.basic_train import LearnerCallback, Learner
from fastai.callback import Callback
from fastai.text import Vocab

from codeprep.api.corpus import PreprocessedCorpus
from langmodels.training.data import create_databunch, check_data
from langmodels.util.file import get_all_files

BIG_EPOCH_FILE_LIMIT = 10 * 1000

TRAIN_SUBDIR = 'train'
TEST_SUBDIR = 'test'
VALID_SUBDIR = 'valid'


logger = logging.getLogger(__name__)


class EpochFileLoader(LearnerCallback):
    def __init__(self, learner: Learner, prep_corpus: PreprocessedCorpus,
                 vocab: Vocab, bs: int, bptt: int, device: str, n_files_per_epoch: Optional[int], allow_unks: bool):
        super().__init__(learner)

        if n_files_per_epoch is not None and n_files_per_epoch <= 0:
            raise ValueError(f'The value of n_files_per_epoch should be > 0 or None but is {n_files_per_epoch}')

        self.prep_corpus = prep_corpus
        self.vocab: Vocab = vocab
        self.bs: int = bs
        self.bptt: int = bptt
        self.device: str = device
        self.n_files_per_epoch = n_files_per_epoch
        self.allow_unks = allow_unks

        self.path_to_train_files = os.path.join(self.prep_corpus.path_to_prep_dataset, TRAIN_SUBDIR)
        self.valid_and_test_files = self._load_valid_and_test_files()
        self.train_files_iterator = self._reset_train_file_iterator()

    def _reset_train_file_iterator(self):
        return get_all_files(self.path_to_train_files, None)

    def _handle_empty_training_files_list(self, **kwargs):
        if 'after_empty_epoch' in kwargs:
            raise ValueError("Looks like your training set is empty.")
        else:
            self.on_epoch_begin(after_empty_epoch=True, **kwargs)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        train_files, all_files_read = self._get_next_training_files(self.n_files_per_epoch)
        if all_files_read:
            self.train_files_iterator = self._reset_train_file_iterator()
        if len(train_files) == 0:
            self._handle_empty_training_files_list(**kwargs)
            return
        small_epoch = len(train_files) < BIG_EPOCH_FILE_LIMIT
        if self.n_files_per_epoch is not None:
            train_subfolders = {f.relative_to(self.path_to_train_files).parts[0] for f in train_files}
            logger.info(f"Using projects for training: {','.join(train_subfolders)}")
        databunch = create_databunch(self.prep_corpus.path_to_prep_dataset, self.valid_and_test_files + train_files,
                                     self.vocab, bs=self.bs, bptt=self.bptt,
                                     device=self.device, verbose=not small_epoch, allow_unks=self.allow_unks)
        check_data(databunch, self.vocab, allow_unks=self.allow_unks, verbose=not small_epoch)
        self.learn.data = databunch
        train_dl = self.learn.data.train_dl
        if isinstance(train_dl, Callback):
            train_dl.on_train_begin()

    def _load_valid_and_test_files(self) -> List[Path]:
        path_to_valid_files = os.path.join(self.prep_corpus.path_to_prep_dataset, VALID_SUBDIR)
        valid_files = [f for f in get_all_files(path_to_valid_files, None)]

        path_to_test_files = os.path.join(self.prep_corpus.path_to_prep_dataset, TEST_SUBDIR)
        if os.path.exists(path_to_test_files):
            test_files = [f for f in get_all_files(path_to_test_files, None)]
        else:
            test_files = []

        return valid_files + test_files

    def _get_next_training_files(self, n: Optional[int]) -> Tuple[List[Path], bool]:
        res: List[Path] = []
        for file in self.train_files_iterator:
            res.append(file)
            if n and n == len(res):
                return res, False
        return res, True
