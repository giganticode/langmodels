import os
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict

import numpy as np
import torch
from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import Vocab, TextList, language_model_learner, AWD_LSTM, PreProcessor, Collection, partition_by_cores, \
    List, awd_lstm_lm_config
from torch import Tensor
from tqdm import tqdm


UNKNOWN_TOKEN_INDEX = 0


class Numericalizer(PreProcessor):
    def __init__(self, vocab: Vocab, n_cpus: int=os.cpu_count()):
        super().__init__()
        self.vocab = vocab
        self.n_cpus = n_cpus

    def process_one(self, item: str):
        with open(item, 'r') as f:
            prep_tokens = [token for line in f for token in line.rstrip('\n').split(' ')]
            return np.array(self.vocab.numericalize(prep_tokens), dtype=np.int64)

    def process(self, ds: Collection):
        ds.vocab = self.vocab
        ds.items = np.array(self._process_in_parallel(ds.items))

    def _process_on_one_core(self, items:Collection[str]) -> List:
        return [self.process_one(item) for item in tqdm(items)]

    def _process_in_parallel(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1:
            return self._process_on_one_core(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_on_one_core, partition_by_cores(texts, self.n_cpus)), [])


def contains_no_value(tensor: Tensor, value: int) -> bool:
    return (tensor == torch.full_like(tensor, value)).float().sum() == 0


def train(prep_corpus: PreprocessedCorpus,
          path_to_model: str, device: Optional[int],
          config: Optional[Dict]=None,
          use_pretrained_model: bool=True):
    vocab = Vocab(['`unk', '`pad'] + list(prep_corpus.load_vocab().keys()))

    print(f'Getting preprocessed corpus from {prep_corpus.path_to_prep_dataset}')
    file_path_list = [Path(os.fsdecode(p)) for p in prep_corpus.get_file_iterator()]
    text_list = TextList(file_path_list, path=prep_corpus.path_to_prep_dataset, processor=Numericalizer(vocab))

    print("Splitting into training/validation sets")
    split_list = text_list.split_by_rand_pct()

    print("Labeling for langmodeling")
    labelled_list = split_list.label_for_lm()

    print("Creating databunches")
    databunched = labelled_list.databunch(bs=16, device=device)

    check_data(databunched, vocab)

    learner = language_model_learner(databunched, AWD_LSTM, drop_mult=0.5, config=config, pretrained=not config)
    if use_pretrained_model and os.path.exists(f'{path_to_model}.pth'):
        print(f"Using pretrained model: {path_to_model}.pth")
        learner.load(path_to_model, device=device)
    else:
        print("Training form scratch")
    learner.fit(25, 1e-3)
    learner.save(path_to_model)


def check_data(databunched, vocab):
    first_batch = databunched.one_batch()[0]
    if not contains_no_value(first_batch, UNKNOWN_TOKEN_INDEX):
        raise ValueError(f"Unknown is found in tensor: {first_batch}")
    print(f'Displaying the first batch:\n{first_batch}')
    token_seqs = [vocab.textify(seq) for seq in first_batch]
    pprint(token_seqs)


def create_custom_config():
    config = awd_lstm_lm_config
    config['n_hid'] = 150
    return config


def encode_config(s: str) -> str:
    return s
