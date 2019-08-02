import os
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Any

import jsons
import numpy as np
import torch
from dataprep.api.corpus import PreprocessedCorpus
from fastai.text import Vocab, TextList, language_model_learner, PreProcessor, Collection, partition_by_cores, \
    List, awd_lstm_lm_config
from fastai.text.models.transformer import init_transformer
from sacred import Experiment
from torch import Tensor
from tqdm import tqdm

from langmodels.config.datamodel import LMTrainingConfig, LstmArch, TransformerArch
from langmodels.training.callbacks.sacred import SacredCallback
from langmodels.training.callbacks.tensorboard import TensorboardLogger

UNKNOWN_TOKEN_INDEX = 0
PAD_TOKEN_INDEX = 1


class Numericalizer(PreProcessor):
    def __init__(self, vocab: Vocab, n_cpus: int = os.cpu_count()):
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

    def _process_on_one_core(self, items: Collection[str]) -> List:
        return [self.process_one(item) for item in tqdm(items)]

    def _process_in_parallel(self, texts: Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1:
            return self._process_on_one_core(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_on_one_core, partition_by_cores(texts, self.n_cpus)), [])


def contains_no_value(tensor: Tensor, value: int) -> bool:
    return (tensor == torch.full_like(tensor, value)).float().sum() == 0


def create_vocab_for_lm(prep_corpus: PreprocessedCorpus) -> Vocab:
    return Vocab(['`unk', '`pad'] + list(prep_corpus.load_vocab().keys()))


def train(prep_corpus: PreprocessedCorpus,
          path_to_model: str, device: Optional[int],
          lm_training_config: LMTrainingConfig,
          use_pretrained_model: bool = True,
          sacred_exp: Optional[Experiment] = None):
    vocab = create_vocab_for_lm(prep_corpus)

    print(f'Getting preprocessed corpus from {prep_corpus.path_to_prep_dataset}')
    file_path_list = [Path(os.fsdecode(p)) for p in prep_corpus.get_file_iterator()]
    text_list = TextList(file_path_list, path=prep_corpus.path_to_prep_dataset, processor=Numericalizer(vocab))

    print("Splitting into training/validation sets")
    split_list = text_list.split_by_rand_pct()

    print("Labeling for langmodeling")
    labelled_list = split_list.label_for_lm()

    print("Creating databunches")
    databunched = labelled_list.databunch(
        bs=lm_training_config.bs,
        bptt=lm_training_config.bptt,
        device=device)

    check_data(databunched, vocab)

    config = create_custom_config(lm_training_config)
    arch_class = lm_training_config.get_arch_class()
    learner = language_model_learner(databunched, arch_class,
                                     # drop_mult=lm_training_config.arch.drop.multiplier,
                                     config=config, pretrained=not config)
    if use_pretrained_model and os.path.exists(f'{path_to_model}.pth'):
        print(f"Using pretrained model: {path_to_model}.pth")
        learner.load(path_to_model, device=device)
    else:
        print("Training form scratch")

    if sacred_exp:
        exp_name = sacred_exp.get_experiment_info()['name']
        tb_callback = TensorboardLogger(learner, f'{exp_name}', jsons.dumps(lm_training_config), del_existing=False)

        sacred_callback = SacredCallback(sacred_exp, ["validation_loss"] + [m.__name__ for m in learner.metrics])

        learner.callbacks.extend([tb_callback, sacred_callback])

    learner.fit(epochs=lm_training_config.training_procedure.cycle.n,
                lr=lm_training_config.training_procedure.base_lr,
                wd=lm_training_config.training_procedure.weight_decay)
    learner.save(path_to_model)


def check_data(databunched, vocab):
    first_batch = databunched.one_batch()[0]
    if not contains_no_value(first_batch, UNKNOWN_TOKEN_INDEX):
        raise ValueError(f"Unknown is found in tensor: {first_batch}")
    print(f'Displaying the first batch:\n{first_batch}')
    token_seqs = [vocab.textify(seq) for seq in first_batch]
    pprint(token_seqs)


def create_custom_lstm_config(arch: LstmArch):
    config = awd_lstm_lm_config
    config['emb_sz'] = arch.emb_sz
    config['n_hid'] = arch.n_hid
    config['n_layers'] = arch.n_layers
    config['pad_token'] = PAD_TOKEN_INDEX
    config['qrnn'] = arch.qrnn
    config['bidir'] = arch.bidir
    config['output_p'] = arch.drop.out
    config['hidden_p'] = arch.drop.outh
    config['input_p'] = arch.drop.outi
    config['embed_p'] = arch.drop.oute
    config['weight_p'] = arch.drop.w
    config['tie_weights'] = arch.tie_weights
    config['out_bias'] = arch.out_bias
    return config


def create_custom_transformer_config(arch: TransformerArch) -> Dict[str, Any]:
    d = arch.__dict__
    d['init'] = init_transformer
    return d


def create_custom_config(lm_training_config: LMTrainingConfig):
    arch = lm_training_config.arch
    if isinstance(arch, LstmArch):
        return create_custom_lstm_config(arch)
    elif isinstance(arch, TransformerArch):
        return create_custom_transformer_config(arch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def encode_config(s: str) -> str:
    return s
