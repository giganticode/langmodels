import logging
import os
from concurrent.futures.process import ProcessPoolExecutor
from pprint import pprint

import numpy as np
import psutil
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import validate
from fastai.callback import CallbackHandler, Callback
from fastai.callbacks.mem import PeakMemMetric
from fastai.callbacks.misc import StopAfterNBatches
from fastai.layers import FlattenedLoss
from fastai.metrics import accuracy
from fastai.text import Vocab, TextList, language_model_learner, PreProcessor, Collection, partition_by_cores, \
    List
from fastai.train import fit_one_cycle, Learner, EarlyStoppingCallback
from math import log
from pathlib import Path
from torch import Tensor
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing import Tuple

import dataprep.api.corpus as api
from dataprep.api.corpus import PreprocessedCorpus
from langmodels import modelregistry
from langmodels.lmconfig.datamodel import LMTrainingConfig, DeviceOptions, Run, PATH_TO_TRAINED_MODELS, Corpus
from langmodels.lmconfig.datamodel import RafaelsTrainingSchedule, CosineLRSchedule, TrainingProcedure
from langmodels.lmconfig.serialization import dump_config
from langmodels.metrics import mrr
from langmodels.model import TrainedModel, create_custom_config
from langmodels.modelregistry import load_from_path
from langmodels.retrier import RetryingSaveModelCalback
from langmodels.training.schedule import ReduceLRCallback
from langmodels.training.tracking.comet import log_to_comet

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN_INDEX = 0

HOME = os.environ['HOME']
PATH_TO_PREP_DATASETS = os.environ['PATH_TO_PREP_DATASETS'] if 'PATH_TO_PREP_DATASETS' in os.environ else os.path.join(HOME, 'prep-datasets')


class Numericalizer(PreProcessor):
    def __init__(self, vocab: Vocab, n_cpus: int = os.cpu_count()):
        super().__init__()
        self.vocab = vocab
        self.n_cpus = n_cpus

    def process_one(self, item: str):
        with open(item, 'r') as f:
            prep_tokens = [token for line in f for token in line.rstrip('\n').split(' ')]
            #TODO XXX temp hack to fix \xa0 issue
            prep_tokens = ['`unk' if t == '\\xa0' else t for t in prep_tokens]
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
    """
    >>> t = torch.full((100,100,), 2)
    >>> contains_no_value(t, 0)
    True

    >>> t = torch.full((100,100,), 2)
    >>> t[1,45] = 0
    >>> contains_no_value(t, 0)
    False

    >>> t = torch.full((100,100,), 0)
    >>> contains_no_value(t, 0)
    False
    """
    return ((tensor == torch.full_like(tensor, value)).float().sum() == 0).item()


def create_vocab_for_lm(prep_corpus: PreprocessedCorpus) -> Vocab:
    return Vocab(['`unk', '`pad'] + list(prep_corpus.load_vocab().keys()))


def get_device_id(device_options: DeviceOptions) -> str:
    if torch.cuda.is_available():
        device_id = device_options.non_default_device_to_use or torch.cuda.current_device()
        return torch.device('cuda', device_id)
    elif device_options.fallback_to_cpu:
        return "cpu"
    else:
        raise EnvironmentError("Cuda not available")


def add_callbacks(learner: Learner, tune: bool) -> None:

    save_every_epoch_callback = RetryingSaveModelCalback(learner, every='epoch', name='epoch')
    learner.callbacks.append(save_every_epoch_callback)
    save_best_model_callback = RetryingSaveModelCalback(learner, every='improvement', name='best')
    learner.callbacks.append(save_best_model_callback)

    if tune:
        logger.warning("Tune mode is ON!")
        learner.callbacks.append(StopAfterNBatches(n_batches=2))


def start_training(learner: Learner, training_procedure: TrainingProcedure) -> None:
    schedule = training_procedure.schedule
    if isinstance(schedule, RafaelsTrainingSchedule):
        reduce_lr_callback = ReduceLRCallback(learner,
                                              mult_coeff=schedule.mult_coeff,
                                              max_times_lr_decrease=schedule.max_lr_reduction_times)
        learner.callbacks.append(reduce_lr_callback)
        learner.fit(epochs=schedule.max_epochs, lr=schedule.init_lr, wd=training_procedure.weight_decay)
    elif isinstance(schedule, CosineLRSchedule):
        if schedule.early_stop:
            learner.callbacks.append(EarlyStoppingCallback(learner, patience=schedule.early_stop.patience))
        fit_one_cycle(learner, cyc_len=schedule.cyc_len, tot_epochs=schedule.max_epochs,
                      max_lr=schedule.max_lr,
                      wd=training_procedure.weight_decay)
    # not saving the model explicitly because it should have been saved by the callbacks


def check_path_to_trained_model(path) -> None:
    if not os.path.exists(path):
        os.mkdir(path)
    elif not os.access(path, os.W_OK | os.X_OK):
        raise FileNotFoundError(path)


def check_path_to_base_model(config: LMTrainingConfig) -> None:
    if config.base_model:
        path_to_best_model = os.path.join(config.base_model, 'best.pth')
        if not os.path.exists(path_to_best_model):
            raise FileNotFoundError(path_to_best_model)


def get_cpu_memory_used_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 2 ** 20


def create_databunch(prep_corpus: PreprocessedCorpus, vocab: Vocab, bs: int, bptt: int, device: str,
                     only_validation_files: bool = False) -> DataBunch:
    print(f'Getting preprocessed corpus from {prep_corpus.path_to_prep_dataset}')
    file_path_list = [Path(os.fsdecode(p)) for p in prep_corpus.get_file_iterator()]
    text_list = TextList(file_path_list, path=prep_corpus.path_to_prep_dataset, processor=Numericalizer(vocab))

    print("Splitting into training/validation sets")
    if only_validation_files:
        split_list = text_list.split_by_valid_func(lambda f: True)
    else:
        split_list = text_list.split_by_folder()

    print("Labeling for langmodeling")
    labelled_list = split_list.label_for_lm()

    cpu_memory_used_mb = get_cpu_memory_used_mb()
    print(f"Cpu memory used: {cpu_memory_used_mb} MB")

    print("Creating databunches")
    databunched = labelled_list.databunch(bs=bs, bptt=bptt, device=device)
    return databunched


def load_base_model_if_needed(learner: Learner, lm_training_config: LMTrainingConfig, model_file='best') -> None:
    if lm_training_config.base_model:
        model = os.path.join(lm_training_config.base_model, model_file)
        print(f"Using pretrained model: {model}.pth")
        # not setting purge to True raises a pickle serialization error
        learner.load(model, purge=False)
    else:
        print("Training form scratch")


VOCAB_FILE_NAME = 'vocab'
CONFIG_FILE_NAME = 'config'


def save_experiment_input(learner: Learner, run: Run, vocab: Vocab, comet: bool):
    vocab.save(os.path.join(run.path_to_trained_model, VOCAB_FILE_NAME))
    dump_config(run.config, os.path.join(run.path_to_trained_model, CONFIG_FILE_NAME))
    if comet:
        log_to_comet(run.id, run.config, learner, vocab)


class CudaNotAvailable(Exception):
    pass


def run_validation(trained_model: TrainedModel, corpus: Corpus, only_validation_files: bool=False):
    prep_corpus: api.PreprocessedCorpus = trained_model.prep_corpus(corpus)
    try:
        device_id = get_device_id(DeviceOptions(fallback_to_cpu=True))
    except EnvironmentError:
        raise CudaNotAvailable()

    databunch = create_databunch(prep_corpus, trained_model.vocab, bs=64, bptt=1, device=device_id,
                                 only_validation_files=only_validation_files)

    class DetupleCallback(Callback):
        def on_loss_begin(self, last_output: Tuple[Tensor, Tensor, Tensor], **kwargs):
            "Save the extra outputs for later and only returns the true output."
            return {'last_output': last_output[0]}

    return validate(trained_model.model, databunch.valid_dl, loss_func=FlattenedLoss(CrossEntropyLoss),
             cb_handler=CallbackHandler([DetupleCallback()]))


def train(lm_training_config: LMTrainingConfig,
          device_options: DeviceOptions(),
          tune=False, comet=True) -> TrainedModel:

    run = Run.with_config(lm_training_config, device_options=device_options)

    check_path_to_base_model(lm_training_config)
    check_path_to_trained_model(run.path_to_trained_model)

    prep_corpus: api.PreprocessedCorpus = lm_training_config.prep_function.apply(lm_training_config.corpus,
                                                                                 output_path=PATH_TO_PREP_DATASETS)
    vocab = create_vocab_for_lm(prep_corpus)

    try:
        device = get_device_id(device_options)
    except EnvironmentError:
        raise CudaNotAvailable()

    databunch = create_databunch(prep_corpus, vocab, bs=lm_training_config.bs, bptt=lm_training_config.bptt,
                                 device=device)

    check_data(databunch, vocab)

    config = create_custom_config(lm_training_config)
    arch_class = lm_training_config.get_arch_class()
    learner = language_model_learner(databunch, arch_class,
                                     # drop_mult=lm_training_config.arch.drop.multiplier,
                                     config=config, pretrained=not config, metrics=[accuracy, mrr],
                                     callback_fns=[PeakMemMetric] if torch.cuda.is_available() else [],
                                     path=PATH_TO_TRAINED_MODELS, model_dir=run.id)

    save_experiment_input(learner, run, vocab, comet=comet)

    add_callbacks(learner, tune=tune)

    load_base_model_if_needed(learner, lm_training_config)

    print(f"Starting training... Model will be saved to {run.path_to_trained_model}")
    start_training(learner, lm_training_config.training_procedure)

    # learner.export('learner.pkl')
    # return load_learner(os.path.join(PATH_TO_TRAINED_MODELS, run.id), 'learner.pkl')
    return load_from_path(run.path_to_trained_model, force_use_cpu=True)
    # return learner


def check_data(databunched: DataBunch, vocab: Vocab) -> None:
    first_batch = databunched.one_batch()[0]

#    if not contains_no_value(first_batch, UNKNOWN_TOKEN_INDEX):
#        raise ValueError(f"Unknown is found in tensor: {first_batch}")
    print(f'Displaying the first batch:\n{first_batch}')
    token_seqs = [vocab.textify(seq) for seq in first_batch]
    pprint(token_seqs)
