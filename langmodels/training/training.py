import logging
import os
from pprint import pformat

import torch
from comet_ml import Experiment
from fastai.basic_data import DataBunch
from fastai.basic_train import validate
from fastai.callback import CallbackHandler, Callback
from fastai.callbacks.mem import PeakMemMetric
from fastai.callbacks.misc import StopAfterNBatches
from fastai.layers import FlattenedLoss
from fastai.metrics import accuracy
from fastai.text import Vocab, language_model_learner
from fastai.train import fit_one_cycle, Learner, EarlyStoppingCallback
from flatdict import FlatDict
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple

import dataprep.api.corpus as api
from dataprep.api.corpus import PreprocessedCorpus
from dataprep.util import to_literal_str
from langmodels.cuda_util import get_device_id
from langmodels.file_util import check_path_exists, check_path_writable, get_all_files
from langmodels.lmconfig.datamodel import LMTrainingConfig, Corpus, \
    RafaelsTrainingSchedule, TrainingProcedure, CosineLRSchedule, ExperimentRun, DeviceOptions
from langmodels.lmconfig.serialization import dump_config, dump_config_to_string, dump_config_to_json
from langmodels.model import TrainedModel, create_custom_config
from langmodels.modelregistry import load_from_path
from langmodels.tensor_ops import mrr
from langmodels.training.data import EmptyDataBunch, create_databunch
from langmodels.training.schedule import ReduceLRCallback
from langmodels.training.subepoch_files import EpochFileLoader
from langmodels.training.tracking import FirstModelTrainedCallback, LrLogger, RetryingSaveModelCalback

logger = logging.getLogger(__name__)

VOCAB_FILE_NAME = 'vocab'
CONFIG_FILE_NAME = 'config'

HOME = os.environ['HOME']
PATH_TO_PREP_DATASETS = os.environ['PATH_TO_PREP_DATASETS'] if 'PATH_TO_PREP_DATASETS' in os.environ \
    else os.path.join(HOME, 'prep-datasets')


def create_vocab_for_lm(prep_corpus: PreprocessedCorpus) -> Vocab:
    return Vocab(['`unk', '`pad'] + list(map(lambda x: to_literal_str(x), prep_corpus.load_vocab().keys())))


def choose_schedule_and_fit(learner: Learner, training_procedure: TrainingProcedure) -> None:
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


def load_base_model_if_needed(learner: Learner, lm_training_config: LMTrainingConfig, model_file='best') -> None:
    if lm_training_config.base_model:
        model = os.path.join(lm_training_config.base_model, model_file)
        print(f"Using pretrained model: {model}.pth")
        # not setting purge to True raises a pickle serialization error
        learner.load(model, purge=False)
    else:
        print("Training form scratch")


def save_experiment_input(run: ExperimentRun, learner: Learner, vocab: Vocab):
    vocab.save(os.path.join(run.path_to_trained_model, VOCAB_FILE_NAME))
    dump_config(run.config, os.path.join(run.path_to_trained_model, CONFIG_FILE_NAME))
    if run.comet_experiment:
        save_params_to_comet(run.comet_experiment, run.config, vocab, get_param_number(learner))


def check_run_prerequisites(run: ExperimentRun) -> None:
    if run.config.base_model:
        check_path_exists(os.path.join(run.config.base_model, 'best.pth'))
    check_path_writable(run.path_to_trained_model)


def run_validation(trained_model: TrainedModel, corpus: Corpus, only_validation_files: bool = False,
                   fallback_to_cpu: bool = True, non_default_device_to_use: Optional[int] = None):
    """
    Validation using fastai's `validation` method
    """
    prep_corpus: api.PreprocessedCorpus = trained_model.prep_corpus(corpus)
    config: LMTrainingConfig = trained_model.config

    device_id = get_device_id(fallback_to_cpu, non_default_device_to_use)

    logger.info(f"Vocab size: {len(trained_model.vocab.itos)}")
    all_files = [f for f in get_all_files(prep_corpus.path_to_prep_dataset, None)]
    databunch = create_databunch(prep_corpus.path_to_prep_dataset, all_files, trained_model.vocab,
                                 bs=config.bs, bptt=config.bptt, device=device_id,
                                 only_validation_files=only_validation_files, allow_unks=True)

    class DetupleCallback(Callback):
        def on_loss_begin(self, last_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
            """Save the extra outputs for later and only returns the true output."""
            return {'last_output': last_output[0]}

    return validate(trained_model.model, databunch.valid_dl, loss_func=FlattenedLoss(CrossEntropyLoss),
                    cb_handler=CallbackHandler([DetupleCallback()]))


def get_param_number(learner: Learner) -> int:
    return sum(p.numel() for p in learner.model.parameters() if p.requires_grad)


def save_params_to_comet(experiment: Experiment, lm_training_config: LMTrainingConfig,
                         vocab: Vocab, trainable_params: int) -> Experiment:
    flat_config = FlatDict(dump_config_to_json(lm_training_config))
    for name, value in flat_config.items():
        experiment.log_parameter(name, value)
    experiment.log_parameter("vocabulary", len(vocab.itos))
    experiment.log_parameter("trainable_params", trainable_params)
    experiment.log_parameter("model_available", False)
    return experiment


def add_callbacks(experiment_run: ExperimentRun, learner: Learner, vocab: Vocab, tune: bool) -> None:
    comet_experiment: Optional[Experiment] = experiment_run.comet_experiment
    if comet_experiment:
        learner.callbacks.append(LrLogger(learner, comet_experiment))

    first_model_trained_callback = FirstModelTrainedCallback(learner, experiment_run)
    learner.callbacks.append(first_model_trained_callback)

    save_every_epoch_callback = RetryingSaveModelCalback(learner, every='epoch', name='epoch')
    learner.callbacks.append(save_every_epoch_callback)
    save_best_model_callback = RetryingSaveModelCalback(learner, every='improvement', name='best')
    learner.callbacks.append(save_best_model_callback)

    if tune:
        logger.warning("Tune mode is ON!")
        learner.callbacks.append(StopAfterNBatches(n_batches=2))


def train(training_config: LMTrainingConfig = LMTrainingConfig(),
          device_options: DeviceOptions() = DeviceOptions(),
          tune=False, comet=True) -> TrainedModel:
    logger.info(f'Using the following config: \n{pformat(dump_config_to_string(training_config))}')
    experiment_run = ExperimentRun.with_config(training_config, device_options=device_options, comet=comet)
    check_run_prerequisites(experiment_run)

    prep_corpus: api.PreprocessedCorpus = training_config.prep_function.apply(training_config.corpus,
                                                                              output_path=PATH_TO_PREP_DATASETS)
    vocab = create_vocab_for_lm(prep_corpus)
    print(f"Vocab size: {len(vocab.itos)}")
    device = get_device_id(device_options.fallback_to_cpu, device_options.non_default_device_to_use)
    empty_data_bunch: DataBunch = EmptyDataBunch(vocab=vocab, path=prep_corpus.path_to_prep_dataset, device=device)

    config = create_custom_config(training_config)
    arch_class = training_config.get_arch_class()
    learner = language_model_learner(empty_data_bunch, arch_class,
                                     drop_mult=training_config.arch.drop.multiplier,
                                     config=config, pretrained=not config, metrics=[accuracy, mrr],
                                     callback_fns=[PeakMemMetric] if torch.cuda.is_available() else [],
                                     path=os.path.dirname(experiment_run.path_to_trained_model),
                                     model_dir=os.path.basename(experiment_run.path_to_trained_model))

    files_per_epoch = training_config.training_procedure.files_per_epoch
    learner.callbacks.append(EpochFileLoader(learner, prep_corpus, vocab,
                                             bs=training_config.bs, bptt=training_config.bptt, device=device,
                                             n_files_per_epoch=files_per_epoch))

    save_experiment_input(experiment_run, learner, vocab)

    add_callbacks(experiment_run, learner, vocab, tune)

    load_base_model_if_needed(learner, training_config)

    print(f"Starting training... Model will be saved to {experiment_run.perm_path_to_model} "
          f"(Saving config and vocab to {experiment_run.path_to_trained_model} before getting the first trained model)")
    choose_schedule_and_fit(learner, training_config.training_procedure)

    # TODO export learner?
    return load_from_path(experiment_run.path_to_trained_model, force_use_cpu=True)
