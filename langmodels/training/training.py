import os

import logging
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import validate
from fastai.callback import CallbackHandler, Callback
from fastai.callbacks.misc import StopAfterNBatches
from fastai.layers import CrossEntropyFlat
from fastai.metrics import accuracy, Perplexity
from fastai.text import Vocab, language_model_learner
from fastai.train import fit_one_cycle, Learner, EarlyStoppingCallback
from typing import Tuple

import codeprep.api.corpus as api
from codeprep.api.corpus import PreprocessedCorpus
from codeprep.util import to_literal_str

from langmodels.cuda_util import DeviceOptions
from langmodels.file_util import get_all_files
from langmodels.lmconfig.datamodel import LMTrainingConfig, Corpus
from langmodels.training.experiment import ExperimentRun
from langmodels.model import TrainedModel, create_custom_config
from langmodels.repository.load import load_from_path
from langmodels.tensor_ops import mrr
from langmodels.training.data import EmptyDataBunch, create_databunch
from langmodels.training.schedule import ReduceLRCallback
from langmodels.training.subepoch_files import EpochFileLoader
from langmodels.training.tracking import FirstModelTrainedCallback, LrLogger, RetryingSaveModelCalback, \
    MetricSavingCallback, report_experiment_terminated_mormally
from langmodels.util import HOME

logger = logging.getLogger(__name__)

PATH_TO_PREP_DATASETS = os.environ['PATH_TO_PREP_DATASETS'] if 'PATH_TO_PREP_DATASETS' in os.environ \
    else os.path.join(HOME, 'prep-datasets')


def create_vocab_for_lm(prep_corpus: PreprocessedCorpus) -> Vocab:
    return Vocab(['`unk', '`pad'] + list(map(lambda x: to_literal_str(x), prep_corpus.load_vocab().keys())))


def load_base_model_if_needed(learner: Learner, lm_training_config: LMTrainingConfig, model_file='best') -> None:
    if lm_training_config.base_model:
        model = os.path.join(lm_training_config.base_model, model_file)
        logger.info(f"Using pretrained model: {model}.pth")
        # not setting purge to True raises a pickle serialization error
        learner.load(model, purge=False)
    else:
        logger.info("Training form scratch")


def run_validation(trained_model: TrainedModel, corpus: Corpus, only_validation_files: bool = False,
                   device_options: DeviceOptions = DeviceOptions(fallback_to_cpu=True)):
    """
    Validation using fastai's `validation` method
    """
    prep_corpus: api.PreprocessedCorpus = trained_model.prep_corpus(corpus)
    config: LMTrainingConfig = trained_model.config

    device_id = device_options.get_device_id()

    logger.info(f"Vocab size: {len(trained_model.vocab.itos)}")
    all_files = [f for f in get_all_files(prep_corpus.path_to_prep_dataset, None)]
    databunch = create_databunch(prep_corpus.path_to_prep_dataset, all_files, trained_model.vocab,
                                 bs=config.bs, bptt=config.bptt, device=device_id,
                                 only_validation_files=only_validation_files, allow_unks=True)

    class DetupleCallback(Callback):
        def on_loss_begin(self, last_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
            """Save the extra outputs for later and only returns the true output."""
            return {'last_output': last_output[0]}

    return validate(trained_model.model, databunch.valid_dl, loss_func=CrossEntropyFlat(),
                    cb_handler=CallbackHandler([DetupleCallback()]))


def add_callbacks(experiment_run: ExperimentRun, learner: Learner, vocab: Vocab, tune: bool, save_every_epoch: bool) -> None:
    learner.callbacks.append(LrLogger(learner, experiment_run))

    first_model_trained_callback = FirstModelTrainedCallback(learner, experiment_run)
    learner.callbacks.append(first_model_trained_callback)
    if save_every_epoch:
        save_every_epoch_callback = RetryingSaveModelCalback(learner, experiment_run, every='epoch', name='epoch')
        learner.callbacks.append(save_every_epoch_callback)
    save_best_model_callback = RetryingSaveModelCalback(learner, experiment_run, every='improvement', name='best')
    learner.callbacks.append(save_best_model_callback)

    metric_saving_callback = MetricSavingCallback(learner, experiment_run)
    learner.callbacks.append(metric_saving_callback)

    if tune:
        logger.warning("Tune mode is ON!")
        learner.callbacks.append(StopAfterNBatches(n_batches=2))


def train(training_config: LMTrainingConfig = LMTrainingConfig(),
          device_options: DeviceOptions = DeviceOptions(),
          tune: bool = False, comet: bool = True, save_every_epoch: bool = False, allow_unks: bool = False) -> TrainedModel:
    experiment_run = ExperimentRun.with_config(training_config, device_options=device_options, comet=comet)
    experiment_run.log_experiment_input()

    if isinstance(training_config.corpus, Corpus):
        prep_corpus: api.PreprocessedCorpus = training_config.prep_function.apply(training_config.corpus,
                                                                              calc_vocab=True,
                                                                              output_path=PATH_TO_PREP_DATASETS)
    else:
        prep_corpus = training_config.corpus

    vocab = create_vocab_for_lm(prep_corpus)
    experiment_run.log_vocab(vocab)

    device = device_options.get_device_id()

    config = create_custom_config(training_config)
    arch_class = training_config.arch.get_module()
    dropout_multiplier = training_config.arch.drop.multiplier
    training = training_config.training

    if training_config.training.sub_epochs:
        data_bunch: DataBunch = EmptyDataBunch(vocab=vocab, path=prep_corpus.path_to_prep_dataset, device=device)
    else:
        data_bunch = create_databunch(prep_corpus.path_to_prep_dataset, get_all_files(prep_corpus.path_to_prep_dataset, None),
                                     vocab, bs=training_config.bs, bptt=training_config.bptt,
                                     device=device, verbose=True, allow_unks=allow_unks)

    learner = language_model_learner(data_bunch, arch_class, opt_func=training.optimizer.get_callable(),
                                     drop_mult=dropout_multiplier,
                                     config=config, pretrained=False, metrics=[accuracy, mrr, Perplexity()],
                                     clip=training.gradient_clip,
                                     alpha=training.activation_regularization.alpha,
                                     beta=training.activation_regularization.beta,
                                     path=os.path.dirname(experiment_run.path_to_trained_model),
                                     model_dir=os.path.basename(experiment_run.path_to_trained_model))

    if training_config.training.sub_epochs:
        files_per_epoch = training_config.training.sub_epochs.n_files
        learner.callbacks.append(EpochFileLoader(learner, prep_corpus, vocab,
                                                 bs=training_config.bs, bptt=training_config.bptt, device=device,
                                                 n_files_per_epoch=files_per_epoch, allow_unks=allow_unks))


    add_callbacks(experiment_run, learner, vocab, tune, save_every_epoch=save_every_epoch)

    load_base_model_if_needed(learner, training_config)

    logger.info(f"Starting training... Model will be saved to {experiment_run.perm_path_to_model} "
          f"(Saving config and vocab to {experiment_run.path_to_trained_model} before getting the first trained model)")
    training_config.training.schedule.fit(learner, training.weight_decay)
    if experiment_run.comet_experiment:
        report_experiment_terminated_mormally(experiment_run.comet_experiment)

    # TODO export learner?
    return load_from_path(experiment_run.path_to_trained_model, force_use_cpu=True)
