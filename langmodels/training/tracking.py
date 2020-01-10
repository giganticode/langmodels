import logging
import os
import sys
from typing import Any

import time
from comet_ml import Experiment
from fastai.basic_train import LearnerCallback, Learner
from fastai.callbacks import SaveModelCallback
from retrying import retry

from langmodels.lmconfig.datamodel import ExperimentRun

logger = logging.getLogger(__name__)


def change_path_to_permanent(experiment_run: ExperimentRun, learner: Learner) -> None:
    old_full_path: str = str(learner.path / learner.model_dir)
    new_full_path: str = experiment_run.perm_path_to_model
    new_path, new_model_dir = os.path.split(new_full_path)
    if str(learner.path) != new_path:
        raise AssertionError(f"When changing model path to permanent, path to models must remain the same. "
                             f"However trying to rename: {old_full_path} -> {new_full_path}")
    learner.model_dir = new_model_dir
    os.rename(old_full_path, new_full_path)


def report_successful_experiment_to_comet(experiment: Experiment):
    experiment.log_parameter("model_available", True)


class TrackingCallback(LearnerCallback):
    def __init__(self, learner: Learner, experiment_run: ExperimentRun):
        super().__init__(learner)
        self.experiment_run = experiment_run


class FirstModelTrainedCallback(TrackingCallback):
    def __init__(self, learner: Learner, experiment_run: ExperimentRun):
        super().__init__(learner, experiment_run)

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        if epoch == 0:
            self.experiment_run.set_first_model_trained()
            change_path_to_permanent(self.experiment_run, super().learn)
            if self.experiment_run.comet_experiment:
                report_successful_experiment_to_comet(self.experiment_run.comet_experiment)


class SaveTimePerEpochCallback(TrackingCallback):
    def __init__(self, learner: Learner, experiment_run: ExperimentRun):
        super().__init__(learner, experiment_run)
        self.n_epochs = 0
        self.total_time_minutes = 0

    def on_epoch_begin(self, **kwargs:Any) ->None:
        self.start_epoch_time = time.time()

    def on_epoch_end(self, **kwargs:Any) -> None:
        self.n_epochs += 1
        self.total_time_minutes += (time.time() - self.start_epoch_time) / 60
        minutes_per_epoch = int(self.total_time_minutes / self.n_epochs)
        self.experiment_run.metric_values.training_time_minutes_per_epoch = minutes_per_epoch
        comet_experiment = self.experiment_run.comet_experiment
        if comet_experiment:
            comet_experiment.log_metric("minutes_per_epoch", minutes_per_epoch, epoch=self.n_epochs-1)


class LrLogger(TrackingCallback):
    def __init__(self, learner, experiment_run: ExperimentRun = None):
        super().__init__(learner, experiment_run)

    def on_batch_end(self, **kwargs):
        num_batch = kwargs['num_batch']
        epoch = kwargs['epoch']
        comet_experiment = self.experiment_run.comet_experiment
        if comet_experiment:
            comet_experiment.log_metric('lr', self.learn.opt.lr, step=num_batch, epoch=epoch)


class RetryingSaveModelCalback(SaveModelCallback):
    def __init__(self, learn: Learner, experiment_run: ExperimentRun, monitor: str = 'valid_loss',
                 mode: str = 'auto', every: str = 'improvement', name: str = 'bestmodel'):
        super().__init__(learn, monitor, mode, every, name)
        self.experiment_run = experiment_run
        self.experiment_run.metric_values.bin_entropy = sys.maxsize

    def _retry_if_io_error(exception):
        """Return True if we should retry (in this case when it's an IOError), False otherwise"""
        exception_type_to_retry = IOError
        wil_retry =  isinstance(exception, exception_type_to_retry)
        if wil_retry:
            logger.warning(f"Caught exception: {repr(exception)}. Will retry after some time ...")
        else:
            logger.warning(f"Exception was raised: {repr(exception)}. Retrying only {exception_type_to_retry}")
        return wil_retry

    WAIT_MULTIPLIER = 30 * 1000
    WAIT_MAX = 3600 * 1000

    @retry(retry_on_exception=_retry_if_io_error, wait_exponential_multiplier=WAIT_MULTIPLIER, wait_exponential_max=WAIT_MAX)
    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        super().on_epoch_end(epoch, **kwargs)
        if self.get_monitor_value() < self.experiment_run.metric_values.bin_entropy:
            self.experiment_run.metric_values.bin_entropy = self.get_monitor_value()
            self.experiment_run.metric_values.best_epoch = self.on_epoch_end
        self.experiment_run.metric_values.n_epochs = self.epoch

    @retry(retry_on_exception=_retry_if_io_error, wait_exponential_multiplier=WAIT_MULTIPLIER, wait_exponential_max=WAIT_MAX)
    def on_train_end(self, **kwargs):
        super().on_train_end(**kwargs)