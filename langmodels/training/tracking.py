import logging
import os
import sys
from typing import Any

import time
from comet_ml import Experiment
from fastai.basic_train import LearnerCallback, Learner
from fastai.callbacks import SaveModelCallback, TrackerCallback
from retrying import retry

from langmodels.lmconfig.datamodel import ExperimentRun, LMTrainingMetrics
from langmodels.lmconfig.serialization import dump_to_file
from langmodels.model import METRICS_FILE_NAME, BEST_MODEL_FILE_NAME
from langmodels.nn import get_param_number

logger = logging.getLogger(__name__)

BYTES_IN_MB = 1 << 20

MODEL_AVAILABLE_METRIC_NAME = 'model available'
TERMINATED_NORMALLY_METRIC_NAME = 'terminated normally'


def change_path_to_permanent(experiment_run: ExperimentRun, learner: Learner) -> None:
    old_full_path: str = str(learner.path / learner.model_dir)
    new_full_path: str = experiment_run.perm_path_to_model
    new_path, new_model_dir = os.path.split(new_full_path)
    if str(learner.path) != new_path:
        raise AssertionError(f"When changing model path to permanent, path to models must remain the same. "
                             f"However trying to rename: {old_full_path} -> {new_full_path}")
    learner.model_dir = new_model_dir
    os.rename(old_full_path, new_full_path)


def report_successful_experiment_to_comet(experiment: Experiment) -> None:
    experiment.log_metric(MODEL_AVAILABLE_METRIC_NAME, True)


def report_experiment_terminated_mormally(experiment: Experiment) -> None:
    experiment.log_metric(TERMINATED_NORMALLY_METRIC_NAME, True)


class ExperimentTrackingCallback(LearnerCallback):
    def __init__(self, learner: Learner, experiment_run: ExperimentRun):
        super().__init__(learner)
        self.experiment_run = experiment_run


class FirstModelTrainedCallback(ExperimentTrackingCallback):
    def __init__(self, learner: Learner, experiment_run: ExperimentRun):
        super().__init__(learner, experiment_run)

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        if epoch == 0:
            self.experiment_run.set_first_model_trained()
            change_path_to_permanent(self.experiment_run, super().learn)
            if self.experiment_run.comet_experiment:
                report_successful_experiment_to_comet(self.experiment_run.comet_experiment)


class LrLogger(ExperimentTrackingCallback):
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

    @retry(retry_on_exception=_retry_if_io_error, wait_exponential_multiplier=WAIT_MULTIPLIER, wait_exponential_max=WAIT_MAX)
    def on_train_end(self, **kwargs):
        super().on_train_end(**kwargs)


class MetricSavingCallback(TrackerCallback):
    def __init__(self, learner:Learner, experiment_run: ExperimentRun):
        super().__init__(learner)
        self.experiment_run = experiment_run
        self.metric_values = LMTrainingMetrics(
            bin_entropy=sys.maxsize,
            training_time_minutes_per_epoch=0,
            n_epochs=0,
            best_epoch=-1,
            trainable_params=get_param_number(learner.model),
            size_on_disk_mb=None
        )

    @staticmethod
    def _log_metrics_to_comet(comet_experiment, metric_values) -> None:
        comet_experiment.log_metric("minutes per epoch", metric_values.training_time_minutes_per_epoch, epoch=metric_values.n_epochs-1)
        comet_experiment.log_metric("trainable params", metric_values.trainable_params)
        comet_experiment.log_metric("model size mb", metric_values.size_on_disk_mb)
        comet_experiment.log_metric("best epoch", metric_values.best_epoch)

    def _update_training_time_per_epoch(self) -> int:
        total_training_time_before_this_epoch = self.metric_values.training_time_minutes_per_epoch * (self.metric_values.n_epochs - 1)
        this_epoch_training_time = (time.time() - self.start_epoch_time) / 60
        self.total_time_minutes = total_training_time_before_this_epoch + this_epoch_training_time
        return int(self.total_time_minutes / self.metric_values.n_epochs)

    def _get_model_size_mb(self) -> int:
        model_file = os.path.join(self.experiment_run.path_to_trained_model, BEST_MODEL_FILE_NAME)
        size_of_model_file_in_bytes = os.stat(model_file).st_size
        return size_of_model_file_in_bytes // BYTES_IN_MB

    def on_epoch_begin(self, **kwargs:Any) ->None:
        self.start_epoch_time = time.time()

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        super().on_epoch_end(**kwargs)

        self.metric_values.n_epochs = epoch + 1
        if self.get_monitor_value() < self.metric_values.bin_entropy:
            self.metric_values.bin_entropy = float(self.get_monitor_value())
            self.metric_values.best_epoch = epoch

        self.metric_values.training_time_minutes_per_epoch = self._update_training_time_per_epoch()
        self.metric_values.size_on_disk_mb = self._get_model_size_mb()

        path_to_metrics_file = os.path.join(self.experiment_run.path_to_trained_model, METRICS_FILE_NAME)
        dump_to_file(self.metric_values, path_to_metrics_file)

        comet_experiment = self.experiment_run.comet_experiment
        if comet_experiment:
            self._log_metrics_to_comet(comet_experiment, self.metric_values)
