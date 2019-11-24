import logging
import os

from comet_ml import Experiment
from fastai.basic_train import LearnerCallback, Learner
from fastai.callback import Callback
from fastai.callbacks import SaveModelCallback
from retrying import retry
from typing import Any

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


class FirstModelTrainedCallback(LearnerCallback):
    def __init__(self, learner: Learner, experiment_run: ExperimentRun):
        super().__init__(learner)
        self.experiment_run = experiment_run

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        if epoch == 0:
            self.experiment_run.set_first_model_trained()
            change_path_to_permanent(self.experiment_run, super().learn)
            if self.experiment_run.comet_experiment:
                report_successful_experiment_to_comet(self.experiment_run.comet_experiment)


class LrLogger(Callback):
    def __init__(self, learner, experiment: Experiment = None):
        super().__init__()
        self.learn = learner
        self.experiment = experiment

    def on_batch_end(self, **kwargs):
        num_batch = kwargs['num_batch']
        epoch = kwargs['epoch']

        self.experiment.log_metric('lr', self.learn.opt.lr, step=num_batch, epoch=epoch)


class RetryingSaveModelCalback(SaveModelCallback):
    def __init__(self, learn: Learner, monitor: str = 'valid_loss', mode: str = 'auto', every: str = 'improvement',
                 name: str = 'bestmodel'):
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