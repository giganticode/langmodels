import logging
from dataclasses import dataclass
from typing import Any, Dict

from fastai.basic_train import Learner
from fastai.callbacks import TrackerCallback

from langmodels.lmconfig.datamodel import TrainingSchedule

logger = logging.getLogger(__name__)


class ReduceLRCallback(TrackerCallback):
    def __init__(self, learn: Learner, mult_coeff: float, max_times_lr_decrease: int, patience: int,
                 monitor: str = 'valid_loss', mode: str = 'auto'):
        super().__init__(learn, monitor, mode)
        self.mult_coeff = mult_coeff
        self.lr_decreased_times = 0
        self.max_lr_decrease_times = max_times_lr_decrease
        self.no_improvement_epochs = 0
        self.patience = patience

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> Dict:
        current = self.get_monitor_value()
        if current is not None and self.operator(current, self.best):
            self.best = current
            self.lr_decreased_times = 0
            self.no_improvement_epochs = 0
        elif self.no_improvement_epochs < self.patience:
            self.no_improvement_epochs += 1
            logger.info(f'The loss has not improved for {self.no_improvement_epochs}.'
                        f'Waiting for {self.patience} epochs, then reducing the learnign rate.')
        else:
            self.no_improvement_epochs = 0
            self.learn.opt.lr *= self.mult_coeff
            logger.info(
                f"The loss has not improved for {self.no_improvement_epochs} epochs, "
                f"reducing learning rate to {self.learn.opt.lr} for the next epoch ...")
            self.lr_decreased_times += 1
            if self.lr_decreased_times > self.max_lr_decrease_times:
                return {"stop_training": True}


@dataclass(frozen=True)
class RafaelsTrainingSchedule(TrainingSchedule):
    name: str = 'rafael'
    init_lr: float = 1e-4
    mult_coeff: float = 0.5
    max_epochs: int = 50
    max_lr_reduction_times: int = 6
    patience: int = 0

    def fit(self, learner: Learner, weigth_decay: float):
        reduce_lr_callback = ReduceLRCallback(learner,
                                              mult_coeff=self.mult_coeff,
                                              max_times_lr_decrease=self.max_lr_reduction_times,
                                              patience=self.patience)
        learner.callbacks.append(reduce_lr_callback)
        learner.fit(epochs=self.max_epochs, lr=self.init_lr, wd=weigth_decay)
