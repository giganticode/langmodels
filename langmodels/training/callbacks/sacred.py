from fastai.callback import Callback
from sacred import Experiment
from typing import List


class SacredCallback(Callback):
    """
    Callback that reports metrics to sacred
    """

    def __init__(self, exp: Experiment, metric_names: List[str]):

        super().__init__()
        self.metric_names = metric_names
        self.exp = exp

    def on_batch_end(self, **kwargs):
        trn_loss = kwargs['last_loss']
        num_batch = kwargs['num_batch']

        self.exp.log_scalar('trn_loss_batch', trn_loss, num_batch)

    def on_epoch_end(self, **kwargs):
        metrics = kwargs['last_metrics']
        epoch = kwargs['epoch']
        trn_loss = kwargs['smooth_loss']

        self.exp.log_scalar('trn_loss', trn_loss, epoch)

        for val, name in zip(metrics, self.metric_names):
            self.exp.log_scalar(name, val, epoch)

    def on_train_end(self, **kwargs):
        self.exp.log_scalar('total_epochs', str(kwargs['epoch']))