"Based on the code kindly provided by TheShadow29: https://forums.fast.ai/t/tensorboard-callback-for-fastai/19048/8"

from fastai.callback import Callback
from pathlib import Path
import shutil
import os

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(Callback):
    """
    A general Purpose Logger for Tensorboard
    Also save a .txt file for the important parts
    """

    def __init__(self, learner, log_name, cfgtxt, del_existing=False, histogram_freq=100):
        """
        Learner is the ConvLearner
        log_name: name of the log directory to be formed. Will be input
        for each run
        cfgtxt: HyperParams
        del_existing: To run the experiment from scratch and remove previous logs
        """
        super().__init__()
        self.learn = learner
        self.model = learner.model
        self.md = learner.data

        self.metrics_names = ["validation_loss"]
        self.metrics_names += [m.__name__ for m in learner.metrics]

        self.best_met = 0

        self.histogram_freq = histogram_freq
        self.cfgtxt = cfgtxt

        path = Path(self.md.path) / "logs"
        if not path.exists():
            os.makedirs(path)        

        print(f"Logging tensorboard data to {path}")
        self.log_name = log_name
        self.log_dir = path / log_name

        self.init_logs(self.log_dir, del_existing)
        self.init_tb_writer()
        self.init_txt_writer(path, log_name)

    def init_logs(self, log_dir, del_existing):
        if log_dir.exists():
            if del_existing:
                print(f'removing existing log with same name {log_dir.stem}')
                shutil.rmtree(self.log_dir)

    def init_tb_writer(self):
        self.writer = SummaryWriter(
            comment='main_mdl', log_dir=str(self.log_dir))
        self.writer.add_text('HyperParams', self.cfgtxt)

    def init_txt_writer(self, path, log_name):
        self.fw_ = path / f'{log_name}.txt'
        self.str_form = '{} \t {} \t '
        for m in self.metrics_names:
            self.str_form += '{} \t '
        self.str_form += '\n'
        self.out_str = self.str_form.format(
            'epoch', 'trn_loss', *self.metrics_names)

        with open(self.fw_, 'w') as f:
            f.write(self.cfgtxt)
            f.write('\n')
            f.write(self.out_str)

    def on_batch_end(self, **kwargs):
        self.trn_loss = kwargs['last_loss']
        num_batch = kwargs['num_batch']
        self.writer.add_scalar(
            'trn_loss_batch', self.trn_loss, num_batch)

    def on_epoch_end(self, **kwargs):
        metrics = kwargs['last_metrics']
        epoch = kwargs['epoch']
        trn_loss = kwargs['smooth_loss']
        self.writer.add_scalar('trn_loss', trn_loss, epoch)

        for val, name in zip(metrics, self.metrics_names):
            self.writer.add_scalar(name, val, epoch)

        self.file_write(self.str_form.format(epoch,
                                             self.trn_loss, *metrics))

        m = metrics[1]
        if m > self.best_met:
            self.best_met = m
            self.learn.save(self.log_name)

    def on_train_end(self, **kwargs):
        self.writer.add_text('Total Epochs', str(kwargs['epoch']))
        self.writer.close()
        self.file_write(f'Epochs done, {kwargs["epoch"]}')

    def file_write(self, outstr):
        with open(self.fw_, 'a') as f:
            f.write(outstr)
