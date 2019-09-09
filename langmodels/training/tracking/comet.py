import jsons
from comet_ml import Experiment
from fastai.basic_train import Learner

from fastai.callback import Callback
from fastai.text import Vocab
from flatdict import FlatDict

from langmodels.lmconfig.datamodel import PrepFunction
from langmodels.lmconfig.serialization import prep_function_deserializer, prep_function_serializer
from langmodels.lmconfig.datamodel import LMTrainingConfig


def get_param_number(learner: Learner) -> int:
    return sum(p.numel() for p in learner.model.parameters() if p.requires_grad)


def log_to_comet(id: str, lm_training_config: LMTrainingConfig,
                 learner: Learner, vocab: Vocab) -> Experiment:
    experiment = Experiment()
    experiment.set_name(id)
    jsons.set_serializer(prep_function_serializer, PrepFunction)
    jsons.set_deserializer(prep_function_deserializer, cls=PrepFunction)
    flat_config = FlatDict(jsons.dump(lm_training_config))
    for name, value in flat_config.items():
        experiment.log_parameter(name, value)
    experiment.log_parameter("trainable_params", get_param_number(learner))
    experiment.log_parameter("vocabulary", len(vocab.itos))

    tb_callback = LrLogger(learner, experiment)
    learner.callbacks.append(tb_callback)

    return experiment


class LrLogger(Callback):
    def __init__(self, learner, experiment: Experiment = None):
        super().__init__()
        self.learn = learner
        self.experiment = experiment

    def on_batch_end(self, **kwargs):
        num_batch = kwargs['num_batch']
        epoch = kwargs['epoch']

        self.experiment.log_metric('lr', self.learn.opt.lr, step=num_batch, epoch=epoch)
