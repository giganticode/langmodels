import importlib
module = importlib.import_module('comet_ml')

from langmodels.lmconfig import defaults
from langmodels.lmconfig.defaults import lm_training_config
from langmodels.model import TrainedModel
from langmodels.training.training import train

from langmodels.lmconfig.datamodel import Gpu


if __name__ == '__main__':
    gpu = Gpu(fallback_to_cpu=True, non_default_device_to_use=0)
    train(lm_training_config=lm_training_config, gpu=gpu)
