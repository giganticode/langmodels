import importlib
module = importlib.import_module('comet_ml')

#from torch.nn.functional import binary_cross_entropy

from langmodels.lmconfig.datamodel import Gpu
from langmodels.lmconfig.defaults import lm_training_config
from langmodels.training.training import train

if __name__ == '__main__':
    gpu = Gpu(fallback_to_cpu=True, non_default_device_to_use=0)
    trained_model = train(lm_training_config, gpu, tune=False)
#    trained_model.test()
#   for w in trained_model.predict_next(how_many=100):
#        print(w)
#    binary_cross_entropy
