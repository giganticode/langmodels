import importlib

from langmodels.model import TrainedModel

module = importlib.import_module('comet_ml')

from langmodels.lmconfig.datamodel import Gpu



if __name__ == '__main__':
    gpu = Gpu(fallback_to_cpu=True, non_default_device_to_use=0)
    trained_model = TrainedModel.get_default_model()
    text = 'import'
    for i in range(1):
        text += ' ' + trained_model.predict_next_full_token(text, 5)
    print(text)
#    trained_model.test()
#   for w in trained_model.predict_next(how_many=100):
#        print(w)
#    binary_cross_entropy
