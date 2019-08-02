import os

import dataprep.api.corpus as api
from dataprep.api.corpus import PreprocessedCorpus
from torch import cuda

from langmodels.training.training import create_custom_config, train

PATH_TO_TRAINED_MODELS = '/home/lv71161/hlibbabii/trained-models'
PATH_TO_PREP_DATASETS = '/home/lv71161/hlibbabii/projects/dataprep'


if __name__ == '__main__':
    prep_corpus:PreprocessedCorpus = api.bpe('/home/lv71161/hlibbabii/raw_datasets/allamanis/nodup_en_only/0/test/600_fedora-client/', '10k', no_str=True, no_com=True, calc_vocab=True, no_case=True,
                                                                                                       output_path=PATH_TO_PREP_DATASETS)
    config = create_custom_config()
    train(prep_corpus=prep_corpus,
          path_to_model=os.path.join(PATH_TO_TRAINED_MODELS, 'fedora_client'),
          device=cuda.current_device(),
          config=config)
