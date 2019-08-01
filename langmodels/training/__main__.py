import copy
import os
import time
from typing import Any

import dataprep.api.corpus as api
import jsons
from sacred import Experiment
from sacred.config.custom_containers import ReadOnlyDict, ReadOnlyList
from torch import cuda

from langmodels.config.datamodel import LMTrainingConfig, PrepFunction, LstmArch, TrainingProcedure, LearningRateCycle, \
    TransformerArch, Corpus
from langmodels.config.serialization import prep_function_serializer, prep_function_deserializer
from langmodels.training.training import train, encode_config

jsons.set_serializer(prep_function_serializer, PrepFunction)
jsons.set_deserializer(prep_function_deserializer, cls=PrepFunction)

PATH_TO_TRAINED_MODELS = '/home/hlib/dev/trained-models'
PATH_TO_PREP_DATASETS = '/home/hlib/dev/prep-datasets'

lm_training_config = LMTrainingConfig(base_model=None,
                                      corpus=Corpus(path='/home/hlib/dev/yahtzee', extensions="java"),
                                      prep_function=PrepFunction(api.bpe, ['10k'],
                                                                 {'no_str': True,
                                                                  'no_com': True,
                                                                  'no_case': True,
                                                                  }),
                                      arch=TransformerArch(),
                                      bs=4, bptt=30,
                                      training_procedure=TrainingProcedure(base_lr=1e-3,
                                                                           cycle=LearningRateCycle(n=5,
                                                                                                   len=1,
                                                                                                   mult=1)))


def create_experiment_id(lm_training_config: LMTrainingConfig):
    return str(time.time())


ex = Experiment(create_experiment_id(lm_training_config))

from sacred.observers import MongoObserver
from langmodels.mongocreds_test import USER, PASSWORD, HOST, PORT, DATABASE

ex.observers.append(MongoObserver.create(url=f'mongodb://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}', db_name=DATABASE))


# ex.add_config(jsons.dump(lm_training_config))


@ex.config
def cfg():
    config = lm_training_config


def deepcopy(read_only_obj: Any):
    if isinstance(read_only_obj, ReadOnlyDict):
        return {deepcopy(k): deepcopy(v) for k, v in read_only_obj.items()}

    if isinstance(read_only_obj, ReadOnlyList):
        return {deepcopy(elm) for elm in read_only_obj}

    return copy.deepcopy(read_only_obj)


@ex.capture
def prep_and_train(config, output_path):
    prep_corpus: api.PreprocessedCorpus = config.prep_function.apply(config.corpus, output_path=output_path)

    train(prep_corpus=prep_corpus,
          path_to_model=os.path.join(PATH_TO_TRAINED_MODELS, encode_config('10k')),
          device=cuda.current_device(),
          lm_training_config=config,
          use_pretrained_model=False,
          sacred_exp=ex)


@ex.automain
def my_main():
    prep_and_train(output_path=PATH_TO_PREP_DATASETS)
