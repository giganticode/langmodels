import sys

import dataprep.api.corpus as api

from langmodels.lmconfig.datamodel import LMTrainingConfig, LstmArch, TrainingProcedure, \
    PrepFunction, Corpus, CosineLRSchedule, PrepFunctionOptions
from langmodels.lmconfig.serialization import load_config_from_string, dump_config_to_string


def test_serialize_deserialize_defaults():
    lm_training_config = LMTrainingConfig(
        base_model=None,
        corpus=Corpus(path='/path/to/corpus', extensions="java"),
        prep_function=PrepFunction(
            api.bpe,
            ['10k'],
            PrepFunctionOptions(no_str=True, no_com=True, no_spaces=False, no_unicode=False, max_str_length=sys.maxsize)
        ),
        arch=LstmArch(), bs=32, bptt=200,
        training_procedure=TrainingProcedure(schedule=CosineLRSchedule(cyc_len=3, max_epochs=30, max_lr=1e-4)))

    assert lm_training_config == load_config_from_string(dump_config_to_string(lm_training_config))
