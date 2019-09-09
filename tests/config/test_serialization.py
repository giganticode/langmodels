import unittest

import dataprep.api.corpus as api

from langmodels.lmconfig.datamodel import LMTrainingConfig, LstmArch, TrainingProcedure, \
    PrepFunction, Corpus
from langmodels.lmconfig.serialization import dump_config, load_config_from_string


class DatamodelTest(unittest.TestCase):
    def test_serialize_deserialize_defaults(self):
        lm_training_config = LMTrainingConfig(
            base_model=None,
            corpus=Corpus(path='/home/hlib/dev/yahtzee',
                            extensions="java"),
            prep_function=PrepFunction(
                api.bpe,
                ['10k'],
                {'no_str': True, 'no_com': True, 'no_spaces': False, 'no_unicode': False}
            ),
            arch=LstmArch(), bs=32, bptt=200,
            training_procedure=TrainingProcedure())

        self.assertEqual(lm_training_config, load_config_from_string(dump_config(lm_training_config)))


if __name__ == '__main__':
    unittest.main()
