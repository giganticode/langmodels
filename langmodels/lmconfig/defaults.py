import dataprep.api.corpus as api

from langmodels.lmconfig.datamodel import *

HOME = os.environ['HOME']

lm_training_config = LMTrainingConfig(base_model=None,
                                      corpus=Corpus(path=os.path.join(HOME, 'raw_datasets/dev'), extensions="java|c"),
                                        prep_function=PrepFunction(api.bpe, ['10k'],
                                                                  {
                                                                  'no_com': False,
                                                                  'no_unicode': True,
                                                                  'no_spaces': True,
                                                                 # 'max_str_length': 14,
                                                                  }),
                                      arch=LstmArch(n_layers=1, emb_sz=512, n_hid=512),
                                      bs=64, bptt=200, training_procedure=TrainingProcedure(
        CosineLRSchedule(max_epochs=30, cyc_len=3, max_lr=1e-4)
    )
                                      )
