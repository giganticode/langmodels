import logging
import os

import dill as pickle
import torch
from fastai.text import AWD_LSTM, SequentialRNN, get_language_model, awd_lstm_lm_config, F
from typing import List, Generator

from torchtext.data import Field

from langmodels.migration import weights_to_v1

logger = logging.getLogger(__name__)

TEXT_FIELD_FILE = 'VOCAB.pkl'
WEIGHTS_FILE = 'models/best.h5'
MODEL_ZOO_PATH_ENV_VAR = 'MODEL_ZOO_PATH'

# TODO this params should not be hardcoded:
MODEL_NAME = '336101/100_baseline'
BPE_CODES_ID = '10k'

VOCAB_SIZE = 10096
EMB_SZ = 300
N_HID = 650
N_LAYERS = 3


if MODEL_ZOO_PATH_ENV_VAR in os.environ:
    MODEL_ZOO_PATH = os.environ[MODEL_ZOO_PATH_ENV_VAR]
else:
    MODEL_ZOO_PATH = os.path.join(os.environ['HOME'], 'modelzoo')

class TrainedModel(object):
    STARTING_TOKENS = ['``']

    def __init__(self, path: str):
        self.model = self._load_model(path)
        self._to_test_mode()
        self.legacy_vocab = self._load_vocab(path)
        self.last_predicted_tokens = self.legacy_vocab.numericalize([self.STARTING_TOKENS], -1)

    def _load_model(self, path: str) -> SequentialRNN:
        path_to_model = os.path.join(path, WEIGHTS_FILE)
        logger.debug(f"Loading model from: {path_to_model} ...")

        awd_lstm_lm_config.update({'emb_sz': EMB_SZ, 'n_hid': N_HID, 'n_layers': N_LAYERS, 'out_bias': False})
        model = get_language_model(AWD_LSTM, VOCAB_SIZE, awd_lstm_lm_config)
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights_to_v1(state_dict), strict=True)

        return model

    def _load_vocab(self, path: str) -> Field:
        path_to_vocab = os.path.join(path, TEXT_FIELD_FILE)
        logger.debug(f"Loading vocab from: {path_to_vocab} ...")
        return pickle.load(open(path_to_vocab, 'rb'))

    def _to_test_mode(self):
        # Set batch size to 1
        self.model[0].bs = 1
        # Turn off dropout
        self.model.eval()
        # Reset hidden state
        self.model.reset()

    @classmethod
    def from_path(cls, path: str) -> 'TrainedModel':
        return cls(path)

    @classmethod
    def get_default_model(cls) -> 'TrainedModel':
        path = os.path.join(MODEL_ZOO_PATH, MODEL_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f'The path does not exist: {path}. ' 
                                    f'Did you set {MODEL_ZOO_PATH_ENV_VAR} env var correctly? '
                                    f'Your current {MODEL_ZOO_PATH_ENV_VAR} path is {MODEL_ZOO_PATH}')
        return cls.from_path(path)

    def get_bpe_codes_id(self):
        return BPE_CODES_ID

    def predict_next(self, how_many: int = 1) -> Generator[str, None, None]:
        for i in range(how_many):
            res, *_ = self.model(self.last_predicted_tokens)
            n = torch.multinomial(res[-1].exp(), 1)
            yield self.legacy_vocab.vocab.itos[n[0]]
            self.last_predicted_tokens = n[0].unsqueeze(0)

    def get_entropy_for_next(self, input: List[str]) -> float:
        numericalized_input = self.legacy_vocab.numericalize([input], -1)
        loss_sum = 0
        for token in numericalized_input:
            res, *_ = self.model(self.last_predicted_tokens)
            loss = F.cross_entropy(res[-1], token).item()
            self.last_predicted_tokens = token.unsqueeze(0)
            loss_sum += loss
        return loss_sum / len(input)
