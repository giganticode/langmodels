import logging
import os
from typing import List, Generator, Dict, Any

import torch
from dataprep.api.corpus import PreprocessedCorpus
from dataprep.parse.model.placeholders import placeholders
from fastai.text import AWD_LSTM, SequentialRNN, get_language_model, F, Vocab, awd_lstm_lm_config
from fastai.text.models.transformer import init_transformer
from torch import cuda

from langmodels.lmconfig.datamodel import Corpus, LstmArch, TransformerArch, LMTrainingConfig
from langmodels.lmconfig.serialization import load_config_from_file
from langmodels.migration import pth_to_torch

logger = logging.getLogger(__name__)

MODEL_ZOO_PATH_ENV_VAR = 'MODEL_ZOO_PATH'

DEFAULT_MODEL_NAME = 'langmodel-small-split_10k_1_512_190906.154943'

PAD_TOKEN_INDEX = 1


if MODEL_ZOO_PATH_ENV_VAR in os.environ:
    MODEL_ZOO_PATH = os.environ[MODEL_ZOO_PATH_ENV_VAR]
else:
    MODEL_ZOO_PATH = os.path.join(os.environ['HOME'], 'modelzoo')


def create_custom_lstm_config(arch: LstmArch):
    config = awd_lstm_lm_config
    config['emb_sz'] = arch.emb_sz
    config['n_hid'] = arch.n_hid
    config['n_layers'] = arch.n_layers
    config['pad_token'] = PAD_TOKEN_INDEX
    config['qrnn'] = arch.qrnn
    config['bidir'] = arch.bidir
    config['output_p'] = arch.drop.out
    config['hidden_p'] = arch.drop.outh
    config['input_p'] = arch.drop.outi
    config['embed_p'] = arch.drop.oute
    config['weight_p'] = arch.drop.w
    config['tie_weights'] = arch.tie_weights
    config['out_bias'] = arch.out_bias
    return config


def create_custom_transformer_config(arch: TransformerArch) -> Dict[str, Any]:
    d = arch.__dict__
    d['init'] = init_transformer
    return d

def create_custom_config(lm_training_config: LMTrainingConfig):
    arch = lm_training_config.arch
    if isinstance(arch, LstmArch):
        return create_custom_lstm_config(arch)
    elif isinstance(arch, TransformerArch):
        return create_custom_transformer_config(arch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_device(force_use_cpu: bool) -> int:
    return cuda.current_device() if cuda.is_available() and not force_use_cpu else -1


class TrainedModel(object):
    STARTING_TOKENS = [placeholders['ect']]

    def __init__(self, path: str, force_use_cpu: bool = False):
        self.force_use_cpu = force_use_cpu
        self.vocab = Vocab.load(os.path.join(path, 'vocab'))
        self.model = self._load_model(path)
        self._to_test_mode()
        self.last_predicted_tokens = torch.tensor([self.vocab.numericalize(self.STARTING_TOKENS)])

    def _load_model(self, path: str) -> SequentialRNN:
        path_to_model = os.path.join(path, 'best.pth')
        logger.debug(f"Loading model from: {path_to_model} ...")

        config = load_config_from_file(os.path.join(path, 'config'))
        self._prep_function = config.prep_function

        model = get_language_model(AWD_LSTM, len(self.vocab.itos), create_custom_lstm_config(config.arch))
        if self.force_use_cpu:
            map_location = lambda storage, loc: storage
            logger.debug("Using CPU for inference")
        elif cuda.is_available():
            map_location = None
            model.cuda()
            logger.debug("Using GPU for inference")
        else:
            map_location = lambda storage, loc: storage
            logger.info("Cuda not available. Falling back to using CPU.")

        state_dict = torch.load(path_to_model, map_location=map_location)

        # a more simple solution is to use fastai's load_learner,
        # however it doesn't seem to work out of the box wiht customizations we've done
        model.load_state_dict(pth_to_torch(state_dict), strict=True)

        return model

    def _to_test_mode(self):
        # Set batch size to 1
        self.model[0].bs = 1
        # Turn off dropout
        self.model.eval()
        # Reset hidden state
        self.model.reset()

    @classmethod
    def from_path(cls, path: str, force_use_cpu: bool = False) -> 'TrainedModel':
        return cls(path, force_use_cpu)

    @classmethod
    def get_default_model(cls, force_use_cpu: bool = False) -> 'TrainedModel':
        path = os.path.join(MODEL_ZOO_PATH, DEFAULT_MODEL_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f'The path does not exist: {path}. ' 
                                    f'Did you set {MODEL_ZOO_PATH_ENV_VAR} env var correctly? '
                                    f'Your current {MODEL_ZOO_PATH_ENV_VAR} path is {MODEL_ZOO_PATH}')
        return cls.from_path(path, force_use_cpu)

    def prep_corpus(self, corpus: Corpus, *kwargs) -> PreprocessedCorpus:
        return self._prep_function.apply(corpus, kwargs)

    def prep_text(self, text, **kwargs):
        import dataprep.api.text as text_api
        text_callable = getattr(text_api, self._prep_function.callable.__name__)
        return text_callable(text, *self._prep_function.params, **self._prep_function.options, **kwargs)

    def predict_next(self, how_many: int = 1) -> Generator[str, None, None]:
        for i in range(how_many):
            res, *_ = self.model(self.last_predicted_tokens)
            n = torch.multinomial(res[-1].exp(), 1)
            yield self.vocab.itos[n[0]]
            self.last_predicted_tokens = n[0].unsqueeze(0)

    def get_entropies_for_next(self, input: List[str]) -> List[float]:
        numericalized_input = torch.tensor([self.vocab.numericalize(input)]).t()
        assert len(numericalized_input.size()) == 2
        losses: List[float] = []
        for _, num_token in zip(input, numericalized_input):
            res, *_ = self.model(self.last_predicted_tokens)
            last_layer = res[-1]
            loss = F.cross_entropy(last_layer, num_token).item()
            self.last_predicted_tokens = num_token.unsqueeze(0)
            losses.append(loss)
        return losses

    # TODO
    # def predict_next_full_token(self):
    #     pass
    #
    # def test(self, corpus: Corpus):
    #     prep_corpus = self.prep_corpus(corpus)

