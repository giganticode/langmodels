import logging
import os

from dataprep.parse.model.metadata import PreprocessingMetadata
from math import exp
from typing import List, Dict, Any, Tuple, Optional

import torch
from dataprep.api.corpus import PreprocessedCorpus
from dataprep.parse.model.placeholders import placeholders
from fastai.text import AWD_LSTM, SequentialRNN, get_language_model, F, Vocab, awd_lstm_lm_config, convert_weights
from fastai.text.models.transformer import init_transformer
from torch import cuda

from langmodels.beamsearch import beam_search
from langmodels.lmconfig.datamodel import Corpus, LstmArch, TransformerArch, LMTrainingConfig
from langmodels.lmconfig.serialization import load_config_from_file
from langmodels.migration import pth_to_torch
from langmodels.nn import to_test_mode, get_last_layer_activations, TORCH_LONG_MIN_VAL, to_binary_entropy
from langmodels.gpuutil import get_device

logger = logging.getLogger(__name__)

MODEL_ZOO_PATH_ENV_VAR = 'MODEL_ZOO_PATH'

RANDOM_MODEL_NAME = 'dev_10k_1_10_190923.132328'
TINY_MODEL_NAME = 'langmodel-small-split_10k_1_512_190906.154943'
SMALL_MODEL_NAME = 'langmodel-medium-split_10k_1_512_190924.154542_-_langmodel-medium-split_10k_1_512_190925.141033'

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


def _create_term_vocab(vocab: Vocab) -> Tuple[Vocab, int]:
    terminal_token_indices = {i for i, k in enumerate(vocab.itos) if k.endswith(placeholders['compound_word_end'])}
    term_vocab_list = [vocab.itos[i] for i in terminal_token_indices]
    non_term_vocab_list = [vocab.itos[i] for i in range(len(vocab.itos)) if i not in terminal_token_indices]
    term_vocab = Vocab(term_vocab_list + non_term_vocab_list)
    return term_vocab, len(term_vocab_list)


def _check_is_term_vocab(vocab: Vocab, first_non_term: int) -> None:
    for token in vocab.itos[:first_non_term]:
        if not token.endswith(placeholders['compound_word_end']):
            raise ValueError(f"First non-terminal token in vocab has index {first_non_term}, "
                             f"hovewer there's a non-terminal token {token} "
                             f"that has index {vocab.itos.index(token)}")
    for token in vocab.itos[first_non_term:]:
        if token.endswith(placeholders['compound_word_end']):
            raise ValueError(f"Starting from index {first_non_term} there should bb only non-term "
                             f"tokens in the vocab, hovewer there's a terminal token {token} "
                             f"that has index {vocab.itos.index(token)}")


DEFAULT_BEAM_SIZE = 500


class TrainedModel(object):
    STARTING_TOKEN = placeholders['ect']

    def __init__(self, path: str, force_use_cpu: bool = False):
        self.force_use_cpu = force_use_cpu
        self.original_vocab = Vocab.load(os.path.join(path, 'vocab'))
        term_vocab, self.first_nonterm_token = _create_term_vocab(self.original_vocab)
        self.model, self.vocab = self._load_model(path, term_vocab)
        to_test_mode(self.model)

        # last_predicted_token_tensor is a rank-2 tensor!
        self.last_predicted_token_tensor = torch.tensor([self.vocab.numericalize([self.STARTING_TOKEN])], device=get_device())
        self.beam_size = DEFAULT_BEAM_SIZE

    def _load_model(self, path: str, custom_vocab: Optional[Vocab] = None) -> Tuple[SequentialRNN, Vocab]:
        path_to_model = os.path.join(path, 'best.pth')
        logger.debug(f"Loading model from: {path_to_model} ...")

        config = load_config_from_file(os.path.join(path, 'config'))
        self._prep_function = config.prep_function

        vocab = custom_vocab if custom_vocab else self.original_vocab
        model = get_language_model(AWD_LSTM, len(vocab.itos), create_custom_lstm_config(config.arch))
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
        weights = pth_to_torch(state_dict)
        if custom_vocab:
            weights = convert_weights(weights, self.original_vocab.stoi, custom_vocab.itos)
        model.load_state_dict(weights, strict=True)

        return model, vocab

    @classmethod
    def from_path(cls, path: str, force_use_cpu: bool = False) -> 'TrainedModel':
        return cls(path, force_use_cpu)

    @classmethod
    def get_default_model(cls, force_use_cpu: bool = False) -> 'TrainedModel':
        return TrainedModel.get_small_model(force_use_cpu)

    @classmethod
    def get_random_model(cls, force_use_cpu: bool = False) -> 'TrainedModel':
        return TrainedModel.load_model_by_name(RANDOM_MODEL_NAME, force_use_cpu)

    @classmethod
    def get_tiny_model(cls, force_use_cpu: bool = False) -> 'TrainedModel':
        return TrainedModel.load_model_by_name(TINY_MODEL_NAME, force_use_cpu)

    @classmethod
    def get_small_model(cls, force_use_cpu: bool = False) -> 'TrainedModel':
        return TrainedModel.load_model_by_name(SMALL_MODEL_NAME, force_use_cpu)

    @classmethod
    def load_model_by_name(cls, name: str, force_use_cpu: bool = False) -> 'TrainedModel':
        path = os.path.join(MODEL_ZOO_PATH, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'The path does not exist: {path}. ' 
                                    f'Did you set {MODEL_ZOO_PATH_ENV_VAR} env var correctly? '
                                    f'Your current {MODEL_ZOO_PATH_ENV_VAR} path is {MODEL_ZOO_PATH}')
        return cls.from_path(path, force_use_cpu)

    def prep_corpus(self, corpus: Corpus, *kwargs) -> PreprocessedCorpus:
        return self._prep_function.apply(corpus, kwargs)

    def prep_text(self, text, **kwargs) -> Tuple[List[str], PreprocessingMetadata]:
        import dataprep.api.text as text_api
        text_callable = getattr(text_api, self._prep_function.callable.__name__)
        return text_callable(text, *self._prep_function.params, **self._prep_function.options, **kwargs)

    def feed_text(self, text: str) -> None:
        prep_text, metadata = self.prep_text(text, return_metadata=True, force_reinit_bpe_data=False)
        context_tensor = torch.tensor([self.vocab.numericalize(prep_text)], device=get_device())
        _ = get_last_layer_activations(self.model, context_tensor[:,:-1])
        self.last_predicted_token_tensor = context_tensor[:,-1:]

    def get_entropies_for_prep_text(self, prep_text: List[str]) -> List[float]:
        """
        changes hidden states of the model!!
        """
        numericalized_prep_text = torch.tensor([[self.vocab.numericalize(prep_text)]]).transpose(0, 2)

        losses: List[float] = []
        for numericalized_token in numericalized_prep_text:
            # TODO this loop can be avoided! Rnn returns all the hidden states!
            last_layer = get_last_layer_activations(self.model, self.last_predicted_token_tensor)
            loss = F.cross_entropy(last_layer, numericalized_token.squeeze(dim=0)).item()
            binary_loss = to_binary_entropy(loss)
            self.last_predicted_token_tensor = numericalized_token
            losses.append(binary_loss)
        return losses

    def reset(self) -> None:
        self.model.reset()
        self.last_predicted_token_tensor = torch.tensor([self.vocab.numericalize([self.STARTING_TOKEN])],
                                                        device=get_device())

    def predict_next_full_token(self, n_suggestions: int) -> List[Tuple[str, float]]:
        subtokens, scores = beam_search(self.model, self.last_predicted_token_tensor[0], self.first_nonterm_token, n_suggestions, self.beam_size)
        return [(self.to_full_token_string(st), 1 / exp(score.item())) for st, score in zip(subtokens, scores)]

    def to_full_token_string(self, subtokens_num: torch.LongTensor, include_debug_tokens=False) -> str:
        try:
            subtokens_num = subtokens_num[:subtokens_num.tolist().index(TORCH_LONG_MIN_VAL)]
        except ValueError: pass

        sep = '|' if include_debug_tokens else ''
        textified = self.vocab.textify(subtokens_num, sep=sep)
        cwe = placeholders['compound_word_end']
        if not textified.endswith(cwe):
            raise ValueError(f'{textified} ({subtokens_num}) is not a full token')
        return textified if include_debug_tokens else textified[:-len(cwe)]
