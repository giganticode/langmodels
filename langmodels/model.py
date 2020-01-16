import logging
import os
from collections import OrderedDict
from threading import Lock
from typing import List, Dict, Any, Tuple, Optional, Union, Type, Generator

import torch
from dataclasses import dataclass, asdict
from fastai.text import SequentialRNN, get_language_model, F, Vocab, awd_lstm_lm_config, convert_weights
from fastai.text.models.transformer import init_transformer
from math import exp
from torch import cuda

from dataprep.api.corpus import PreprocessedCorpus
from dataprep.pipeline.dataset import normalize_extension_string
from dataprep.preprocess.metadata import check_metadata_validity, PreprocessingMetadata
from dataprep.preprocess.placeholders import placeholders
from dataprep.subtokens import is_terminal_subtoken, FullTokenIterator, SubtokenIterator
from langmodels.beamsearch import beam_search
from langmodels.cuda_util import get_device, get_map_location
from langmodels.lmconfig.datamodel import Corpus, LstmArch, TransformerArch, LMTrainingConfig, GruArch, \
    LMTrainingMetrics
from langmodels.lmconfig.serialization import load_config_or_metrics_from_file, read_value_from_file
from langmodels.nn import to_test_mode, get_last_layer_activations, TORCH_LONG_MIN_VAL
from langmodels.util import to_binary_entropy

logger = logging.getLogger(__name__)


PAD_TOKEN_INDEX = 1

BEST_MODEL_FILE_NAME = 'best.pth'
METRICS_FILE_NAME = 'metrics'
CONFIG_FILE_NAME = 'config'
VOCAB_FILE_NAME = 'vocab'
TAGS_FILE_NAME = 'tags'


def create_custom_lstm_or_gru_config(arch: Union[GruArch, LstmArch]):
    config = awd_lstm_lm_config
    config['emb_sz'] = arch.emb_sz
    config['n_hid'] = arch.n_hid
    config['n_layers'] = arch.n_layers
    config['pad_token'] = PAD_TOKEN_INDEX
    if isinstance(arch, LstmArch):
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
    d = {'init': init_transformer,
         'ctx_len': arch.ctx_len,
         'n_layers': arch.n_layers,
         'n_heads': arch.n_heads,
         'd_model': arch.d_model,
         'd_head': arch.d_head,
         'd_inner': arch.d_inner,
         'resid_p': arch.drop.resid,
         'attn_p': arch.drop.attn,
         'ff_p': arch.drop.ff,
         'embed_p': arch.drop.embed,
         'output_p': arch.drop.output,
         'bias': arch.bias,
         'scale': arch.scale,
         'act': arch.act,
         'double_drop': arch.double_drop,
         'tie_weights': arch.tie_weights,
         'out_bias': arch.out_bias,
         'mask': arch.mask,
    }
    return d


def create_custom_config(lm_training_config: LMTrainingConfig):
    arch = lm_training_config.arch
    if isinstance(arch, LstmArch) or isinstance(arch, GruArch):
        return create_custom_lstm_or_gru_config(arch)
    elif isinstance(arch, TransformerArch):
        return create_custom_transformer_config(arch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def _create_term_vocab(vocab: Vocab) -> Tuple[Vocab, int]:
    terminal_token_indices = {i for i, k in enumerate(vocab.itos) if is_terminal_subtoken(k)}
    term_vocab_list = [vocab.itos[i] for i in terminal_token_indices]
    non_term_vocab_list = [vocab.itos[i] for i in range(len(vocab.itos)) if i not in terminal_token_indices]
    term_vocab = Vocab(term_vocab_list + non_term_vocab_list)
    return term_vocab, len(term_vocab_list)


def _check_is_term_vocab(vocab: Vocab, first_non_term: int) -> None:
    for token in vocab.itos[:first_non_term]:
        if not is_terminal_subtoken(token):
            raise ValueError(f"First non-terminal token in vocab has index {first_non_term}, "
                             f"however there's a non-terminal token {token} "
                             f"that has index {vocab.itos.index(token)}")
    for token in vocab.itos[first_non_term:]:
        if is_terminal_subtoken(token):
            raise ValueError(f"Starting from index {first_non_term} there should bb only non-term "
                             f"tokens in the vocab, however there's a terminal token {token} "
                             f"that has index {vocab.itos.index(token)}")


def to_full_token_string(subtokens: List[str], include_debug_tokens=False) -> str:
    """
    >>> to_full_token_string(['the', 're</t>'], include_debug_tokens=True)
    'the|re</t>'

    >>> to_full_token_string(['re', 'vol', 'v', 'er</t>'])
    'revolver</t>'

    >>> to_full_token_string([placeholders['olc_end']])
    '<EOL>'
    """
    separator = '|' if include_debug_tokens else ''
    full_token = separator.join(subtokens)
    if full_token in placeholders.values():
        return full_token

    if not is_terminal_subtoken(full_token):
        raise ValueError(f'{full_token} is not a full token')
    return full_token


PredictionList = List[Tuple[str, float]]

@dataclass
class ModelDescription(object):
    id: str
    bpe_merges: str
    layers_config: str
    arch: str
    bin_entropy: float
    training_time_minutes_per_epoch: int
    n_epochs: int
    best_epoch: int
    size_on_disk_mb: int
    tags: List[str]

    def is_tagged_by(self, tag: str) -> bool:
        return tag in self.tags

    @staticmethod
    def get_attribute_list() -> List[str]:
        return [k for k in ModelDescription.__annotations__.keys()]

    def get_value_list(self) -> List[str]:
        return list(map(lambda a: self.__getattribute__(a), ModelDescription.get_attribute_list()))


lock = Lock()


class TrainedModel(object):
    STARTING_TOKEN = placeholders['ect']

    def __init__(self, path: str, force_use_cpu: bool = False, load_only_description: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path does not exist: {path}')
        self._force_use_cpu = force_use_cpu
        self._id = os.path.basename(path)
        path_to_config_file = os.path.join(path, CONFIG_FILE_NAME)
        path_to_metrics_file = os.path.join(path, METRICS_FILE_NAME)
        path_to_tags_file = os.path.join(path, TAGS_FILE_NAME)
        self._metrics = None
        self._config = None
        self._tags = []
        self._context: List[str] = []
        try:
            self._config: LMTrainingConfig = load_config_or_metrics_from_file(path_to_config_file, LMTrainingConfig)
        except FileNotFoundError:
            logger.warning(f'Config file not found: {path_to_config_file}')
        try:
            self._metrics: LMTrainingMetrics = load_config_or_metrics_from_file(os.path.join(path, METRICS_FILE_NAME), LMTrainingMetrics)
        except FileNotFoundError:
            logger.warning(f'File with metrics not found: {path_to_metrics_file}')
        if os.path.exists(path_to_tags_file):
            value = read_value_from_file(path_to_tags_file, value_type=str)
            if value != '':
                self._tags = value.split(',')
        self._prep_function = self._config.prep_function

        self._load_only_description = load_only_description
        if not load_only_description:
            # we might want to load only description without loading actual weights when we want
            # to save time when loading multiple models to choose one of them to work with

            self._original_vocab = Vocab.load(os.path.join(path, VOCAB_FILE_NAME))
            term_vocab, self._first_nonterm_token = _create_term_vocab(self._original_vocab)
            self._model, self._vocab = self._load_model(path, term_vocab)
            to_test_mode(self._model)

            # last_predicted_token_tensor is a rank-2 tensor!
            self._last_predicted_token_tensor = torch.tensor([self._vocab.numericalize([self.STARTING_TOKEN])],
                                                             device=get_device(self._force_use_cpu))

    @property
    def id(self):
        return self._id

    @property
    def metrics(self):
        return self._metrics

    @property
    def config(self):
        return self._config

    @property
    def tags(self):
        return self._tags

    @property
    def context(self):
        return self._context

    @property
    def model(self):
        return self._model

    @property
    def vocab(self):
        return self._vocab

    def _load_model(self, path: str, custom_vocab: Optional[Vocab] = None) -> Tuple[SequentialRNN, Vocab]:
        path_to_model = os.path.join(path, BEST_MODEL_FILE_NAME)
        logger.debug(f"Loading model from: {path_to_model} ...")

        vocab = custom_vocab if custom_vocab else self._original_vocab
        model = get_language_model(self._config.arch.get_module(), len(vocab.itos), create_custom_config(self._config))
        map_location = get_map_location(self._force_use_cpu)
        if cuda.is_available():
            model.cuda()
        state_dict = torch.load(path_to_model, map_location=map_location)

        # a more simple solution is to use fastai's load_learner,
        # however it doesn't seem to work out of the box with customizations we've done

        # update: it appeared that it's quite simple to load weights. So maybe not worth it loading a learner?
        weights = OrderedDict(state_dict['model'] if ('model' in state_dict) else state_dict)
        if custom_vocab:
            weights = convert_weights(weights, self._original_vocab.stoi, custom_vocab.itos)
        model.load_state_dict(weights, strict=True)

        return model, vocab

    BEAM_SIZE = 500
    SAVE_CONTEXT_LIMIT = 1000

    def _save_context(self, prep_tokens: List[str]) -> None:
        self._context.extend(prep_tokens)
        if len(self._context) > TrainedModel.SAVE_CONTEXT_LIMIT:
            self._context = self._context[-TrainedModel.SAVE_CONTEXT_LIMIT:]

    def reset_context(self) -> None:
        self._context = []

    def prep_corpus(self, corpus: Corpus, **kwargs) -> PreprocessedCorpus:
        return self._prep_function.apply(corpus, **kwargs)

    def prep_text(self, text: str, extension: str, **kwargs) -> Union[Tuple[List[str], PreprocessingMetadata], List[str]]:
        import dataprep.api.text as text_api
        return_metadata: bool = 'return_metadata' in kwargs and kwargs['return_metadata']
        text_callable = getattr(text_api, self._prep_function.callable.__name__)
        preprocessing_result = text_callable(text, extension=extension, force_reinit_bpe_data=False,
                                            *self._prep_function.params, **asdict(self._prep_function.options), **kwargs)
        if return_metadata:
            check_metadata_validity(*preprocessing_result)
        return preprocessing_result

    def get_predictions_and_feed(self, text: str, extension: str, n_suggestions: int, append_eof: bool)\
            -> Generator[Tuple[PredictionList, str, Type], None, None]:
        prep_text, metadata = self.prep_text(text, extension, return_metadata=True, append_eof=append_eof)

        for ind, prep_token in FullTokenIterator(prep_text, metadata.word_boundaries, return_full_token_index=True):
            predictions = self.predict_next_full_token(n_suggestions)
            self._feed_prep_tokens([prep_token])

            yield predictions, prep_token, metadata.token_types[ind]

    def _check_model_loaded(self, only_description=False):
        if not only_description and self._load_only_description:
            raise RuntimeError("Operation not supported. Only model's description is loaded. "
                               "Prease reload the model with param load_only_description set to False.")

    def _feed_prep_tokens(self, prep_tokens: List[str]) -> None:
        self._check_model_loaded()
        context_tensor = torch.tensor([self._vocab.numericalize(prep_tokens)], device=get_device(self._force_use_cpu))
        with lock:
            self._save_context(prep_tokens)
            _ = get_last_layer_activations(self._model, context_tensor[:, :-1])
            self._last_predicted_token_tensor = context_tensor[:, -1:]

    def check_inference_possible_for_file_type(self, extension: str) -> bool:
        try:
            self._assert_inference_possible_for_file_type(extension)
            return True
        except ValueError:
            return False

    def feed_text(self, text: str, extension: str) -> None:
        self._assert_inference_possible_for_file_type(extension)

        prep_text, metadata = self.prep_text(text, extension=extension, return_metadata=True)
        self._feed_prep_tokens(prep_text)

    def get_entropies_for_text(self, text: str, extension: str, full_tokens: bool, append_eof: bool) \
            -> Tuple[List[float], List[str], List[Type]]:
        tokens, metadata = self.prep_text(text, extension, return_metadata=True, append_eof=append_eof)
        subtoken_entropies = self._get_entropies_for_prep_text(tokens)

        iterator_type = FullTokenIterator if full_tokens else SubtokenIterator

        def formatter(lst: List[Tuple[float, str]]) -> Tuple[float, str]:
            entropies, tokens = tuple(zip(*lst))
            return sum(entropies), "".join(tokens)

        iterator = iterator_type(list(zip(subtoken_entropies, tokens)), metadata.word_boundaries,
                                 format=formatter, return_full_token_index=True)
        e, t, tt = [], [], []
        for ind, (entropy, token) in iterator:
            e.append(entropy)
            t.append(token)
            tt.append(metadata.token_types[ind])
        return e, t, tt


    def _get_entropies_for_prep_text(self, prep_text: List[str]) -> List[float]:
        """
        changes hidden states of the model!!
        """
        with lock:
            self._check_model_loaded()

            if not prep_text:
                return []

            loss_list = []
            max_subtokens_per_chunk = 200
            # if the line is too big, we break it down to chunks to fit it into gpu memory
            # big chunks require more memory, small chunks require more time
            n_chunks = (len(prep_text)-1) // max_subtokens_per_chunk + 1
            for i in range(n_chunks):
                pt = prep_text[i*max_subtokens_per_chunk:(i+1)*max_subtokens_per_chunk]
                numericalized_prep_text = torch.tensor([self._vocab.numericalize(pt)],
                                                       device=get_device(self._force_use_cpu))

                self._save_context(pt)
                last_layer = get_last_layer_activations(self._model, torch.cat([self._last_predicted_token_tensor, numericalized_prep_text[:, :-1]], dim=1))
                loss = F.cross_entropy(last_layer.view(-1, last_layer.shape[-1]),
                                       numericalized_prep_text.view(-1),
                                       reduction='none')
                binary_loss = to_binary_entropy(loss)
                loss_list.extend(binary_loss.tolist())
                self._last_predicted_token_tensor = numericalized_prep_text[:, -1:]
            return loss_list

    def reset(self) -> None:
        with lock:
            self._check_model_loaded()

            self._model.reset()
            self.reset_context()
            self._last_predicted_token_tensor = torch.tensor([self._vocab.numericalize([self.STARTING_TOKEN])],
                                                             device=get_device(self._force_use_cpu))

    def predict_next_full_token(self, n_suggestions: int = 1, include_debug_tokens: bool = False) -> PredictionList:
        with lock:
            self._check_model_loaded()

            numericalized_subtokens_list, scores = beam_search(self._model, self._last_predicted_token_tensor[0], self._first_nonterm_token, n_suggestions, self.BEAM_SIZE)
            suggestions: PredictionList = []
            for numericalized_subtokens, score in zip(numericalized_subtokens_list, scores):
                try:
                    start_of_empty_numbers = numericalized_subtokens.tolist().index(TORCH_LONG_MIN_VAL)
                except ValueError:
                    start_of_empty_numbers = len(numericalized_subtokens)
                numericalized_subtokens = numericalized_subtokens[:start_of_empty_numbers]
                subtokens = self._vocab.textify(numericalized_subtokens, sep=None)
                full_token = (to_full_token_string(subtokens, include_debug_tokens))
                suggestions.append((full_token,  1 / exp(score.item())))
            return suggestions

    def _format_layers_config(self) -> str:
        if isinstance(self._config.arch, TransformerArch):
            return "transformer"  # TODO add proper layer description
        n_layers = self._config.arch.n_layers
        emb_size = self._config.arch.emb_sz
        n_hid = self._config.arch.n_hid
        return f'{emb_size}/{n_layers}/{n_hid}={self._metrics.trainable_params}'

    def get_model_description(self) -> ModelDescription:
        return ModelDescription(id=self._id,
                                bpe_merges=self._config.prep_function.params[0],
                                layers_config=self._format_layers_config(),
                                arch=str(self._config.arch.get_module().__name__),
                                bin_entropy=self._metrics.bin_entropy if self._metrics else 1e+6,
                                training_time_minutes_per_epoch=self._metrics.training_time_minutes_per_epoch
                                if self._metrics else 0,
                                n_epochs=self._metrics.n_epochs if self._metrics else 0,
                                best_epoch=self._metrics.best_epoch if self._metrics else -1,
                                size_on_disk_mb=self._metrics.size_on_disk_mb,
                                tags=self._tags)

    def _assert_inference_possible_for_file_type(self, extension: str) -> None:
        if extension not in normalize_extension_string(self._config.corpus.extensions):
            raise ValueError(f'The model was not trained on .{extension} files. Cannot do inference.')

    def __str__(self) -> str:
        return str(self.get_model_description())
