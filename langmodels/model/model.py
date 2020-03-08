import logging
import os
import sys
from collections import OrderedDict
from threading import Lock
from typing import List, Tuple, Optional, Union, Type, Generator

import torch
from dataclasses import asdict
from fastai.text import SequentialRNN, get_language_model, F, Vocab, convert_weights
from math import exp
from torch import cuda

from codeprep.api.corpus import PreprocessedCorpus
from codeprep.pipeline.dataset import normalize_extension_string
from codeprep.preprocess.placeholders import placeholders
from codeprep.tokens import PreppedSubTokenSequence, PreppedFullTokenSequence, is_terminal_subtoken

from langmodels.model.beamsearch import beam_search
from langmodels.lmconfig.datamodel import Corpus, TransformerArch, LMTrainingConfig, LMTrainingMetrics
from langmodels.lmconfig.serialization import load_config_or_metrics_from_file, read_value_from_file
from langmodels.model.nn import to_test_mode, get_last_layer_activations, TORCH_LONG_MIN_VAL
from langmodels.util.misc import to_binary_entropy, chunk_prepped_tokens
from langmodels.model.nn import take_hidden_state_snapshot
from langmodels.model.context import ModelContext, ContextInformation
from langmodels.model.config import create_custom_config
from langmodels.model.context import ContextUsage
from langmodels.model.summary import ModelSummary
from langmodels.util.cuda import get_map_location, get_device
from langmodels.model.context import ContextModification, modify_context

logger = logging.getLogger(__name__)

BEST_MODEL_FILE_NAME = 'best.pth'
METRICS_FILE_NAME = 'metrics'
CONFIG_FILE_NAME = 'config'
VOCAB_FILE_NAME = 'vocab'
TAGS_FILE_NAME = 'tags'


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
        self._context: ModelContext = ModelContext()
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
            self._initial_snapshot = take_hidden_state_snapshot(self._model)

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

    def prep_corpus(self, corpus: Corpus, **kwargs) -> PreprocessedCorpus:
        return self._prep_function.apply(corpus, **kwargs)

    def prep_text(self, text: str, extension: str, **kwargs) -> PreppedSubTokenSequence:
        import codeprep.api.text as text_api
        text_callable = getattr(text_api, self._prep_function.callable.__name__)
        prepped_token = text_callable(text, extension=extension, force_reinit_bpe_data=False,
                                            *self._prep_function.params, **asdict(self._prep_function.options), **kwargs)
        return prepped_token

    def get_predictions_and_feed(self, text: str, extension: str, n_suggestions: int, append_eof: bool)\
            -> Generator[Tuple[PredictionList, str, Type], None, None]:
        prepped_tokens = self.prep_text(text, extension, return_metadata=True, append_eof=append_eof)

        for prepped_token, token_type in prepped_tokens.full_tokens_view(return_token_type=True):
            predictions = self.predict_next_full_token(n_suggestions)
            self._feed_prep_tokens(PreppedSubTokenSequence.from_full_token(prepped_token, token_type))

            yield predictions, "".join(prepped_token), token_type

    def _check_model_loaded(self, only_description=False):
        if not only_description and self._load_only_description:
            raise RuntimeError("Operation not supported. Only model's description is loaded. "
                               "Prease reload the model with param load_only_description set to False.")

    def _feed_prep_tokens(self, prepped_tokens: PreppedSubTokenSequence) -> None:
        self._check_model_loaded()
        context_tensor = torch.tensor([self._vocab.numericalize(prepped_tokens.tokens)], device=get_device(self._force_use_cpu))
        with lock:
            self._context.add(prepped_tokens)
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

        prepped_tokens = self.prep_text(text, extension=extension)
        self._feed_prep_tokens(prepped_tokens)

    def get_entropies_for_text(self, text: str, extension: str, full_tokens: bool,
                               append_eof: bool, context_modification: Optional[ContextModification]) \
            -> Tuple[List[float], List[str], List[Type], List[ContextInformation]]:
        current_context_size = self.context.size(full_tokens)
        max_context_allowed = context_modification.max_context_length if context_modification else sys.maxsize
        if current_context_size > max_context_allowed:
            raise ValueError(f"The length of the current context ({current_context_size}) is larger "
                             f"than the max allowed that was set ({max_context_allowed}).\n"
                             f"Please reset the trainedmodel's context")

        prepped_tokens = self.prep_text(text, extension, append_eof=append_eof)

        prepped_tokens = prepped_tokens.full_tokens_view(formatter=lambda x: "".join(x)) if full_tokens else prepped_tokens
        chunked_prepped_tokens = chunk_prepped_tokens(prepped_tokens, max_context_allowed - current_context_size, max_context_allowed)
        context_usage = ContextUsage(length_start=current_context_size,
                                     reset_at=max_context_allowed,
                                     reset_times=len(chunked_prepped_tokens) - 1,
                                     length_end=len(chunked_prepped_tokens[-1]) if len(chunked_prepped_tokens) > 1 else len(chunked_prepped_tokens[-1]) + current_context_size,
                                     shuffle_interval=context_modification.get_absolute_shuffle_interval() if context_modification else None)
        if context_modification:
            chunked_prepped_tokens = list(map(lambda x: modify_context(x, context_modification), chunked_prepped_tokens))
        subtoken_entropies = self._get_entropies_for_prep_text(
            chunked_prepped_tokens,
            reset_after_last_chunk=context_usage.is_last_chunk_complete()
        )

        tokens = [i for i in prepped_tokens]
        all_token_types = [i for i in prepped_tokens.get_iterator(prepped_tokens.metadata.token_types, over_full_tokens=True)]
        all_entropies = [i for i in prepped_tokens.get_iterator(subtoken_entropies, over_full_tokens=False, formatter=sum)]
        context_information = [i for i in prepped_tokens.get_iterator(context_usage, over_full_tokens=True)]

        return all_entropies, tokens, all_token_types, context_information

    def _get_entropies_for_prep_text(self, prepped_token_sequence_chunks: Union[List[PreppedSubTokenSequence], List[PreppedFullTokenSequence]],
                                     reset_after_last_chunk: bool) -> List[float]:
        """
        changes hidden states of the model!!
        """
        max_subtokens_per_chunk = 200

        with lock:
            self._check_model_loaded()

        if len(prepped_token_sequence_chunks) == 0:
            return []

        loss_list = []

        # if the line is too big, we break it down to chunks to fit it into gpu memory
        # big chunks require more memory, small chunks require more time
        for ind, prepped_token_sequence in enumerate(prepped_token_sequence_chunks):
            prepped_sub_token_sequence = prepped_token_sequence.sub_token_view()
            for i in range(0, len(prepped_sub_token_sequence), max_subtokens_per_chunk):
                sub_chunk = prepped_sub_token_sequence[i:i+max_subtokens_per_chunk]
                numericalized_prep_text = torch.tensor([self._vocab.numericalize(sub_chunk.tokens)],
                                                       device=get_device(self._force_use_cpu))

                inp = torch.cat([self._last_predicted_token_tensor, numericalized_prep_text[:, :-1]], dim=1)
                last_layer = get_last_layer_activations(self._model, inp)
                loss = F.cross_entropy(last_layer.view(-1, last_layer.shape[-1]),
                                       numericalized_prep_text.view(-1),
                                       reduction='none')
                binary_loss = to_binary_entropy(loss)
                loss_list.extend(binary_loss.tolist())
                self._last_predicted_token_tensor = numericalized_prep_text[:, -1:]
            self._context.add(prepped_sub_token_sequence)
            if ind < len(prepped_token_sequence_chunks) - 1 or reset_after_last_chunk:
                self._reset()
        return loss_list

    def _reset(self) -> None:
        self._model.reset()
        self._context.reset()

    def reset(self) -> None:
        with lock:
            self._check_model_loaded()
            self._reset()
            self._last_predicted_token_tensor = torch.tensor([self._vocab.numericalize([self.STARTING_TOKEN])],
                                                             device=get_device(self._force_use_cpu))

    def predict_next_full_token(self, n_suggestions: int = 1, include_debug_tokens: bool = False, max_prob: float = 0.05) -> PredictionList:
        with lock:
            self._check_model_loaded()

            def complete_token_predicate(last_predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                full_token_flags_sorted = last_predictions[:, -1] < self._first_nonterm_token
                ready_candidate_idxs = full_token_flags_sorted.nonzero().squeeze(dim=1)
                pending_candidate_idxs = (full_token_flags_sorted == 0).nonzero().squeeze(dim=1)
                return ready_candidate_idxs, pending_candidate_idxs

            numericalized_subtokens_list, scores = beam_search(self._model, self._last_predicted_token_tensor[0], complete_token_predicate, n_suggestions, self.BEAM_SIZE)
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

    def get_model_summary(self) -> ModelSummary:
        return ModelSummary(id=self._id,
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
        return str(self.get_model_summary())
