import logging
import os
from dataclasses import dataclass

from dataprep.parse.model.metadata import PreprocessingMetadata
from math import exp
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
from dataprep.api.corpus import PreprocessedCorpus
from dataprep.parse.model.placeholders import placeholders
from fastai.text import AWD_LSTM, SequentialRNN, get_language_model, F, Vocab, awd_lstm_lm_config, convert_weights
from fastai.text.models.transformer import init_transformer
from torch import cuda

from langmodels.beamsearch import beam_search
from langmodels.lmconfig.datamodel import Corpus, LstmArch, TransformerArch, LMTrainingConfig, GruArch
from langmodels.lmconfig.serialization import load_config_from_file, read_value_from_file
from langmodels.migration import pth_to_torch
from langmodels.nn import to_test_mode, get_last_layer_activations, TORCH_LONG_MIN_VAL, to_binary_entropy
from langmodels.gpuutil import get_device

logger = logging.getLogger(__name__)


PAD_TOKEN_INDEX = 1


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
    d = arch.__dict__
    d['init'] = init_transformer
    return d


def create_custom_config(lm_training_config: LMTrainingConfig):
    arch = lm_training_config.arch
    if isinstance(arch, LstmArch) or isinstance(arch, GruArch):
        return create_custom_lstm_or_gru_config(arch)
    elif isinstance(arch, TransformerArch):
        return create_custom_transformer_config(arch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_terminal_placeholders() -> List[str]:
    return [placeholders['comment'], placeholders['string_literal'],
            placeholders['ect'], placeholders['non_eng'], placeholders['olc_end']]


def is_terminal_subtoken(subtoken: str) -> bool:
    return subtoken.endswith(placeholders['compound_word_end']) or subtoken in get_terminal_placeholders()


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


DEFAULT_BEAM_SIZE = 500


def check_metadata_validity(prep_text: List[str], metadata: PreprocessingMetadata):
    if 0 not in metadata.word_boundaries:
        raise AssertionError('')

    for idx, token in enumerate(prep_text):
        end_according_to_data = is_terminal_subtoken(token)
        end_according_to_metadata = (idx + 1) in metadata.word_boundaries
        if end_according_to_data != end_according_to_metadata:
            error_context_start_index = idx - 20 if idx - 20 > 0 else 0
            raise AssertionError(f'Token {token} according to metadata is'
                                 f'{" " if end_according_to_metadata else " NOT"} end-token. '
                                 f'Showing context: {prep_text[error_context_start_index:idx+1]}')


@dataclass
class ModelDescription(object):
    name: str
    bpe_merges: str
    layers_config: str
    arch: str
    bin_entropy: float
    training_time_minutes_per_epoch: int
    n_epochs: int
    best_epoch: int

    def __str__(self):
        return f'{self.name}\t{self.bpe_merges}\t{self.layers_config}\t{self.arch}' \
               f'\t{self.bin_entropy}\t{self.training_time_minutes_per_epoch}\t{self.n_epochs}(best: {self.best_epoch})'


class TrainedModel(object):
    STARTING_TOKEN = placeholders['ect']

    def __init__(self, path: str, force_use_cpu: bool = False):
        self.force_use_cpu = force_use_cpu
        self.model_name = os.path.basename(path)
        self.original_vocab = Vocab.load(os.path.join(path, 'vocab'))
        term_vocab, self.first_nonterm_token = _create_term_vocab(self.original_vocab)
        self.model, self.vocab = self._load_model(path, term_vocab)
        to_test_mode(self.model)

        # last_predicted_token_tensor is a rank-2 tensor!
        self.last_predicted_token_tensor = torch.tensor([self.vocab.numericalize([self.STARTING_TOKEN])],
                                                        device=get_device(self.force_use_cpu))
        self.beam_size = DEFAULT_BEAM_SIZE

    def _load_model(self, path: str, custom_vocab: Optional[Vocab] = None) -> Tuple[SequentialRNN, Vocab]:
        path_to_model = os.path.join(path, 'best.pth')
        logger.debug(f"Loading model from: {path_to_model} ...")

        self.config = load_config_from_file(os.path.join(path, 'config'))
        self._prep_function = self.config.prep_function
        # self.bin_entropy = read_value_from_file(os.path.join(path, 'binentropy'), value_type=float)
        # self.training_time_minutes_per_epoch = read_value_from_file(os.path.join(path, 'ttime'), value_type=int)
        # self.n_epochs = read_value_from_file(os.path.join(path, 'epochs'), value_type=int)
        # self.best_epoch = read_value_from_file(os.path.join(path, 'epoch.best'), value_type=int)

        vocab = custom_vocab if custom_vocab else self.original_vocab
        model = get_language_model(self.config.get_arch_class(), len(vocab.itos), create_custom_config(self.config))
        if self.force_use_cpu:
            map_location = lambda storage, loc: storage
            logger.debug("Using CPU for inference")
        elif cuda.is_available():
            map_location = torch.device('cuda:0')
            model.cuda()
            logger.debug("Using GPU for inference")
        else:
            map_location = lambda storage, loc: storage
            logger.info("Cuda not available. Falling back to using CPU.")

        state_dict = torch.load(path_to_model, map_location=map_location)

        # a more simple solution is to use fastai's load_learner,
        # however it doesn't seem to work out of the box with customizations we've done
        weights = pth_to_torch(state_dict)
        if custom_vocab:
            weights = convert_weights(weights, self.original_vocab.stoi, custom_vocab.itos)
        model.load_state_dict(weights, strict=True)

        return model, vocab

    def prep_corpus(self, corpus: Corpus, *kwargs) -> PreprocessedCorpus:
        return self._prep_function.apply(corpus, kwargs)

    def prep_text(self, text, **kwargs) -> Tuple[List[str], PreprocessingMetadata]:
        import dataprep.api.text as text_api
        text_callable = getattr(text_api, self._prep_function.callable.__name__)
        prep_text, metadata = text_callable(text, *self._prep_function.params, **self._prep_function.options, **kwargs)
        check_metadata_validity(prep_text, metadata)
        return prep_text, metadata

    def feed_text(self, text: str) -> None:
        prep_text, metadata = self.prep_text(text, return_metadata=True, force_reinit_bpe_data=False)
        context_tensor = torch.tensor([self.vocab.numericalize(prep_text)], device=get_device(self.force_use_cpu))
        _ = get_last_layer_activations(self.model, context_tensor[:,:-1])
        self.last_predicted_token_tensor = context_tensor[:,-1:]

    def get_entropies_for_prep_text(self, prep_text: List[str]) -> List[float]:
        """
        changes hidden states of the model!!
        """
        if not prep_text:
            return []

        loss_list = []
        max_subtokens_per_chunk = 1000
        # if the line is too big, we break it down to chunks to fit it into gpu memory
        # big chunks require more memory, small chunks require more time
        n_chunks = (len(prep_text)-1) // max_subtokens_per_chunk + 1
        for i in range(n_chunks):
            numericalized_prep_text = torch.tensor([self.vocab.numericalize(prep_text[i*max_subtokens_per_chunk:(i+1)*max_subtokens_per_chunk])],
                                                   device=get_device(self.force_use_cpu))

            last_layer = get_last_layer_activations(self.model, torch.cat([self.last_predicted_token_tensor, numericalized_prep_text[:, :-1]], dim=1))
            loss = F.cross_entropy(last_layer.view(-1, last_layer.shape[-1]),
                                   numericalized_prep_text.view(-1),
                                   reduction='none')
            binary_loss = to_binary_entropy(loss)
            loss_list.extend(binary_loss.tolist())
            self.last_predicted_token_tensor = numericalized_prep_text[:, -1:]
        return loss_list

    def reset(self) -> None:
        self.model.reset()
        self.last_predicted_token_tensor = torch.tensor([self.vocab.numericalize([self.STARTING_TOKEN])],
                                                        device=get_device(self.force_use_cpu))

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
        if textified in placeholders.values():
            return textified

        if not textified.endswith(cwe):
            raise ValueError(f'{textified} ({subtokens_num}) is not a full token')
        return textified if include_debug_tokens else textified[:-len(cwe)]

    def _format_layers_config(self) -> str:
        n_layers = self.config.arch.n_layers
        emb_size = self.config.arch.emb_sz
        n_hid = self.config.arch.n_hid
        return f'{emb_size}/{n_layers}/{n_hid}'

    def get_model_description(self) -> ModelDescription:
        return ModelDescription(name=self.model_name,
                                bpe_merges=self.config.prep_function.params[0],
                                layers_config=self._format_layers_config(),
                                arch=str(self.config.get_arch_class()),
                                bin_entropy=self.bin_entropy,
                                training_time_minutes_per_epoch=self.training_time_minutes_per_epoch,
                                n_epochs=self.n_epochs,
                                best_epoch=self.best_epoch)

