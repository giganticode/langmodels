import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, asdict

from dataprep.subtokens import is_terminal_subtoken

from dataprep.infrastructure.dataset import normalize_extension_string

from dataprep.parse.model.metadata import PreprocessingMetadata, check_metadata_validity
from math import exp
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
from dataprep.api.corpus import PreprocessedCorpus
from dataprep.parse.model.placeholders import placeholders
from fastai.text import SequentialRNN, get_language_model, F, Vocab, awd_lstm_lm_config, convert_weights
from fastai.text.models.transformer import init_transformer
from torch import cuda

from langmodels.beamsearch import beam_search
from langmodels.lmconfig.datamodel import Corpus, LstmArch, TransformerArch, LMTrainingConfig, GruArch, \
    LMTrainingMetrics
from langmodels.lmconfig.serialization import load_config_from_file, read_value_from_file
from langmodels.migration import pth_to_torch
from langmodels.nn import to_test_mode, get_last_layer_activations, TORCH_LONG_MIN_VAL
from langmodels.util import to_binary_entropy
from langmodels.cuda_util import get_device, get_map_location

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


DEFAULT_BEAM_SIZE = 500


def to_full_token_string(vocab: Vocab, subtokens_num: torch.LongTensor, include_debug_tokens=False) -> str:
    try:
        subtokens_num = subtokens_num[:subtokens_num.tolist().index(TORCH_LONG_MIN_VAL)]
    except ValueError: pass

    sep = '|' if include_debug_tokens else ''
    textified = vocab.textify(subtokens_num, sep=sep)
    cwe = placeholders['compound_word_end']
    if textified in placeholders.values():
        return textified

    if not is_terminal_subtoken(textified):
        raise ValueError(f'{textified} ({subtokens_num}) is not a full token')
    return textified if include_debug_tokens else textified[:-len(cwe)]


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
    tags: List[str]

    def is_tagged_by(self, tag: str) -> bool:
        return tag in self.tags

    @staticmethod
    def get_attribute_list() -> List[str]:
        return [k for k in ModelDescription.__annotations__.keys()]

    def get_value_list(self) -> List[str]:
        return list(map(lambda a: self.__getattribute__(a), ModelDescription.get_attribute_list()))


class TrainedModel(object):
    STARTING_TOKEN = placeholders['ect']

    def __init__(self, path: str, force_use_cpu: bool = False, load_only_description: bool = False):
        self.force_use_cpu = force_use_cpu
        self.id = os.path.basename(path)
        path_to_config_file = os.path.join(path, 'config')
        path_to_metrics_file = os.path.join(path, 'metrics')
        path_to_tags_file = os.path.join(path, 'tags')
        self.metrics = None
        self.config = None
        self.tags = []
        try:
            self.config: LMTrainingConfig = load_config_from_file(path_to_config_file)
        except FileNotFoundError:
            logger.warning(f'Config file not found: {path_to_config_file}')
        try:
            self.metrics: LMTrainingMetrics = load_config_from_file(os.path.join(path, 'metrics'))
        except FileNotFoundError:
            logger.warning(f'File with metrics not found: {path_to_metrics_file}')
        if os.path.exists(path_to_tags_file):
            value = read_value_from_file(path_to_tags_file, value_type=str)
            if value != '':
                self.tags = value.split(',')
        self._prep_function = self.config.prep_function

        self.load_only_description = load_only_description
        if not load_only_description:
            # we might want to load only description without loading actual weights when we want
            # to save time when loading multiple models to choose one of them to work with

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

        vocab = custom_vocab if custom_vocab else self.original_vocab
        model = get_language_model(self.config.get_arch_class(), len(vocab.itos), create_custom_config(self.config))
        map_location = get_map_location(self.force_use_cpu)
        if cuda.is_available():
            model.cuda()
        state_dict = torch.load(path_to_model, map_location=map_location)

        # a more simple solution is to use fastai's load_learner,
        # however it doesn't seem to work out of the box with customizations we've done

        # update: it appeared that it's quite simple to load weights. So maybe not worth it loading a learner?
        weights = OrderedDict(state_dict['model'] if ('model' in state_dict) else state_dict)
        if custom_vocab:
            weights = convert_weights(weights, self.original_vocab.stoi, custom_vocab.itos)
        model.load_state_dict(weights, strict=True)

        return model, vocab

    def prep_corpus(self, corpus: Corpus, **kwargs) -> PreprocessedCorpus:
        return self._prep_function.apply(corpus, **kwargs)

    def prep_text(self, text, **kwargs) -> Union[Tuple[List[str], PreprocessingMetadata], List[str]]:
        import dataprep.api.text as text_api
        text_callable = getattr(text_api, self._prep_function.callable.__name__)
        prep_text, metadata = text_callable(text, *self._prep_function.params, **asdict(self._prep_function.options), **kwargs)
        check_metadata_validity(prep_text, metadata)
        return (prep_text, metadata) if 'return_metadata' in kwargs and kwargs['return_metadata'] else prep_text

    def _check_model_loaded(self, only_description=False):
        if not only_description and self.load_only_description:
            raise RuntimeError("Operation not supported. Only model's description is loaded. "
                               "Prease reload the model with param load_only_description set to False.")

    def feed_text(self, text: str) -> None:
        self._check_model_loaded()

        prep_text, metadata = self.prep_text(text, return_metadata=True, force_reinit_bpe_data=False)
        context_tensor = torch.tensor([self.vocab.numericalize(prep_text)], device=get_device(self.force_use_cpu))
        _ = get_last_layer_activations(self.model, context_tensor[:,:-1])
        self.last_predicted_token_tensor = context_tensor[:,-1:]

    def get_entropies_for_prep_text(self, prep_text: List[str]) -> List[float]:
        """
        changes hidden states of the model!!
        """
        self._check_model_loaded()

        if not prep_text:
            return []

        loss_list = []
        max_subtokens_per_chunk = 1000
        # if the line is too big, we break it down to chunks to fit it into gpu memory
        # big chunks require more memory, small chunks require more time
        n_chunks = (len(prep_text)-1) // max_subtokens_per_chunk + 1
        for i in range(n_chunks):
            pt = prep_text[i*max_subtokens_per_chunk:(i+1)*max_subtokens_per_chunk]
            numericalized_prep_text = torch.tensor([self.vocab.numericalize(pt)],
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
        self._check_model_loaded()

        self.model.reset()
        self.last_predicted_token_tensor = torch.tensor([self.vocab.numericalize([self.STARTING_TOKEN])],
                                                        device=get_device(self.force_use_cpu))

    def predict_next_full_token(self, n_suggestions: int, include_debug_tokens: bool = False) -> List[Tuple[str, float]]:
        self._check_model_loaded()

        subtokens, scores = beam_search(self.model, self.last_predicted_token_tensor[0], self.first_nonterm_token, n_suggestions, self.beam_size)
        return [(to_full_token_string(self.vocab, st, include_debug_tokens=include_debug_tokens), 1 / exp(score.item())) for st, score in zip(subtokens, scores)]

    def _format_layers_config(self) -> str:
        if isinstance(self.config.arch, TransformerArch):
            return "transformer"  # TODO add proper layer description
        n_layers = self.config.arch.n_layers
        emb_size = self.config.arch.emb_sz
        n_hid = self.config.arch.n_hid
        return f'{emb_size}/{n_layers}/{n_hid}'

    def get_model_description(self) -> ModelDescription:
        return ModelDescription(id=self.id,
                                bpe_merges=self.config.prep_function.params[0],
                                layers_config=self._format_layers_config(),
                                arch=str(self.config.get_arch_class().__name__),
                                bin_entropy=self.metrics.bin_entropy if self.metrics else 1e+6,
                                training_time_minutes_per_epoch=self.metrics.training_time_minutes_per_epoch
                                if self.metrics else 0,
                                n_epochs=self.metrics.n_epochs if self.metrics else 0,
                                best_epoch=self.metrics.best_epoch if self.metrics else -1,
                                tags=self.tags)

    def check_inference_possible_for_file_type(self, extension):
        if extension not in normalize_extension_string(self.config.corpus.extensions):
            raise ValueError(f'The model was not trained on .{extension} files. Cannot do inference.')

    def __str__(self) -> str:
        return str(self.get_model_description())
