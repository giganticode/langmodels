from abc import ABC, abstractmethod
from typing import Optional, Any, List

from fastai.text import Vocab
from math import log

from codeprep.preprocess.tokens import TokenSequence
from langmodels.model.context import ContextInformation
from langmodels.model.model import to_full_token_string


class Characteristic(ABC):
    @abstractmethod
    def __call__(self, token: TokenSequence, context_information: Optional[ContextInformation] = None) -> Any:
        pass


class TokenType(Characteristic):
    def __call__(self, token: TokenSequence, context_information: Optional[ContextInformation] = None) -> Any:
        return token.metadata.token_type().__name__


class SubtokenNumber(Characteristic):
    def __call__(self, token: TokenSequence, context_information: Optional[ContextInformation] = None) -> Any:
        return token.metadata.n_subtokens()


class FrequencyRank(Characteristic):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def __call__(self, token: TokenSequence, context_information: Optional[ContextInformation] = None) -> Any:
        token_str = to_full_token_string(token.tokens, keep_word_end_token=False) if token.is_complete() else token.tokens[0]
        return int(log(self.vocab.stoi[token_str] + 1, 2))


def characterize_token(token: TokenSequence, characteristics: List[Characteristic],
                       context_information: Optional[ContextInformation] = None) -> List[Any]:
    return [f(token, context_information) for f in characteristics]
