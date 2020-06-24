from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, List, Union

from fastai.text import Vocab
from math import log

from codeprep.preprocess.codestructure import CodeLocation
from codeprep.preprocess.tokens import TokenSequence
from langmodels.model.context import ContextInformation


class Characteristic(ABC):
    @abstractmethod
    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        pass


class TokenType(Characteristic):
    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        return token.metadata.token_type().__name__


class SubtokenNumber(Characteristic):
    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        return token.metadata.n_subtokens()


class FrequencyRank(Characteristic):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        token_str = token.to_full_token_string(keep_word_end_token=False) if token.is_complete() else token.token_str()
        return int(log(self.vocab.stoi[token_str] + 1, 2))


class Project(Characteristic):
    def __init__(self, relative_to: Union[Path, str]):
        self.relative_to = relative_to if isinstance(relative_to, Path) else Path(relative_to)

    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        return str(code_location.path.relative_to(self.relative_to).parts[0])


class LinePosition(Characteristic):
    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        return code_location.line


class TokenPosition(Characteristic):
    def __call__(self, token: TokenSequence, code_location: CodeLocation, context_information: Optional[ContextInformation] = None) -> Any:
        return code_location.token


def characterize_token(token: TokenSequence, characteristics: List[Characteristic], code_location: CodeLocation,
                       context_information: Optional[ContextInformation] = None) -> List[Any]:
    return [f(token, code_location, context_information) for f in characteristics]
