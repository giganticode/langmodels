import logging
import os
from dataclasses import dataclass
from enum import Enum

from typing import List, Callable, Dict, Any, Tuple, Generator, Set, Optional, Type, Union

from dataprep.parse.model.metadata import PreprocessingMetadata
from dataprep.parse.model.placeholders import placeholders
from langmodels.model import is_terminal_subtoken

logger = logging.getLogger(__name__)

MetricName = str


class TokenTypes(str, Enum):
    ALL = 'all'
    ONLY_COMMENTS = 'only_comments'
    ALL_BUT_COMMENTS = 'all_but_comments'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EvaluationResult(object):
    subtoken_values: List[Optional[float]]  # this value should correspond to the number of subtokens
    average: float
    n_samples: int


@dataclass(frozen=True)
class EvaluationScenario(object):
    metric_name: MetricName
    token_types: TokenTypes

    def __str__(self):
        return f'{self.metric_name}/{self.token_types}'

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Evaluation(object):
    text: str
    prep_text: List[str]
    prep_metadata: PreprocessingMetadata
    scenarios: Dict[EvaluationScenario, EvaluationResult]


def get_file_total_lines(file):
    with open(file, 'r') as f:
        return len([i for i in f])


def get_file_extension(file: str) -> str:
    """
    >>> get_file_extension('new_file.java')
    'java'
    """
    return os.path.splitext(file)[1][1:]


class SubtokenIterator(object):
    def __init__(self, subwords: List[Any], word_boundaries: List[int], agg: Callable[[List[str]], Any]):
        self.it = iter(subwords)

    def __iter__(self):
        return self

    def __next__(self):
        return [next(self.it)], 1


# TODO this class should be included in dataprep lib
class FullTokenIterator(object):
    """
    >>> [token for token in FullTokenIterator(['hi', 'the', 're'], [0, 1, 3])]
    ['hi', 'there']

    >>> [token for token in FullTokenIterator(['hel', 'l', 'o'], [0, 3])]
    ['hello']

    >>> [token for token in FullTokenIterator([1, 2, 4], [0, 2, 3], agg=sum)]
    [3, 4]

    >>> [token for token in FullTokenIterator([], [])]
    Traceback (most recent call last):
    ...
    ValueError: Word boundaries list should contain at least 0!

    >>> [token for token in FullTokenIterator(['hi'], [0])]
    Traceback (most recent call last):
    ...
    ValueError: Word boundaries list should contain the indices of the last word.
    However, the subword entropies list has 1 elements, and value 1 is not found in word boundaries list: [0]

    >>> [token for token in FullTokenIterator(['hi'], [1])]
    Traceback (most recent call last):
    ...
    ValueError: Word boundaries list must start with 0!
    """
    def __init__(self, subwords: List[Any],
                 word_boundaries: List[int],
                 agg: Callable[[List[str]], Any] = lambda s: ''.join(s)):
        if len(word_boundaries) == 0:
            raise ValueError("Word boundaries list should contain at least 0!")
        if len(subwords) != word_boundaries[-1]:
            raise ValueError(f"Word boundaries list should contain the indices of the last word.\n"
                             f"However, the subword entropies list has {len(subwords)} elements, and "
                             f"value {len(subwords)} is not found in word boundaries list: {word_boundaries}")
        if word_boundaries[0] != 0:
            raise ValueError('Word boundaries list must start with 0!')

        self.subwords = subwords
        self.word_boundaries = word_boundaries
        self.agg = agg
        self.current_full_word = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_full_word >= len(self.word_boundaries) - 1:
            raise StopIteration
        word_start = self.word_boundaries[self.current_full_word]
        word_end = self.word_boundaries[self.current_full_word + 1]
        self.current_full_word += 1
        return self.agg(self.subwords[word_start:word_end])


def to_full_token_string(subtokens: List[str], include_debug_tokens: bool = False):
    """
    >>> to_full_token_string(['the', 're</t>'], include_debug_tokens=True)
    'the|re</t>'

    >>> to_full_token_string(['re', 'vol', 'v', 'er</t>'])
    'revolver'

    >>> to_full_token_string([placeholders['olc_end']])
    '<EOL>'
    """
    if not isinstance(subtokens, List):
        raise TypeError(f'This methods accept a list of strings, however {type(subtokens)} was passed')

    if len(subtokens) == 1 and subtokens[0] in placeholders.values():
        return subtokens[0]

    sep = '|' if include_debug_tokens else ''
    joined = sep.join(subtokens)
    cwe = placeholders['compound_word_end']
    if not is_terminal_subtoken(joined):
        raise ValueError(f'{joined} ({subtokens}) is not a full token')
    return joined if include_debug_tokens else joined[:-len(cwe)]


TokenIterator = Union[SubtokenIterator, FullTokenIterator]


class FilteringTokenIterator(object):
    """
    >>> from langmodels.evaluation.metrics import inclusion_func_dict
    >>> metadata = PreprocessingMetadata(word_boundaries=[0, 1, 3], comments=[(1,3)])
    >>> [token for token in FilteringTokenIterator(['hi', '/', '/'], metadata, inclusion_function=inclusion_func_dict[TokenTypes.ALL_BUT_COMMENTS])]
    ['hi']

    >>> from langmodels.evaluation.metrics import inclusion_func_dict
    >>> metadata = PreprocessingMetadata(word_boundaries=[0, 1, 3], comments=[(1,3)])
    >>> [token for token in FilteringTokenIterator(['hi', '/', '/'], metadata)]
    ['hi', '//']

    >>> from langmodels.evaluation.metrics import inclusion_func_dict
    >>> metadata = PreprocessingMetadata(word_boundaries=[0, 1, 3], comments=[(1,3)])
    >>> it = FilteringTokenIterator(['hi', '/', '/'], metadata, token_iterator_type=SubtokenIterator, agg=lambda l: l[0])
    >>> [token for token in it]
    ['hi', '/', '/']

    >>> from langmodels.evaluation.metrics import inclusion_func_dict
    >>> metadata = PreprocessingMetadata(word_boundaries=[0, 1, 3], comments=[(1,3)])
    >>> it = FilteringTokenIterator(['hi', '/', '/'], metadata, inclusion_function=inclusion_func_dict[TokenTypes.ONLY_COMMENTS], token_iterator_type=SubtokenIterator, agg=lambda l: l[0])
    >>> [token for token in it]
    ['/', '/']
    """
    def __init__(self, subwords: List[Any], metadata: PreprocessingMetadata,
                 inclusion_function: Callable[[PreprocessingMetadata, int], bool] = lambda m, i: True,
                 token_iterator_type: Type[TokenIterator] = FullTokenIterator,
                 agg: Callable[[List[str]], Any] = lambda s: ''.join(s)):
        self.full_token_iterator = token_iterator_type(subwords, metadata.word_boundaries, agg=lambda s: (s, len(s)))
        self.inclusion_function = inclusion_function
        self.metadata = metadata
        self.agg = agg
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            next_full_token, n_subtokens = next(self.full_token_iterator)
            current_index = self.current_index
            self.current_index += n_subtokens
            if self.inclusion_function(self.metadata, current_index):
                return self.agg(next_full_token)


def read_file_contents(file_path: str) -> str:
    try:
        return read_file_with_encoding(file_path, 'utf-8')
    except UnicodeDecodeError:
        try:
            return read_file_with_encoding(file_path, 'ISO-8859-1')
        except UnicodeDecodeError:
            logger.error(f"Unicode decode error in file: {file_path}")


def read_file_with_encoding(file_path: str, encoding: str) -> str:
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def get_all_files(path: str) -> Generator[str, None, None]:
    if os.path.isfile(path):
        yield path
    else:
        for root, dirs, files in os.walk(path, followlinks=True):
            for file in files:
                if file.endswith('.java'):
                    yield os.path.join(root, file)
