import logging
import os
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Tuple, Generator

from dataprep.parse.model.metadata import PreprocessingMetadata
from dataprep.parse.model.placeholders import placeholders
from langmodels.model import TrainedModel


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult(object):
    text: str
    prep_text: List[str]
    prep_metadata: PreprocessingMetadata
    results: Dict[str, List[float]]
    aggregated_result: Dict[str, float]


LMEvaluator = Callable[[TrainedModel, List[str], PreprocessingMetadata], Tuple[List, float]]


def get_file_total_lines(file):
    with open(file, 'r') as f:
        return len([i for i in f])


def get_file_extension(file: str) -> str:
    """
    >>> get_file_extension('new_file.java')
    'java'
    """
    return os.path.splitext(file)[1][1:]

# TODO this class should be included in dataprep lib
class FullWordIterator:
    """
    >>> [token for token in FullWordIterator(['hi', 'the', 're'], [0, 1, 3])]
    ['hi', 'there']

    >>> [token for token in FullWordIterator(['hel', 'l', 'o'], [0, 3])]
    ['hello']

    >>> [token for token in FullWordIterator([1, 2, 4], [0, 2, 3], agg=sum)]
    [3, 4]

    >>> [token for token in FullWordIterator([], [])]
    Traceback (most recent call last):
    ...
    ValueError: Word boundaries list should contain at least 0!

    >>> [token for token in FullWordIterator(['hi'], [0])]
    Traceback (most recent call last):
    ...
    ValueError: Word boundaries list should contain the indices of the last word.
    However, the subword entropies list has 1 elements, and value 1 is not found in word boundaries list: [0]

    >>> [token for token in FullWordIterator(['hi'], [1])]
    Traceback (most recent call last):
    ...
    ValueError: Word boundaries list must start with 0!
    """
    def __init__(self, subwords: List[Any], word_boundaries: List[int], agg=lambda s: ''.join(s)):
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

    def __iter__(self):
        self.current_full_word = 0
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
    if len(subtokens) == 1 and subtokens[0] in placeholders.values():
        return subtokens[0]

    sep = '|' if include_debug_tokens else ''
    joined = sep.join(subtokens)
    cwe = placeholders['compound_word_end']
    if not joined.endswith(cwe):
        raise ValueError(f'{joined} ({subtokens}) is not a full token')
    return joined if include_debug_tokens else joined[:-len(cwe)]


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
