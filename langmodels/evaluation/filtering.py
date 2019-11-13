import logging
from enum import Enum

from typing import List, Callable, Any, Type, Union

from dataprep.parse.model.metadata import PreprocessingMetadata
from dataprep.parse.model.placeholders import placeholders
from dataprep.subtokens import FullTokenIterator, is_terminal_subtoken, SubtokenIterator

logger = logging.getLogger(__name__)

MetricName = str


class TokenTypes(str, Enum):
    ALL = 'all'
    ONLY_COMMENTS = 'only_comments'
    ALL_BUT_COMMENTS = 'all_but_comments'

    def __str__(self) -> str:
        return self.value


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