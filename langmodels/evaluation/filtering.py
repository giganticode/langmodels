import logging
from functools import reduce
from typing import List, Callable, Any, Type, Union, Set, Iterable, FrozenSet, Tuple, Optional, Dict

from dataclasses import dataclass, field

from dataprep.preprocess.metadata import PreprocessingMetadata
from dataprep.preprocess.placeholders import placeholders
from dataprep.subtokens import FullTokenIterator, is_terminal_subtoken, SubtokenIterator
from dataprep.tokens.containers import Comment, SplitContainer, OneLineComment
from dataprep.tokens.rootclasses import ParsedToken

logger = logging.getLogger(__name__)


def all_subclasses(classes: Iterable[Type]) -> Set[Type]:
    """
    >>> subclasses = all_subclasses([ParsedToken])
    >>> sorted(map(lambda t: t.__name__, subclasses))
    ['Comment', 'KeyWord', 'MultilineComment', 'NewLine', 'NonCodeChar', 'NonEng', 'NonProcessibleToken', \
'Number', 'OneLineComment', 'Operator', 'ParsedToken', 'ProcessableTokenContainer', 'SpaceInString', 'SpecialToken', \
'SplitContainer', 'StringLiteral', 'Tab', 'TextContainer', 'Whitespace']
    """
    return reduce(set.union, [{cls}.union(
        [s for c in cls.__subclasses__() for s in all_subclasses([c])])
        for cls in classes], set())


@dataclass(eq=True, frozen=True)
class TokenTypeSubset(object):
    """
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(object): pass

    >>> token_types = TokenTypeSubset.Builder().add({A, D}).remove({B, D}).build()
    >>> token_types.summary()
    "A,D[except:B,D] -> ['A', 'C']"
    >>> token_types.contains(A)
    True
    >>> token_types.contains(B)
    False
    >>> token_types.contains(C)
    True
    >>> token_types.contains(D)
    False


    """
    all_included_types: FrozenSet[Type]
    short_summary: str

    @dataclass()
    class Builder(object):
        included: Set[Type] = field(default_factory=set)
        excluded: Set[Type] = field(default_factory=set)

        def _short_summary(self) -> str:
            res = ','.join(sorted(map(lambda t: t.__name__, self.included)))
            if self.excluded:
                res += f'[except:{",".join(sorted(map(lambda x: x.__name__, self.excluded)))}]'
            return res

        def add(self, types: Union[Type, Iterable[Type]]) -> 'TokenTypeSubset.Builder':
            self.included = self.included.union(set(types) if isinstance(types, Iterable) else {types})
            return self

        def remove(self, types: Union[Type, Iterable[Type]]) -> 'TokenTypeSubset.Builder':
            self.excluded = self.excluded.union(set(types) if isinstance(types, Iterable) else {types})
            return self

        def build(self) -> 'TokenTypeSubset':
            all_included = all_subclasses(self.included)
            all_excluded = all_subclasses(self.excluded)
            return TokenTypeSubset(all_included_types=frozenset(all_included.difference(all_excluded)),
                                   short_summary=self._short_summary())

    def summary(self) -> str:
        return f'{self.short_summary} -> {sorted(map(lambda x: x.__name__, self.all_included_types))}'

    def __str__(self) -> str:
        return self.short_summary

    def __repr__(self):
        return str(self)
    
    @classmethod
    def full_set(cls) -> 'TokenTypeSubset':
        return cls.Builder().add(ParsedToken).build()

    @classmethod
    def full_set_without_comments(cls) -> 'TokenTypeSubset':
        return cls.Builder().add(ParsedToken).remove(Comment).build()

    @classmethod
    def only_comments(cls) -> 'TokenTypeSubset':
        return cls.Builder().add(Comment).build()

    def contains(self, type: Type) -> bool:
        return type in self.all_included_types


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
    >>> metadata = PreprocessingMetadata(word_boundaries=[0, 1, 3], token_types=[SplitContainer, OneLineComment])

    >>> token_type_subset = TokenTypeSubset.full_set_without_comments()
    >>> [token for token in FilteringTokenIterator(['hi', '/', '/'], metadata, token_type_subset)]
    ['hi']

    >>> [token for token in FilteringTokenIterator(['hi', '/', '/'], metadata)]
    ['hi', '//']

    >>> [token for token in FilteringTokenIterator(['hi', '/', '/'], metadata, return_token_type=True)]
    [('hi', <class 'dataprep.tokens.containers.SplitContainer'>), ('//', <class 'dataprep.tokens.containers.OneLineComment'>)]

    >>> it = FilteringTokenIterator(['hi', '/', '/'], metadata, token_iterator_type=SubtokenIterator)
    >>> [token for token in it]
    ['hi', '/', '/']

    >>> it = FilteringTokenIterator(['hi', '/', '/'], metadata, token_type_subset=TokenTypeSubset.only_comments(), \
token_iterator_type=SubtokenIterator)
    >>> [token for token in it]
    ['/', '/']
    """
    def __init__(self, subwords: List[Any], metadata: PreprocessingMetadata,
                 token_type_subset: TokenTypeSubset = TokenTypeSubset.full_set(),
                 token_iterator_type: Type[TokenIterator] = FullTokenIterator,
                 format: Callable[[List[str]], Any] = lambda s: ''.join(s),
                 return_token_type: bool = False):
        self.full_token_iterator = token_iterator_type(subwords, metadata.word_boundaries,
                                                       format=lambda l:l, return_full_token_index=True)
        self.token_type_subset = token_type_subset
        self.metadata = metadata
        self.format = format
        self.current_full_token = 0
        self.return_token_type = return_token_type

    def __iter__(self):
        return self

    def __next__(self) -> Union[Any, Tuple[Any, Type]]:
        while True:
            current_full_token_ind, next_full_token = next(self.full_token_iterator)
            token_type = self.metadata.token_types[current_full_token_ind]
            if self.token_type_subset.contains(token_type):
                formatted_value = self.format(next_full_token)
                return (formatted_value, token_type) if self.return_token_type else formatted_value


@dataclass(frozen=True)
class TokenTypeWeights(object):
    non_default_weights: Optional[Dict[Type, float]] = None

    def __getitem__(self, item: Type) -> float:
        return self.non_default_weights[item] if self.non_default_weights is not None else 1.

    def __str__(self):
        return f'{self.non_default_weights}' if self.non_default_weights is not None else 'default_weights'

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class EvaluationCustomization(object):
    type_subset: TokenTypeSubset = TokenTypeSubset.full_set()
    weights: TokenTypeWeights = TokenTypeWeights()

    def __str__(self):
        return f'{self.type_subset}/{self.weights}'

    def __repr__(self):
        return str(self)

    @classmethod
    def no_customization(cls):
        return EvaluationCustomization()
