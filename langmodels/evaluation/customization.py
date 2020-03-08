import logging
from functools import reduce
from typing import List, Callable, Any, Type, Union, Set, Iterable, FrozenSet, Tuple

from dataclasses import dataclass, field

from codeprep.preprocess.metadata import PreppedTokenMetadata
from codeprep.tokens import PreppedSubTokenSequence, PreppedFullTokenSequence
from codeprep.tokentypes.containers import Comment, SplitContainer, OneLineComment
from codeprep.tokentypes.rootclasses import ParsedToken

logger = logging.getLogger(__name__)


def all_subclasses(classes: Iterable[Type]) -> Set[Type]:
    """
    >>> subclasses = all_subclasses([ParsedToken])
    >>> sorted(map(lambda t: t.__name__, subclasses))
    ['ClosingBracket', 'ClosingCurlyBracket', 'Comment', 'KeyWord', 'MultilineComment', 'NewLine', 'NonCodeChar', \
'NonEng', 'NonProcessibleToken', 'Number', 'One', 'OneLineComment', 'OpeningBracket', 'OpeningCurlyBracket', \
'Operator', 'ParsedToken', 'ProcessableTokenContainer', 'Semicolon', 'SpaceInString', 'SpecialToken', \
'SplitContainer', 'StringLiteral', 'Tab', 'TextContainer', 'Whitespace', 'Zero']
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


class FilteringTokenIterator(object):
    """
    >>> metadata = PreppedTokenMetadata(n_subtokens_per_token=[1, 2], token_types=[SplitContainer, OneLineComment])

    >>> token_type_subset = TokenTypeSubset.full_set_without_comments()
    >>> [token for token in FilteringTokenIterator(PreppedSubTokenSequence(['hi', '/', '/'], metadata), token_type_subset)]
    ['hi']

    >>> [token for token in FilteringTokenIterator(PreppedFullTokenSequence(['hi', '/', '/'], metadata))]
    ['hi', '//']

    >>> [token for token in FilteringTokenIterator(PreppedSubTokenSequence(['hi', '/', '/'], metadata), return_token_type=True)]
    [('hi', <class 'codeprep.tokentypes.containers.SplitContainer'>), ('/', <class 'codeprep.tokentypes.containers.OneLineComment'>), ('/', <class 'codeprep.tokentypes.containers.OneLineComment'>)]

    >>> it = FilteringTokenIterator(PreppedSubTokenSequence(['hi', '/', '/'], metadata))
    >>> [token for token in it]
    ['hi', '/', '/']

    >>> it = FilteringTokenIterator(PreppedSubTokenSequence(['hi', '/', '/'], metadata), token_type_subset=TokenTypeSubset.only_comments())
    >>> [token for token in it]
    ['/', '/']

    """
    def __init__(self, prepped_tokens: Union[PreppedSubTokenSequence, PreppedFullTokenSequence],
                 token_type_subset: TokenTypeSubset = TokenTypeSubset.full_set(),
                 format: Callable[[List[str]], Any] = lambda s: ''.join(s),
                 return_token_type: bool = False):
        self.token_type_subset = token_type_subset
        self.format = format
        self.current_full_token = 0
        self.return_token_type = return_token_type

        if isinstance(prepped_tokens, PreppedSubTokenSequence):
            self.iterator = iter(zip(prepped_tokens.sub_token_view(), prepped_tokens.get_iterator(over=prepped_tokens.metadata.token_types, over_full_tokens=True)))
        else:
            self.iterator = iter(prepped_tokens.full_tokens_view(return_token_type=True, formatter=format))

    def __iter__(self):
        return self

    def __next__(self) -> Union[Any, Tuple[Any, Type]]:
        while True:
            token, token_type = next(self.iterator)
            if self.token_type_subset.contains(token_type):
                return (token, token_type) if self.return_token_type else token


def each_token_type_separately() -> Set[TokenTypeSubset]:
    all_parsed_token_subclasses = all_subclasses([ParsedToken])
    return {TokenTypeSubset.Builder().add(clazz).build() for clazz in all_parsed_token_subclasses}
