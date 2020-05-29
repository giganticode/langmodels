import logging
from collections import defaultdict
from functools import reduce
from typing import Type, Union, Set, Iterable, FrozenSet, Dict, Mapping, List

from dataclasses import dataclass, field
from frozendict import frozendict

from codeprep.preprocess.metadata import PreppedTokenMetadata
from codeprep.preprocess.tokens import TokenSequence
from codeprep.tokentypes.containers import Comment, Identifier
from codeprep.tokentypes.rootclasses import ParsedToken

logger = logging.getLogger(__name__)


def all_subclasses(classes: Iterable[Type]) -> Set[Type]:
    """
    >>> subclasses = all_subclasses([ParsedToken])
    >>> sorted(map(lambda t: t.__name__, subclasses))
    ['ClosingBracket', 'ClosingCurlyBracket', 'Comment', 'Identifier', 'KeyWord', 'MultilineComment', 'NewLine', 'NonCodeChar', \
'NonEng', 'NonProcessibleToken', 'Number', 'One', 'OneLineComment', 'OpeningBracket', 'OpeningCurlyBracket', \
'Operator', 'ParsedToken', 'ProcessableTokenContainer', 'Semicolon', 'SpaceInString', 'SpecialToken', \
'StringLiteral', 'Tab', 'TextContainer', 'Whitespace', 'Zero']
    """
    return reduce(set.union, [{cls}.union(
        [s for c in cls.__subclasses__() for s in all_subclasses([c])])
        for cls in classes], set())


@dataclass
class TokenCharacteristics(object):
    token_type: Type
    n_subtokens: int

    @classmethod
    def from_metadata(cls, metadata: PreppedTokenMetadata) -> 'TokenCharacteristics':
        return cls(metadata.token_type(), metadata.n_subtokens() if metadata.n_subtokens() < 5 else 5)


def get_token_characteristics(prepped_token_sequence: TokenSequence) -> List[TokenCharacteristics]:
    return [TokenCharacteristics.from_metadata(t.metadata) for t in prepped_token_sequence.with_metadata()]


@dataclass(eq=True, frozen=True)
class TokenCategory(object):
    """
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(object): pass

    >>> token_types1 = TokenCategory.Builder().add({A, D}).remove({B, D}).build()
    >>> token_types1.summary()
    "A,D[except:B,D] -> ['A', 'C']"
    >>> token_types = TokenCategory.Builder().add({A, D}).remove({B, D}).add_with_n_sub_tokens(C, 1).build()
    >>> token_types.summary()
    "A,D[except:B,C,D]+C{1} -> ['A']+C{1}"
    >>> token_types.contains(TokenCharacteristics.from_metadata(PreppedTokenMetadata([1], [A])))
    True
    >>> token_types.contains(TokenCharacteristics.from_metadata(PreppedTokenMetadata([1], [B])))
    False
    >>> token_types.contains(TokenCharacteristics.from_metadata(PreppedTokenMetadata([1], [C])))
    True
    >>> token_types.contains(TokenCharacteristics.from_metadata(PreppedTokenMetadata([2], [C])))
    False
    >>> token_types.contains(TokenCharacteristics.from_metadata(PreppedTokenMetadata([1], [D])))
    False


    """
    all_included_types: FrozenSet[Type]
    short_summary: str
    spec_len_summary: str
    types_of_specific_length: Mapping[Type, FrozenSet[int]]

    @dataclass()
    class Builder(object):
        included: Set[Type] = field(default_factory=set)
        excluded: Set[Type] = field(default_factory=set)
        types_of_specific_length: Dict[Type, Set[int]] = field(default_factory=lambda: defaultdict(set))

        def _short_summary(self) -> str:
            res = ','.join(sorted(map(lambda t: t.__name__, self.included)))
            if self.excluded:
                res += f'[except:{",".join(sorted(map(lambda x: x.__name__, self.excluded)))}]'
            return res

        def _spec_len_summary(self) -> str:
            s = ','.join([f"{k.__name__}{v}" for k, v in self.types_of_specific_length.items()])
            return f'+{s}' if s else ''

        def _change_types(self, current: Set[Type], types: Union[Type, Iterable[Type]]) -> Set[Type]:
            added_types = set(types) if isinstance(types, Iterable) else {types}
            to_return = current.union(added_types)
            for t in added_types:
                if t in self.types_of_specific_length:
                    del self.types_of_specific_length[t]
            return to_return

        def add(self, types: Union[Type, Iterable[Type]]) -> 'TokenCategory.Builder':
            self.included = self._change_types(self.included, types)
            return self

        def add_with_n_sub_tokens(self, t: Type, n_sub_tokens: int) -> 'TokenCategory.Builder':
            self.excluded.add(t)
            self.types_of_specific_length[t].add(n_sub_tokens)
            return self

        def remove(self, types: Union[Type, Iterable[Type]]) -> 'TokenCategory.Builder':
            self.excluded = self._change_types(self.excluded, types)
            return self

        def build(self) -> 'TokenCategory':
            all_included = all_subclasses(self.included)
            all_excluded = all_subclasses(self.excluded)
            spec = {k: frozenset(v) for k, v in self.types_of_specific_length.items()}
            return TokenCategory(all_included_types=frozenset(all_included.difference(all_excluded)),
                                 short_summary=self._short_summary(),
                                 spec_len_summary=self._spec_len_summary(),
                                 types_of_specific_length=frozendict(spec))

    def summary(self) -> str:
        return f'{self.short_summary}{self.spec_len_summary} -> ' \
               f'{sorted(map(lambda x: x.__name__, self.all_included_types))}{self.spec_len_summary}'

    def __str__(self) -> str:
        return f'{self.short_summary}{self.spec_len_summary}'

    def __repr__(self):
        return str(self)
    
    @classmethod
    def full_set(cls) -> 'TokenCategory':
        return cls.Builder().add(ParsedToken).build()

    @classmethod
    def full_set_without_comments(cls) -> 'TokenCategory':
        return cls.Builder().add(ParsedToken).remove(Comment).build()

    @classmethod
    def only_comments(cls) -> 'TokenCategory':
        return cls.Builder().add(Comment).build()

    def contains(self, token_characteristics: TokenCharacteristics) -> bool:
        token_type = token_characteristics.token_type
        n_subtokens = token_characteristics.n_subtokens
        return token_type in self.all_included_types or \
               (token_type in self.types_of_specific_length and n_subtokens in self.types_of_specific_length[token_type])


def each_token_type_separately() -> Set[TokenCategory]:
    all_parsed_token_subclasses = all_subclasses([ParsedToken])
    return {TokenCategory.Builder().add(clazz).build() for clazz in all_parsed_token_subclasses}


def each_token_type_separately_and_lengths_for_identifiers() -> Set[TokenCategory]:
    all_token_types = each_token_type_separately()
    for i in range(5):
        all_token_types.add(TokenCategory.Builder().add_with_n_sub_tokens(Identifier, i).build())
    return all_token_types
