import random
from dataclasses import dataclass, field
from typing import Optional

import sys

from codeprep.preprocess.metadata import PreppedTokenMetadata
from codeprep.preprocess.tokens import TokenSequence
from codeprep.tokentypes.containers import Identifier


@dataclass(frozen=True)
class RelativeShuffleInterval(object):
    start: int
    end: int


@dataclass(frozen=True)
class AbsoluteShuffleInterval(object):
    """
    >>> AbsoluteShuffleInterval(10, 20).get_relative_interval(to=25)
    RelativeShuffleInterval(start=5, end=15)
    >>> AbsoluteShuffleInterval(20, 30).get_relative_interval(to=25)
    RelativeShuffleInterval(start=-5, end=5)
    """
    start: int
    end: int

    def get_relative_interval(self, to: int) -> RelativeShuffleInterval:
        return RelativeShuffleInterval(start=to - self.end, end=to-self.start)


@dataclass(frozen=True)
class ContextInformation(object):
    length: int
    shuffle_interval: Optional[RelativeShuffleInterval] = None


@dataclass
class ModelContext(object):
    """
    >>> context = ModelContext()
    >>> context
    ModelContext(prepped_sub_tokens=[])
    >>> context.add(TokenSequence.create(['h', 'i</t>', 'th', 'er', 'e</t>'], PreppedTokenMetadata([2, 3], [Identifier, Identifier])))
    >>> context
    ModelContext(prepped_sub_tokens=[['h', 'i</t>'], ['th', 'er', 'e</t>']])
    >>> context.size()
    2
    >>> context.size(full_token=False)
    5
    >>> ModelContext.SAVE_FULL_TOKENS_CONTEXT_LIMIT = 2
    >>> context.add(TokenSequence.create(['b', 'g</t>'], PreppedTokenMetadata([2], [Identifier])))
    >>> context
    ModelContext(prepped_sub_tokens=[['th', 'er', 'e</t>'], ['b', 'g</t>']])
    >>> context.reset()
    >>> context
    ModelContext(prepped_sub_tokens=[])
    """
    prepped_sub_tokens: TokenSequence = field(default_factory=TokenSequence.empty)

    SAVE_FULL_TOKENS_CONTEXT_LIMIT = 1000

    def reset(self):
        self.prepped_sub_tokens = TokenSequence.empty()

    def add(self, prepped_tokens: TokenSequence) -> None:
        self.prepped_sub_tokens.extend(prepped_tokens)
        self.prepped_sub_tokens = self.prepped_sub_tokens.full_token_view()[-ModelContext.SAVE_FULL_TOKENS_CONTEXT_LIMIT:]

    def size(self, full_token: bool = True):
        return len(self.prepped_sub_tokens.full_token_view() if full_token else self.prepped_sub_tokens.sub_token_view())


@dataclass(frozen=True)
class ContextShuffling(object):
    start: int
    end: int


@dataclass(frozen=True)
class ContextModifier(object):
    max_context_length: int = sys.maxsize
    context_shuffling: Optional[ContextShuffling] = None

    def get_absolute_shuffle_interval(self) -> Optional[AbsoluteShuffleInterval]:
        return AbsoluteShuffleInterval(self.context_shuffling.start, self.context_shuffling.end) if self.context_shuffling else None

    def modify(self, tokens: TokenSequence, current_full_token_context: int) -> TokenSequence:
        """
        >>> from codeprep.preprocess.tokens import TokenSequence
        >>> class TypeA: pass
        >>> token_seq = TokenSequence.create(['hi</t>', 'the' ,'re</t>'], PreppedTokenMetadata([1, 2], [TypeA, TypeA]), word_end_token_added=True)
        >>> ContextModifier(context_shuffling=ContextShuffling(0, 1)).modify(token_seq, 0)
        [['hi</t>'], ['the', 're</t>']]
        """
        start = self.context_shuffling.start - current_full_token_context
        start = start if start > 0 else 0
        end = self.context_shuffling.end - current_full_token_context
        end = end if end > 0 else 0
        part_to_bhe_shuffled = tokens.full_token_view()[start:end]
        random.shuffle(part_to_bhe_shuffled)
        return tokens[:start].extend(part_to_bhe_shuffled).extend(tokens[end:])