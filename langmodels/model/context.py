import random
from typing import Optional

import sys

from dataclasses import dataclass, field

from codeprep.preprocess.metadata import PreppedTokenMetadata
from codeprep.preprocess.result import PreppedSubTokenSequence, PreppedFullTokenSequence
from codeprep.tokentypes.containers import SplitContainer


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


@dataclass(frozen=True)
class ContextUsage(object):
    """
    >>> ContextUsage(length_start=10, reset_at=200, reset_times=0, length_end=10).n_predictions()
    0
    >>> ContextUsage(length_start=50, reset_at=200, reset_times=1, length_end=50).n_predictions()
    200
    >>> ContextUsage(length_start=10, reset_at=200, reset_times=2, length_end=50).n_predictions()
    440
    >>> ContextUsage(length_start=20, reset_at=10, reset_times=2, length_end=50).n_predictions()
    Traceback (most recent call last):
    ...
    AssertionError

    >>> ContextUsage(length_start=0, reset_at=1, reset_times=0, length_end=0).n_predictions()
    0

    >>> ContextUsage(length_start=199, reset_at=200, reset_times=0, length_end=200).n_predictions()
    1

    >>> context_usage = ContextUsage(length_start=199, reset_at=200, reset_times=1, length_end=1, \
shuffle_interval=AbsoluteShuffleInterval(100, 150))
    >>> context_usage.n_predictions()
    2

    >>> [i for i in iter(context_usage)]
    [ContextInformation(length=199, shuffle_interval=RelativeShuffleInterval(start=49, end=99)), \
ContextInformation(length=0, shuffle_interval=RelativeShuffleInterval(start=-150, end=-100))]

    >>> context_usage = ContextUsage(length_start=0, reset_at=1, reset_times=2, length_end=1)
    >>> context_usage.n_predictions()
    3
    >>> [i for i in iter(context_usage)]
    [ContextInformation(length=0, shuffle_interval=None), ContextInformation(length=0, shuffle_interval=None), \
ContextInformation(length=0, shuffle_interval=None)]

    >>> context_usage = ContextUsage(length_start=2, reset_at=3, reset_times=2, length_end=1, shuffle_interval=AbsoluteShuffleInterval(0, 2))
    >>> context_usage.n_predictions()
    5
    >>> [i for i in iter(context_usage)]
    [ContextInformation(length=2, shuffle_interval=RelativeShuffleInterval(start=0, end=2)), \
ContextInformation(length=0, shuffle_interval=RelativeShuffleInterval(start=-2, end=0)), \
ContextInformation(length=1, shuffle_interval=RelativeShuffleInterval(start=-1, end=1)), \
ContextInformation(length=2, shuffle_interval=RelativeShuffleInterval(start=0, end=2)), \
ContextInformation(length=0, shuffle_interval=RelativeShuffleInterval(start=-2, end=0))]
    """
    length_start: int
    reset_at: int
    reset_times: int
    length_end: int
    shuffle_interval: Optional[AbsoluteShuffleInterval] = None

    def is_last_chunk_complete(self):
        return self.reset_at == self.length_end

    class ContextUsageIterator(object):
        def __init__(self, context_usage: 'ContextUsage'):
            self.context_usage = context_usage
            self.current_length = context_usage.length_start
            self.context_times_reset = 0
            self.shuffle_interval = context_usage.shuffle_interval

        def __iter__(self):
            return self

        def __next__(self) -> ContextInformation:
            if (self.context_times_reset == self.context_usage.reset_times \
                    and self.current_length == self.context_usage.length_end) \
                    or self.context_times_reset > self.context_usage.reset_times:
                raise StopIteration
            to_return = self.current_length
            self.current_length += 1
            if self.current_length == self.context_usage.reset_at:
                self.current_length = 0
                self.context_times_reset += 1
            return ContextInformation(to_return,
                                      self.shuffle_interval.get_relative_interval(to_return) if self.shuffle_interval else None)

    def __post_init__(self):
        assert self.reset_at > self.length_start and self.reset_at >= self.length_end

    def n_predictions(self) -> int:
        return (self.reset_at - self.length_start) \
               + self.reset_at * (self.reset_times - 1) + self.length_end

    def __iter__(self):
        return ContextUsage.ContextUsageIterator(self)


@dataclass
class ModelContext(object):
    """
    >>> context = ModelContext()
    >>> context
    ModelContext(prepped_sub_tokens=[])
    >>> context.add(PreppedSubTokenSequence(['h', 'i</t>', 'th', 'er', 'e</t>'], PreppedTokenMetadata([2, 3], [SplitContainer, SplitContainer])))
    >>> context
    ModelContext(prepped_sub_tokens=['h', 'i</t>', 'th', 'er', 'e</t>'])
    >>> context.size()
    2
    >>> context.size(full_token=False)
    5
    >>> ModelContext.SAVE_FULL_TOKENS_CONTEXT_LIMIT = 2
    >>> context.add(PreppedSubTokenSequence(['b', 'g</t>'], PreppedTokenMetadata([2], [SplitContainer])))
    >>> context
    ModelContext(prepped_sub_tokens=['th', 'er', 'e</t>', 'b', 'g</t>'])
    >>> context.reset()
    >>> context
    ModelContext(prepped_sub_tokens=[])
    """
    prepped_sub_tokens: PreppedSubTokenSequence = field(default_factory=PreppedSubTokenSequence)

    SAVE_FULL_TOKENS_CONTEXT_LIMIT = 1000

    def reset(self):
        self.prepped_sub_tokens = PreppedSubTokenSequence()

    def add(self, prepped_tokens: PreppedSubTokenSequence) -> None:
        self.prepped_sub_tokens = self.prepped_sub_tokens.add(prepped_tokens)
        self.prepped_sub_tokens = self.prepped_sub_tokens.full_tokens_view()[-ModelContext.SAVE_FULL_TOKENS_CONTEXT_LIMIT:].sub_token_view()

    def size(self, full_token: bool = True):
        return len(self.prepped_sub_tokens.full_tokens_view() if full_token else self.prepped_sub_tokens)


@dataclass(frozen=True)
class ContextShuffling(object):
    start: int
    end: int


@dataclass(frozen=True)
class ContextModification(object):
    max_context_length: int = sys.maxsize
    context_shuffling: Optional[ContextShuffling] = None

    def get_absolute_shuffle_interval(self) -> Optional[AbsoluteShuffleInterval]:
        return AbsoluteShuffleInterval(self.context_shuffling.start, self.context_shuffling.end) if self.context_shuffling else None


def modify_context(tokens: PreppedFullTokenSequence, context_modification: ContextModification):
    """
    >>> class TypeA: pass
    >>> prepped_tokens = PreppedFullTokenSequence(['hi</t>', 'the' ,'re</t>'], PreppedTokenMetadata([1, 2], [TypeA, TypeA]), word_end_token_added=True)
    >>> modify_context(prepped_tokens, ContextModification(context_shuffling=ContextShuffling(0, 1)))
    [['hi</t>'], ['the', 're</t>']]
    """
    if not context_modification:
        return tokens
    start = context_modification.context_shuffling.start
    end = context_modification.context_shuffling.end
    part_to_bhe_shuffled = tokens[start:end]
    random.shuffle(part_to_bhe_shuffled)
    return tokens[:start].add(part_to_bhe_shuffled).add(tokens[end:])