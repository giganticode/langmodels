from dataclasses import dataclass, field

from codeprep.preprocess.result import PreppedSubTokenSequence


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

    >>> context_usage = ContextUsage(length_start=199, reset_at=200, reset_times=1, length_end=1)
    >>> context_usage.n_predictions()
    2

    >>> [i for i in iter(context_usage)]
    [199, 0]

    >>> context_usage = ContextUsage(length_start=0, reset_at=1, reset_times=2, length_end=1)
    >>> context_usage.n_predictions()
    3
    >>> [i for i in iter(context_usage)]
    [0, 0, 0]

    >>> context_usage = ContextUsage(length_start=2, reset_at=3, reset_times=2, length_end=1)
    >>> context_usage.n_predictions()
    5
    >>> [i for i in iter(context_usage)]
    [2, 0, 1, 2, 0]
    """
    length_start: int
    reset_at: int
    reset_times: int
    length_end: int

    def is_last_chunk_complete(self):
        return self.reset_at == self.length_end

    class ContextUsageIterator(object):
        def __init__(self, context_usage: 'ContextUsage'):
            self.context_usage = context_usage
            self.current_length = context_usage.length_start
            self.context_times_reset = 0

        def __iter__(self):
            return self

        def __next__(self):
            if (self.context_times_reset == self.context_usage.reset_times \
                    and self.current_length == self.context_usage.length_end) \
                    or self.context_times_reset > self.context_usage.reset_times:
                raise StopIteration
            to_return = self.current_length
            self.current_length += 1
            if self.current_length == self.context_usage.reset_at:
                self.current_length = 0
                self.context_times_reset += 1
            return to_return

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