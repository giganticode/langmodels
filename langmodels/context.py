from dataclasses import dataclass, field

from codeprep.preprocess.metadata import PreppedTokenMetadata
from codeprep.preprocess.result import PreppedSubTokenSequence
from codeprep.tokens.containers import SplitContainer


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
