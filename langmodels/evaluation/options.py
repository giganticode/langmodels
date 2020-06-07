from dataclasses import field, dataclass
from typing import List, Optional

from langmodels.evaluation.characteristics import Characteristic, TokenType, SubtokenNumber
from langmodels.model.context import ContextModifier


@dataclass
class EvaluationOptions(object):
    metric_names: List[str] = field(default_factory=lambda: ['Entropy'])
    characteristics: List[Characteristic] = frozenset({TokenType(), SubtokenNumber()})
    context_modifier: Optional[ContextModifier] = None

    def __str__(self):
        return f'{self.metric_names}/{self.characteristics}'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self