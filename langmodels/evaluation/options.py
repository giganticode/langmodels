from dataclasses import field, dataclass
from typing import List

from langmodels.evaluation.characteristics import Characteristic, TokenType, SubtokenNumber


@dataclass
class EvaluationOptions(object):
    metric_names: List[str] = field(default_factory=lambda: ['Entropy'])
    characteristics: List[Characteristic] = frozenset({TokenType(), SubtokenNumber()})

    def __str__(self):
        return f'{self.metric_names}/{self.characteristics}'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self