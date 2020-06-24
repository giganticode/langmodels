from dataclasses import field, dataclass
from typing import List, Optional

from langmodels.evaluation.characteristics import Characteristic, TokenType, SubtokenNumber, Project
from langmodels.model.context import ContextModifier


@dataclass
class EvaluationOptions(object):
    metric_names: List[str]
    characteristics: List[Characteristic]
    context_modifier: Optional[ContextModifier] = None

    def __str__(self):
        return f'{self.metric_names}/{self.characteristics}'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self