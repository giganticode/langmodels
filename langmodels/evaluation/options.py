from dataclasses import dataclass
from typing import List, Optional

from langmodels.evaluation.characteristics import Characteristic
from langmodels.context import ContextModifier


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