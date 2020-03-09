from pprint import pformat
from typing import List, Optional, Tuple, Dict

import numpy as np
from dataclasses import dataclass

from langmodels.evaluation.customization import TokenCategory
from langmodels.model.context import ContextInformation
from langmodels.util.misc import merge_dicts_

MetricName = str


@dataclass(frozen=True)
class EvaluationResult(object):
    tokens: List[str]
    token_types: List[str]
    values: List[float]
    aggregated_value: float
    values_for_contexts: Optional[Dict[ContextInformation, Tuple[float, int]]] = None

    def to_summary(self) -> 'EvaluationResultSummary':
        return EvaluationResultSummary(self.aggregated_value, sum(x is not None for x in self.values), self.values_for_contexts)


@dataclass(frozen=True)
class EvaluationResultSummary(object):
    value: float
    n_samples: int
    values_for_contexts: Optional[Dict[ContextInformation, Tuple[float, int]]]

    def _avg(self, values: List[float], n_samples_list: List[int]) -> Tuple[float, int]:
        total_samples = sum(n_samples_list)
        if total_samples == 0:
            return 0.0, 0
        resulting_value = np.average(values, weights=n_samples_list)
        return resulting_value, total_samples

    def merge(self, other_summary: 'EvaluationResultSummary'):
        resulting_value, total_samples = self._avg([self.value, other_summary.value], [self.n_samples, other_summary.n_samples])
        if self.values_for_contexts is None:
            of_context_length_result = other_summary.values_for_contexts
        elif other_summary.values_for_contexts is None:
            of_context_length_result = self.values_for_contexts
        else:
            of_context_length_result = merge_dicts_(self.values_for_contexts,
                                                    other_summary.values_for_contexts,
                                                    value_merger=lambda x,y: self._avg([x[0], y[0]], [x[1], y[1]]))
        return EvaluationResultSummary(resulting_value, total_samples, of_context_length_result)


@dataclass(frozen=True)
class EvaluationScenario(object):
    metric_name: MetricName
    token_category: TokenCategory = TokenCategory.full_set()

    def __str__(self):
        return f'{self.metric_name}/{self.token_category}'

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Evaluation(object):
    text: str
    scenarios: Dict[EvaluationScenario, EvaluationResult]

    def __str__(self):
        return f'{self.text}\n{pformat(self.scenarios)}'

    def __repr__(self):
        return str(self)
