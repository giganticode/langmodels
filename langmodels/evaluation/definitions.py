from pprint import pformat
from typing import List, Optional, Tuple, Dict

import numpy as np
from dataclasses import dataclass

from evaluation import TokenTypeSubset
from evaluation.metrics import MetricName


@dataclass(frozen=True)
class EvaluationResult(object):
    tokens: List[str]
    token_types: List[str]
    values: List[float]
    aggregated_value: float
    of_context_length: Optional[List[Tuple[float, int]]] = None

    def to_summary(self) -> 'EvaluationResultSummary':
        return EvaluationResultSummary(self.aggregated_value, sum(x is not None for x in self.values), self.of_context_length)


@dataclass(frozen=True)
class EvaluationResultSummary(object):
    value: float
    n_samples: int
    of_context_length: Optional[List[Tuple[float, int]]]

    def _avg(self, values: List[float], n_samples_list: List[int]) -> Tuple[float, int]:
        total_samples = sum(n_samples_list)
        if total_samples == 0:
            return 0.0, 0
        resulting_value = np.average(values, weights=n_samples_list)
        return resulting_value, total_samples

    def merge(self, other_summary: 'EvaluationResultSummary'):
        resulting_value, total_samples = self._avg([self.value, other_summary.value], [self.n_samples, other_summary.n_samples])
        if self.of_context_length is None:
            of_context_length_result = other_summary.of_context_length
        elif other_summary.of_context_length is None:
            of_context_length_result = self.of_context_length
        else:
            of_context_length_result = [self._avg([v1, v2], [n1, n2]) for ((v1, n1), (v2, n2))
                                        in zip(self.of_context_length, other_summary.of_context_length)]
        return EvaluationResultSummary(resulting_value, total_samples, of_context_length_result)


@dataclass(frozen=True)
class EvaluationScenario(object):
    metric_name: MetricName
    type_subset: TokenTypeSubset = TokenTypeSubset.full_set()

    def __str__(self):
        return f'{self.metric_name}/{self.type_subset}'

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