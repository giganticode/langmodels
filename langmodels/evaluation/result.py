from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pprint import pformat
from typing import List, Optional, Mapping, Set

from math import exp

from codeprep.preprocess.tokens import TokenSequence
from langmodels.evaluation.codestructure import CodeLocation
from langmodels.model.context import ContextInformation, ContextModifier
from langmodels.model.tokencategories import TokenCategory
from langmodels.util.misc import merge_dicts_


@dataclass
class EvaluationResult(ABC):
    @abstractmethod
    def add(self, token: TokenSequence, value: float):
        pass

    @abstractmethod
    def merge(self, other_summary: 'EvaluationResult') -> 'EvaluationResult':
        pass


@dataclass
class FullEvaluationResult(EvaluationResult):
    tokens: TokenSequence
    values: List[Optional[float]]
    exp: bool = True

    def add(self, token: TokenSequence, value: float):
        self.tokens = self.tokens.add(token)
        self.values.append(value)

    def merge(self, other_summary: 'FullEvaluationResult') -> 'FullEvaluationResult':
        assert self.exp == other_summary.exp
        return FullEvaluationResult(self.tokens + other_summary.tokens, self.values + other_summary.values)

    def get_present_values(self) -> List[float]:
        return [v for v in self.values if v is not None]

    def to_short_evaluation_result(self) -> 'ShortEvaluationResult':
        present_values = self.get_present_values()
        return ShortEvaluationResult(sum(present_values), len(present_values), exp=self.exp)


@dataclass()
class ShortEvaluationResult(EvaluationResult):
    sum: float = 0.0
    n_samples: int = 0
    exp: bool = True

    def add(self, token: TokenSequence, value: float):
        self.sum += value
        self.n_samples += 1

    def merge(self, other_summary: 'ShortEvaluationResult') -> 'ShortEvaluationResult':
        assert self.exp == other_summary.exp
        return ShortEvaluationResult(self.sum + other_summary.sum, self.n_samples + other_summary.n_samples, exp=self.exp)

    @property
    def value(self):
        avg: float = self.sum / self.n_samples
        return exp(avg) if self.exp else avg

    def __str__(self):
        return f'{self.value} ({self.n_samples} samples)'


@dataclass(frozen=True)
class EvaluationScenario(object):
    metric_name: str
    token_category: TokenCategory = TokenCategory.full_set()
    context_information: Optional[ContextInformation] = None

    def __str__(self):
        return f'{self.metric_name}/{self.token_category}'

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class EvaluationScenarioGrid(object):
    metric_names: Set[str] = frozenset({'perplexity'})
    token_categories: Set[TokenCategory] = frozenset({TokenCategory.full_set()})
    context_modifier: Optional[ContextModifier] = None

    def __str__(self):
        return f'{self.metric_names}/{self.token_categories}'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self


@dataclass
class FullEvaluation(object):
    scenarios: Mapping[EvaluationScenario, FullEvaluationResult] = field(default_factory=dict)

    def merge(self, other_summary: 'FullEvaluation'):
        return FullEvaluation(merge_dicts_(self.scenarios, other_summary.scenarios, value_merger=FullEvaluationResult.merge))

    def __str__(self):
        return f'{pformat(self.scenarios)}'

    def __repr__(self):
        return str(self)


@dataclass
class ShortEvaluation(object):
    scenarios: Mapping[EvaluationScenario, ShortEvaluationResult] = field(default_factory=dict)

    def merge(self, other_summary: 'ShortEvaluation'):
        return ShortEvaluation(merge_dicts_(self.scenarios, other_summary.scenarios, value_merger=ShortEvaluationResult.merge))

    def __str__(self):
        return f'{pformat({str(k): str(v) for k, v in self.scenarios.items()})}'

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class PerLineEvaluation:
    locations: Mapping[CodeLocation, FullEvaluation]


@dataclass(frozen=True)
class PerLineEvaluationSummary:
    locations: Mapping[CodeLocation, ShortEvaluation]
