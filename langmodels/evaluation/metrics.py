import logging
from typing import List, Optional, Set, Callable, Dict, Type
from pprint import pformat
from dataclasses import dataclass

from langmodels.evaluation.customization import TokenTypeSubset
from langmodels.model import TrainedModel

DEFAULT_N_MODEL_SUGGESTIONS = 100


logger = logging.getLogger(__name__)


MetricName = str


@dataclass(frozen=True)
class EvaluationResult(object):
    tokens: List[str]
    token_types: List[str]
    values: List[float]
    aggregated_value: float


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


def bin_entropy(model: TrainedModel, line: str, extension: str, append_eof: bool,
                token_type_subsets: Optional[Set[TokenTypeSubset]] = None,
                full_tokens: bool = True) \
        -> Dict[TokenTypeSubset, EvaluationResult]:
    """
    Changes the state of the model!
    """
    token_type_subsets = token_type_subsets or {TokenTypeSubset.full_set()}

    all_entropies, tokens, all_token_types = model.get_entropies_for_text(line, extension, full_tokens=full_tokens, append_eof=append_eof)
    evaluation_results: Dict[TokenTypeSubset, EvaluationResult] = {}
    for token_type_subset in token_type_subsets:
        res = []
        sum = 0.0
        count = 0
        for entropy, token_type in zip(all_entropies, all_token_types):
            if token_type_subset.contains(token_type):
                res.append(entropy)
                sum += entropy
                count += 1
            else:
                res.append(None)
        evaluation_results[token_type_subset] = EvaluationResult(tokens, list(map(lambda tt: tt.__name__, all_token_types)),
                                                                 res, sum / count if count else 0.)
    return evaluation_results


def mrr(model: TrainedModel, line: str, extension: str, append_eof: bool,
        token_type_subsets: Optional[Set[TokenTypeSubset]] = None) \
        -> Dict[TokenTypeSubset, EvaluationResult]:
    """
    Changes the state of the model!
    """
    token_type_subsets = token_type_subsets or {TokenTypeSubset.full_set()}

    evaluation_results: Dict[TokenTypeSubset, EvaluationResult] = {}
    for token_type_subsets in token_type_subsets:
        inverse_rank_sum = .0
        count = 0
        inverse_ranks: List[Optional[float]] = []
        all_tokens: List[str] = []
        all_token_types: List[str] = []

        for predictions, prep_token, token_type in \
                model.get_predictions_and_feed(line, extension,
                                               n_suggestions=DEFAULT_N_MODEL_SUGGESTIONS,
                                               append_eof=append_eof):
            all_tokens.append(prep_token)
            all_token_types.append(token_type.__name__)
            predicted_tokens = list(map(lambda p: p[0], predictions))
            if token_type_subsets.contains(token_type):
                try:
                    rank = predicted_tokens.index(prep_token) + 1
                    inverse_rank = 1. / rank
                except ValueError:  # actual token is not in prediction list
                    inverse_rank = 0.
                inverse_rank_sum += inverse_rank
                inverse_ranks.append(inverse_rank)
                count += 1
            else:
                inverse_ranks.append(None)
        evaluation_results[token_type_subsets] = EvaluationResult(all_tokens, all_token_types, inverse_ranks, inverse_rank_sum / count if count else 1.)

    return evaluation_results


Metric = Callable[[TrainedModel, List[str], str, bool, Optional[Set[TokenTypeSubset]], Dict[Type, float]],
                  Dict[TokenTypeSubset, EvaluationResult]]


def entropy_to_probability(entropy: float) -> float:
    """
    >>> entropy_to_probability(0.0)
    1.0

    >>> entropy_to_probability(1.0)
    0.5

    >>> entropy_to_probability(3.0)
    0.125

    >>> entropy_to_probability(100.0)
    7.888609052210118e-31
    """
    return 2 ** -entropy


