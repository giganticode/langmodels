import logging
from dataclasses import dataclass

import numpy as np
from typing import List, Tuple, Optional, Set, Callable, Dict, Type

from dataprep.preprocess.metadata import PreprocessingMetadata
from dataprep.tokens.containers import Comment
from langmodels.evaluation.filtering import FullTokenIterator, to_full_token_string, FilteringTokenIterator, \
    TokenIterator, SubtokenIterator, MetricName, TokenTypeSubset
from langmodels.model import TrainedModel

DEFAULT_N_MODEL_SUGGESTIONS = 100


logger = logging.getLogger(__name__)


def conditional_average(subword_entropies: List[float],
                        metadata: PreprocessingMetadata,
                        token_type_subset: TokenTypeSubset = TokenTypeSubset.full_set(),
                        token_iterator_type: Type[TokenIterator] = FullTokenIterator) \
        -> Tuple[float, int]:
    """evaluator
    >>> conditional_average([], PreprocessingMetadata(set(), [0], []))
    (0.0, 0)

    >>> from dataprep.tokens.containers import SplitContainer
    >>> metadata = PreprocessingMetadata(set(), word_boundaries=[0, 1, 3], token_types=[Comment, SplitContainer])
    >>> conditional_average([1.0, 2.0, 3.0], metadata)
    (3.0, 2)

    >>> metadata = PreprocessingMetadata(set(), word_boundaries=[0, 1, 3], token_types=[Comment, SplitContainer])
    >>> token_type_subset = TokenTypeSubset.only_comments()
    >>> conditional_average([1.0, 2.0, 3.0], metadata, token_type_subset)
    (1.0, 1)

    >>> metadata = PreprocessingMetadata(set(), word_boundaries=[0, 1, 3], token_types=[Comment, SplitContainer])
    >>> token_type_subset = TokenTypeSubset.full_set_without_comments()
    >>> conditional_average([1.0, 2.0, 3.0], metadata, token_type_subset)
    (5.0, 1)

    >>> metadata = PreprocessingMetadata(set(), word_boundaries=[0, 1, 3], token_types=[Comment, SplitContainer])
    >>> token_type_subset = TokenTypeSubset.full_set_without_comments()
    >>> conditional_average([1.0, 2.0, 3.0], metadata, token_type_subset, SubtokenIterator)
    (2.5, 2)
    """
    word_entropies = [we for we in FilteringTokenIterator(subword_entropies, metadata, token_type_subset,
                                                          token_iterator_type=token_iterator_type, format=sum)]
    if not word_entropies:
        return .0, 0

    n_full_tokens = len(word_entropies)
    return sum(word_entropies) / n_full_tokens, n_full_tokens


@dataclass(frozen=True)
class EvaluationResult(object):
    subtoken_values: List[Optional[float]]  # this value should correspond to the number of subtokens
    average: float
    n_samples: int


@dataclass(frozen=True)
class EvaluationScenario(object):
    metric_name: MetricName
    token_type_subset: TokenTypeSubset

    def __str__(self):
        return f'{self.metric_name}/{self.token_type_subset}'

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Evaluation(object):
    text: str
    prep_text: List[str]
    prep_metadata: PreprocessingMetadata
    scenarios: Dict[EvaluationScenario, EvaluationResult]


def bin_entropy(model: TrainedModel, prep_line: List[str], prep_metadata: PreprocessingMetadata,
                token_types_subsets: Optional[Set[TokenTypeSubset]] = None,
                token_iterator_types: Set[Type[TokenIterator]] = None) -> Dict[EvaluationScenario, EvaluationResult]:
    """
    Changes the state of the model!
    """
    token_types_subsets = token_types_subsets or [TokenTypeSubset.full_set()]
    token_iterator_types = token_iterator_types or {FullTokenIterator}

    iterator_to_metric_map = {FullTokenIterator: 'full_token_entropy', SubtokenIterator: 'subtoken_entropy'}

    all_entropies = model.get_entropies_for_prep_text(prep_line)
    evaluation_results = {}
    for token_types_subset in token_types_subsets:
        for token_iterator_type in token_iterator_types:
            filtered_entropies = []
            for ind, e in FullTokenIterator(all_entropies, prep_metadata.word_boundaries, format=lambda x:x, return_full_token_index=True):
                filtered_entropies.extend(e if token_types_subset.contains(prep_metadata.token_types[ind]) else [None] * len(e))
            aggregated_entropy, n_full_tokens = conditional_average(filtered_entropies, prep_metadata,
                                                                    token_types_subset, token_iterator_type)
            evaluation_results[EvaluationScenario(iterator_to_metric_map[token_iterator_type], token_types_subset)] = \
                EvaluationResult(filtered_entropies, aggregated_entropy, n_full_tokens)
    return evaluation_results


def mrr(model: TrainedModel,
        prep_line: List[str],
        metadata: PreprocessingMetadata,
        token_types_subsets: Optional[Set[TokenTypeSubset]] = None) -> Dict[EvaluationScenario, EvaluationResult]:
    """
    Changes the state of the model!
    """
    token_types_subsets = token_types_subsets or [TokenTypeSubset.full_set()]
    result: Dict[EvaluationScenario, EvaluationResult] = {}
    for token_types_subset in token_types_subsets:
        inverse_rank_sum = .0
        count = 0
        inverse_ranks: List[Optional[float]] = []
        for full_token in FilteringTokenIterator(prep_line, metadata, token_types_subset, format=lambda l:l):
            predictions = model.predict_next_full_token(n_suggestions=DEFAULT_N_MODEL_SUGGESTIONS)
            predicted_tokens = list(map(lambda p: p[0], predictions))
            actual_token = to_full_token_string(full_token)
            if token_types_subset.contains(metadata.token_types[count]):
                try:
                    rank = predicted_tokens.index(actual_token) + 1
                    inverse_rank = 1. / rank
                except ValueError:  # actual token is not in prediction list
                    inverse_rank = 0.
                inverse_rank_sum += inverse_rank
                inverse_ranks.append(inverse_rank)
                inverse_ranks.extend([None for i in range(len(full_token) - 1)])
                count += 1
            else:
                inverse_ranks.extend([None for i in range(len(full_token))])
            model.feed_prep_tokens(full_token)
        result[EvaluationScenario('mrr', token_types_subset)] = EvaluationResult(inverse_ranks, inverse_rank_sum / count if count else 0., count)

    return result


def average_ranks(values: List[float], weights: List[int] = None) -> Tuple[float, int]:
    """
    >>> average_ranks([1, 5])
    (1.6666666666666667, 2)

    >>> average_ranks([1, 5, 9])
    (2.2881355932203387, 3)

    >>> average_ranks([1.6666666666666667, 9], weights=[2, 1])
    (2.2881355932203387, 3)
    """
    weights = weights or [1.0] * len(values)
    return 1.0 / np.average(list(map(lambda x: 1 / x, values)), weights=weights), int(sum(weights))


def average_entropy(values: List[float], weights: List[int] = None) -> Tuple[float, int]:
    """
    >>> average_entropy([6, 9], weights=[2, 1])
    (7.0, 3)
    """
    return np.average(values, weights=weights), sum(weights)


Metric = Callable[[TrainedModel, List[str], PreprocessingMetadata, Optional[Set[TokenTypeSubset]], Type[TokenIterator]],
                  Dict[EvaluationScenario, EvaluationResult]]
MetricAggregator = Callable[[List[float], List[int]], Tuple[float, int]]

metric_dict: Dict[MetricName, Tuple[Metric, MetricAggregator]] = {
    'subtoken_entropy': (lambda m, sw, md, tt: bin_entropy(m, sw, md, tt, {SubtokenIterator}), average_entropy),
    'full_token_entropy': (bin_entropy, average_entropy),
    'mrr': (mrr, average_ranks)
}


def get_metric_func_by_name(name: str) -> Metric:
    return metric_dict[name][0]


def get_metric_aggregator_by_name(name: str) -> MetricAggregator:
    return metric_dict[name][1]


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


