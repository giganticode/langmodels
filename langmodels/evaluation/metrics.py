from sys import maxsize

from typing import List, Tuple

from dataprep.parse.model.metadata import PreprocessingMetadata
from langmodels.evaluation.common import FullWordIterator, to_full_token_string
from langmodels.model import TrainedModel

DEFAULT_N_MODEL_SUGGESTIONS = 100


def word_average(subword_entropies: List[float], word_boundaries: List[int]) -> float:
    """
    >>> word_average([], [0])
    0.0

    >>> word_average([1.0, 2.0, 3.0], [0, 1, 3])
    3.0
    """
    word_entropies = [we for we in FullWordIterator(subword_entropies, word_boundaries, agg=sum)]
    if not word_entropies:
        return .0

    return sum(word_entropies) / len(word_entropies)


def bin_entropy(model: TrainedModel,
                prep_line: List[str],
                prep_metadata: PreprocessingMetadata) -> Tuple[List[float], float]:

    entropies = model.get_entropies_for_prep_text(prep_line)
    aggregated_entropy = word_average(entropies, prep_metadata.word_boundaries)
    return entropies, aggregated_entropy


def full_token_mrr(model: TrainedModel, prep_line: List[str], metadata: PreprocessingMetadata) \
        -> Tuple[List[int], float]:
    inverse_rank_sum = .0
    count = 0
    results = []
    for full_token in FullWordIterator(prep_line, metadata.word_boundaries):
        predictions = model.predict_next_full_token(n_suggestions=DEFAULT_N_MODEL_SUGGESTIONS)
        predicted_tokens = list(map(lambda p: p[0], predictions))
        actual_token = to_full_token_string(full_token)
        try:
            rank = predicted_tokens.index(actual_token) + 1
            inverse_rank_sum += 1. / rank
            results.append(rank)
        except ValueError:
            results.append(maxsize)
            inverse_rank_sum += 0.
        count += 1
        model.feed_text(actual_token)
    return results, count / inverse_rank_sum if inverse_rank_sum != 0 else 0.
