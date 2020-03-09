import logging
from collections import defaultdict

from typing import List, Optional, Set, Callable, Dict, Type

from codeprep.tokens import PreppedTokenSequence
from langmodels.evaluation.definitions import EvaluationResult
from langmodels.evaluation.customization import TokenCategory
from langmodels.model.model import TrainedModel, TokenCharacteristics
from langmodels.model.context import ContextModification

DEFAULT_N_MODEL_SUGGESTIONS = 100


logger = logging.getLogger(__name__)


def get_token_characteristics(prepped_token_sequence: PreppedTokenSequence) -> List[TokenCharacteristics]:
    return [TokenCharacteristics.from_metadata(t.metadata) for t in prepped_token_sequence.with_metadata()]


def bin_entropy(model: TrainedModel, line: str, extension: str, append_eof: bool,
                token_categories: Optional[Set[TokenCategory]] = None,
                context_modification: Optional[ContextModification] = None,
                full_tokens: bool = True) -> Dict[TokenCategory, EvaluationResult]:
    """
    Changes the state of the model!
    """
    token_categories = token_categories or {TokenCategory.full_set()}

    prepped_token_sequence, all_entropies, context_information_list = model.get_entropies_for_text(line, extension,
                                                                                                    full_tokens=full_tokens,
                                                                                                    append_eof=append_eof,
                                                                                                    context_modification=context_modification)

    tokens = [i for i in prepped_token_sequence.without_metadata()]
    token_characteristics_list = get_token_characteristics(prepped_token_sequence)
    all_token_types = [i for i in prepped_token_sequence.get_iterator(prepped_token_sequence.metadata.token_types, over_full_tokens=True)]
    evaluation_results: Dict[TokenCategory, EvaluationResult] = {}
    for token_category in token_categories:
        res = []
        sum = 0.0
        count = 0
        for entropy, token_characteristics in zip(all_entropies, token_characteristics_list):
            if token_category.contains(token_characteristics):
                res.append(entropy)
                sum += entropy
                count += 1
            else:
                res.append(None)
        if context_modification:
            of_context_length_cumul = defaultdict(lambda: (0.0, 0))
            for entropy, token_characteristics, context_information in zip(all_entropies, token_characteristics_list, context_information_list):
                if token_category.contains(token_characteristics):
                    if context_information is not None:
                        of_context_length_cumul[context_information] = (of_context_length_cumul[context_information][0] + entropy, of_context_length_cumul[context_information][1] + 1)
            of_context_length = {k: (val / n if n != 0 else 0.0, n) for k, (val, n) in of_context_length_cumul.items()}
        else:
            of_context_length = None
        evaluation_results[token_category] = EvaluationResult(tokens, list(map(lambda tt: tt.__name__, all_token_types)),
                                                                 res, sum / count if count else 0., of_context_length)
    return evaluation_results


def mrr(model: TrainedModel, line: str, extension: str, append_eof: bool,
        token_categories: Optional[Set[TokenCategory]] = None) \
        -> Dict[TokenCategory, EvaluationResult]:
    """
    Changes the state of the model!
    """
    token_categories = token_categories or {TokenCategory.full_set()}

    evaluation_results: Dict[TokenCategory, EvaluationResult] = {}
    for token_categories in token_categories:
        inverse_rank_sum = .0
        count = 0
        inverse_ranks: List[Optional[float]] = []
        all_tokens: List[str] = []
        all_token_types: List[str] = []

        for predictions, prep_token, token_characteristics in \
                model.get_predictions_and_feed(line, extension,
                                               n_suggestions=DEFAULT_N_MODEL_SUGGESTIONS,
                                               append_eof=append_eof):
            all_tokens.append(prep_token)
            all_token_types.append(token_characteristics.token_type.__name__)
            predicted_tokens = list(map(lambda p: p[0], predictions))
            if token_categories.contains(token_characteristics):
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
        evaluation_results[token_categories] = EvaluationResult(all_tokens, all_token_types, inverse_ranks, inverse_rank_sum / count if count else 1.)

    return evaluation_results


Metric = Callable[[TrainedModel, List[str], str, bool, Optional[Set[TokenCategory]], Dict[Type, float], int],
                  Dict[TokenCategory, EvaluationResult]]


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


