from tqdm import tqdm
from typing import List, Tuple, Callable

from langmodels.evaluation.common import EvaluationResult, get_file_total_lines, get_file_extension, LMEvaluator
from langmodels.model import TrainedModel

ImprovementMetric = Callable[[float, float], float]


def zip_subwords(subwords: Tuple[List[str], List[str]],
                 entropies: Tuple[List[float], List[float]]
                 ) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    >>> zip_subwords((['a'], ['a']), ([1.0], [2.0]))
    (['a'], [(1.0, 2.0)])

    >>> zip_subwords((['a', 'b'], ['a', 'b']), ([1.0, 2.0], [3.0, 4.0]))
    (['a', 'b'], [(1.0, 3.0), (2.0, 4.0)])

    >>> zip_subwords((['a', 'b'], ['ab']), ([1.0, 2.0], [7.0]))
    (['ab'], [(3.0, 7.0)])

    >>> zip_subwords((['abcdef'], ['ab', 'cd', 'ef']), ([1.0], [7.0, 4.0, 5.0]))
    (['abcdef'], [(1.0, 16.0)])

    >>> zip_subwords((['a', 'bc', 'd'], ['ab', 'c', 'd']), ([1.0, 2.0, 7.0], [4.0, 5.0, 78.0]))
    (['abc', 'd'], [(3.0, 9.0), (7.0, 78.0)])

    >>> zip_subwords((['a'], ['ab']), ([1.0], [2.0]))
    Traceback (most recent call last):
    ...
    AssertionError
    """
    res: Tuple[List[str], List[Tuple[float, float]]] = ([], [])
    indices = [0, 0]
    next_subwords = tuple([s[i] for s, i in zip(subwords, indices)])
    entropy_sums = [.0, .0]
    prefix = ""
    while True:
        if next_subwords[0] == next_subwords[1]:
            res[0].append(prefix + next_subwords[0])
            res[1].append(tuple(
                e + es[i] for e, es, i in zip(entropy_sums, entropies, indices)
            ))
            entropy_sums = [.0, .0]
            indices = tuple(i+1 for i in indices)
            if indices[1] == len(subwords[1]):
                break
            next_subwords = tuple(s[i] for s, i in zip(subwords, indices))
            prefix = ""
        elif not next_subwords[0].startswith(next_subwords[1]) and \
                not next_subwords[1].startswith(next_subwords[1]):
            raise ValueError(f"Different words passed: {prefix}{next_subwords[1]} != {prefix}{next_subwords[0]}")
        else:
            if next_subwords[0].startswith(next_subwords[1]):
                longer = 0
                shorter = 1
            else:
                longer = 1
                shorter = 0

            entropy_sums[shorter] += entropies[shorter][indices[shorter]]
            indices[shorter] += 1
            prefix += next_subwords[shorter]
            if indices[shorter] == len(subwords[shorter]):
                break
            first_common_chars = len(next_subwords[shorter])
            next_subwords = (
                next_subwords[longer][first_common_chars:],
                subwords[shorter][indices[shorter]]
            )
            if longer == 1:
                next_subwords = next_subwords[1], next_subwords[0]

    for i, s in zip(indices, subwords):
        assert i == len(s)

    return res


def run_evaluation(models: List[TrainedModel], file: str, evaluators: List[LMEvaluator]) -> Tuple[List[EvaluationResult]]:
    lines_in_file = get_file_total_lines(file)
    result: List[List[EvaluationResult]] = []

    for model in models:
        model_evaluation: List[EvaluationResult] = []
        with open(file, 'r') as f:
            t = tqdm(f, total=lines_in_file)
            extension = get_file_extension(file)
            for line in t:
                prep_line, metadata = model.prep_text(line, return_metadata=True, force_reinit_bpe_data=False,
                                                      extension=extension)

                metrics_dict = {}
                agg_metrics_dict = {}
                for evaluator in evaluators:
                    metrics, agg_metrics = evaluator(model, prep_line, metadata)
                    metrics_dict[evaluator.__name__] = metrics
                    agg_metrics_dict[evaluator.__name__] = agg_metrics

                line_result = EvaluationResult(text=line, prep_text=prep_line, prep_metadata=metadata,
                                               results=metrics_dict, aggregated_result=agg_metrics_dict)

                model_evaluation.append(line_result)
        result.append(model_evaluation)
    return tuple(result)
