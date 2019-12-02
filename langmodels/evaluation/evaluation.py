import logging
import os
from pathlib import Path

from tqdm import tqdm
from typing import List, Tuple, Callable, Optional, Union, Dict, Set

from langmodels.file_util import get_all_files, read_file_contents, get_file_extension
from langmodels.evaluation.metrics import bin_entropy, mrr, get_metric_aggregator_by_name, Metric, \
    get_metric_func_by_name, TokenTypes, EvaluationScenario, Evaluation
from langmodels.model import TrainedModel

logger = logging.getLogger(__name__)


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


def metrics_from_strings(_metrics: Union[Set[str], Set[Callable], None]) -> Optional[Set[Callable]]:
    if _metrics is None:
        return None

    return set(map(lambda m: m if isinstance(m, Callable) else get_metric_func_by_name(m), _metrics))


DEFAULT_METRIC = bin_entropy


def evaluate_model_on_string(model: TrainedModel, text: str, extension='java',
                             token_types: Optional[Set[TokenTypes]] = None,
                             metrics: Optional[Union[Set[Metric], Set[str]]] = None,
                             result_per_line=True, append_eof: bool = False) -> List[Evaluation]:
    metrics = metrics_from_strings(metrics) or {DEFAULT_METRIC}

    show_progress = result_per_line and mrr in metrics

    text_lines = text.split('\n') if result_per_line else [text]
    model_evaluation: List[Evaluation] = []
    line_iterable = tqdm(text_lines) if show_progress else text_lines
    for i, line in enumerate(line_iterable):
        prep_line, metadata = model.prep_text(line, return_metadata=True,
                                              force_reinit_bpe_data=False,
                                              extension=extension,
                                              append_eof=append_eof and i == len(line_iterable) - 1)
        scenarios = {k: v for metric in metrics for k, v in metric(model, prep_line, metadata, token_types).items()}
        line_evaluation = Evaluation(text=line, prep_text=prep_line, prep_metadata=metadata, scenarios=scenarios)
        model_evaluation.append(line_evaluation)
    return model_evaluation


def evaluate_model_on_file(model: TrainedModel, file: Path,
                           token_types: Optional[Set[TokenTypes]] = None,
                           metrics: Optional[Union[Set[Metric], Set[str]]] = None,
                           result_per_line: bool = True) -> Union[List[Evaluation], Evaluation]:
    suffix: str = file.suffix[1:]
    model.check_inference_possible_for_file_type(suffix)
    text = read_file_contents(file)
    result = evaluate_model_on_string(model, text, suffix, token_types, metrics,
                                      result_per_line=result_per_line, append_eof=True)
    return result if result_per_line else result[0]


def _format_postfix(current_metrics: Dict[EvaluationScenario, Tuple[float, int]]) -> Dict[str, str]:
    if current_metrics:
        postfix = {str(eval_scenario): f'{value:.2f} (n={n_samples})'
                   for eval_scenario, (value, n_samples) in current_metrics.items()}
    else:
        postfix = {}
    return postfix


def evaluate_model_on_project_set(model: TrainedModel, path: str,
                                  token_types: Optional[Set[TokenTypes]] = None,
                                  metrics: Optional[List[Metric]] = None) \
        -> Dict[str, Dict[EvaluationScenario, Tuple[float, int]]]:
    result: Dict[str, Dict[EvaluationScenario, Tuple[float, int]]] = {}
    root, dirs, files = next(os.walk(path, followlinks=True))
    for directory in dirs:
        logger.info(f'Evaluating {directory} ...')
        result[directory] = evaluate_model_on_path(model, os.path.join(root, directory), token_types, metrics)
    return result


def evaluate_model_on_path(model: TrainedModel, path: str, token_types: Optional[Set[TokenTypes]] = None,
                           metrics: Optional[Union[Set[Metric], Set[str]]] = None) \
        -> Dict[EvaluationScenario, Tuple[float, int]]:
    token_types = token_types or {TokenTypes.ALL}

    logger.info("Counting total file number ...")
    all_files = [f for f in get_all_files(path)]

    cumulative_metrics: Dict[EvaluationScenario, Tuple[float, int]] = {}
    t = tqdm(all_files)
    for file in t:
        postfix = _format_postfix(cumulative_metrics)
        t.set_postfix(postfix)
        evaluation: Evaluation = evaluate_model_on_file(model, file, token_types, metrics, result_per_line=False)

        current_file_metrics = {scenario: (eval_result.average, eval_result.n_samples)
                                for scenario, eval_result in evaluation.scenarios.items()}

        if cumulative_metrics:
            for scenario, cumulative_eval_result in cumulative_metrics.items():
                metric_agg = get_metric_aggregator_by_name(scenario.metric_name)
                cumulative_metrics[scenario] = metric_agg(*zip(cumulative_eval_result, current_file_metrics[scenario]))
        else:
            cumulative_metrics = current_file_metrics
    if not cumulative_metrics:
        raise Exception(f"No files to evaluate are found in {path}")

    return cumulative_metrics
