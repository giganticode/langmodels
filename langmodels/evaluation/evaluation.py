import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Set

import numpy as np
from tqdm import tqdm

from langmodels.evaluation.customization import TokenTypeSubset
from langmodels.evaluation.metrics import bin_entropy
from langmodels.evaluation.metrics import mrr, Metric, EvaluationScenario, Evaluation, EvaluationResult, \
    MetricName
from langmodels.file_util import get_all_files, read_file_contents
from langmodels.model import TrainedModel

logger = logging.getLogger(__name__)


metric_dict: Dict[MetricName, Metric] = {
    'subtoken_entropy': lambda m, s, ext, eof, evc: bin_entropy(m, s, ext, eof, evc, False),
    'full_token_entropy': bin_entropy,
    'mrr': mrr
}


def _get_metric_by_name(name: str) -> Metric:
    return metric_dict[name]


def get_metrics_name(metric: Metric) -> str:
    for k, v in metric_dict.items():
        if v == metric:
            return k
    raise KeyError(f'Unknown metric: {metric}')


DEFAULT_METRIC_NAME = 'full_token_entropy'


def _evaluate_model_on_line(model: TrainedModel, line: str, extension: str,
                            metrics: Optional[Set[MetricName]],
                            token_type_subsets: Optional[Set[TokenTypeSubset]],
                            append_eof: bool) -> Evaluation:
    results: Dict[EvaluationScenario, EvaluationResult] = {}
    for metric_name in metrics:
        metric = _get_metric_by_name(metric_name)
        for evaluation_customization, evaluation_result in metric(model, line, extension, append_eof, token_type_subsets).items():
            results[EvaluationScenario(metric_name, evaluation_customization)] = evaluation_result
    return Evaluation(text=line, scenarios=results)


def evaluate_model_on_string(model: TrainedModel, text: str, extension='java',
                             metrics: Optional[Set[str]] = None,
                             token_type_subsets: Optional[Set[TokenTypeSubset]] = None,
                             result_per_line=True, append_eof: bool = False) -> Union[List[Evaluation], Evaluation]:

    metrics = metrics or {DEFAULT_METRIC_NAME}

    show_progress = result_per_line and mrr in metrics

    text_lines = text.split('\n') if result_per_line else [text]
    model_evaluation: List[Evaluation] = []
    line_iterable = tqdm(text_lines) if show_progress else text_lines
    for i, line in enumerate(line_iterable):
        last_line = i == len(line_iterable) - 1
        line_evaluation = _evaluate_model_on_line(model, line, extension, metrics, token_type_subsets, append_eof and last_line)
        model_evaluation.append(line_evaluation)
    return model_evaluation if result_per_line else model_evaluation[0]


def evaluate_model_on_file(model: TrainedModel, file: Path, metrics: Optional[Set[str]] = None,
                           token_type_subsets: Optional[Set[TokenTypeSubset]] = None,
                           result_per_line: bool = True) -> Union[List[Evaluation], Evaluation]:
    suffix: str = file.suffix[1:]
    model._assert_inference_possible_for_file_type(suffix)
    text = read_file_contents(file)
    return evaluate_model_on_string(model, text, suffix, metrics, token_type_subsets,
                                    result_per_line=result_per_line, append_eof=True)


def evaluate_model_on_project_set(model: TrainedModel, path: Path, metrics: Optional[Set[str]] = None,
                                  token_type_subsets: Optional[Set[TokenTypeSubset]] = None) \
        -> Dict[str, Dict[EvaluationScenario, Tuple[float, int]]]:
    result: Dict[str, Dict[EvaluationScenario, Tuple[float, int]]] = {}
    try:
        root, dirs, _ = next(os.walk(str(path), followlinks=True))
    except StopIteration:
        raise ValueError(f'Path {path} is a file or does not exist?')

    if not dirs:
        raise ValueError(f'Path {path} contains no projects')

    for directory in dirs:
        logger.info(f'Evaluating {directory} ...')
        result[directory] = evaluate_model_on_path(model, Path(os.path.join(root, directory)),
                                                   metrics, token_type_subsets)
    return result


def _format_postfix(current_metrics: Dict[EvaluationScenario, Tuple[float, int]]) -> Dict[str, str]:
    if current_metrics:
        return {}

    return {str(eval_scenario): f'{value:.2f} (n={n_samples})'
            for eval_scenario, (value, n_samples) in current_metrics.items()}


def evaluate_model_on_path(model: TrainedModel, path: Path, metrics: Optional[Set[str]] = None,
                           token_type_subsets: Optional[Set[TokenTypeSubset]] = None) \
        -> Dict[EvaluationScenario, Tuple[float, int]]:

    logger.info("Counting total file number ...")
    all_files = [f for f in get_all_files(str(path))]

    cumulative_metrics: Dict[EvaluationScenario, Tuple[float, int]] = {}
    t = tqdm(all_files)
    for file in t:
        postfix = _format_postfix(cumulative_metrics)
        t.set_postfix(postfix)
        evaluation: Evaluation = evaluate_model_on_file(model, file, metrics,
                                                        token_type_subsets, result_per_line=False)

        current_file_metrics = {scenario: (eval_result.aggregated_value, sum(x is not None for x in eval_result.values))
                                for scenario, eval_result in evaluation.scenarios.items()}

        if cumulative_metrics:
            for scenario, cumulative_eval_result in cumulative_metrics.items():
                def avg(v, w):
                    total_count = sum(w)
                    if total_count == 0:
                        return 0.0, 0
                    return np.average(v, weights=w), total_count

                cumulative_metrics[scenario] = avg(*zip(cumulative_eval_result, current_file_metrics[scenario]))
        else:
            cumulative_metrics = current_file_metrics
    if not cumulative_metrics:
        raise Exception(f"No files to evaluate are found in {path}")

    return cumulative_metrics
