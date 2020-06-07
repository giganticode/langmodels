import logging
import os
from pathlib import Path
from typing import Optional

from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.evaluation.metrics import metric_name_to_function
from langmodels.evaluation.options import EvaluationOptions
from langmodels.evaluation.result import EvaluationResult, EvaluationResultAccumulator
from langmodels.model.model import TrainedModel

logger = logging.getLogger(__name__)


def evaluate(model: TrainedModel, token_loader: BatchedTokenLoader, save_to: Optional[Path] = None,
             evaluation_options: EvaluationOptions = EvaluationOptions(), full_tokens: bool = True) -> EvaluationResult:
    if save_to:
        if not save_to.exists():
            save_to.mkdir(parents=False)
        elif not os.access(save_to, os.W_OK | os.X_OK):
            raise FileNotFoundError(save_to)
        logger.info(f'Evaluation results will be written to {save_to}')

    total_evaluation_result_accumulator = EvaluationResultAccumulator.empty(characteristics=evaluation_options.characteristics, metric_names=evaluation_options.metric_names)
    for metric_name in evaluation_options.metric_names:
        try:
            metric_func = metric_name_to_function[metric_name]
        except KeyError:
            raise ValueError(f'Metric not implemented yet: {metric_name}')
        evaluation_summary = metric_func(model, token_loader, evaluation_options, full_tokens=full_tokens)
        total_evaluation_result_accumulator = total_evaluation_result_accumulator.merge(evaluation_summary)
    total_evaluation_result: EvaluationResult = total_evaluation_result_accumulator.build()
    if save_to:
        total_evaluation_result.save(save_to)
    return total_evaluation_result


def evaluate_on_string(model: TrainedModel, text: str,
                       evaluation_options: EvaluationOptions = EvaluationOptions(),
                       full_tokens: bool = True,
                       extension='java', append_eof: bool = False) -> EvaluationResult:
    """
    Evaluates the `model` on the provided `text` in scenarios specified by `evaluation_scenario_grid`
    """
    token_loader = BatchedTokenLoader.from_text(text, model.prep_text,
                                                extension=extension, append_eof=append_eof)

    return evaluate(model, token_loader, evaluation_options=evaluation_options, full_tokens=full_tokens)


def evaluate_on_file(model: TrainedModel, file: Path,
                     evaluation_options: EvaluationOptions = EvaluationOptions(),
                     full_tokens: bool = True) -> EvaluationResult:
    suffix: str = file.suffix[1:]
    model.assert_extension_supported(suffix)

    token_loader = BatchedTokenLoader.from_file(file, prep_func=model.prep_text, return_file_structure=True)

    return evaluate(model, token_loader, evaluation_options=evaluation_options, full_tokens=full_tokens)


def evaluate_on_path(model: TrainedModel, path: Path, save_to: Path,
                     evaluation_options: EvaluationOptions = EvaluationOptions(),
                     full_tokens: bool = True,
                     batch_size: int = 32) -> EvaluationResult:

    token_loader = BatchedTokenLoader.from_path(path, model.prep_text, batch_size=batch_size,
                                                return_file_structure=False,
                                                context_modifier=evaluation_options.context_modifier)

    return evaluate(model, token_loader, save_to, evaluation_options, full_tokens=full_tokens)
