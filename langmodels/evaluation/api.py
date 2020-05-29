import logging
from pathlib import Path

from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.evaluation.metrics import perplexity
from langmodels.evaluation.result import EvaluationScenarioGrid
from langmodels.evaluation.result import ShortEvaluation
from langmodels.model.model import TrainedModel

logger = logging.getLogger(__name__)


def evaluate_on_string(model: TrainedModel, text: str,
                       evaluation_scenario_grid: EvaluationScenarioGrid = EvaluationScenarioGrid(),
                       full_tokens: bool = True,
                       extension='java', append_eof: bool = False) -> ShortEvaluation:
    """
    Evaluates the `model` on the provided `text` in scenarios specified by `evaluation_scenario_grid`
    """

    total_evaluation = ShortEvaluation()
    for metric_name in evaluation_scenario_grid.metric_names:
        if metric_name == 'perplexity':
            evaluation = perplexity(model,
                                    BatchedTokenLoader.from_text(text, model.prep_text,
                                                                 extension=extension, append_eof=append_eof),
                                    evaluation_scenario_grid, full_tokens=full_tokens)
        else:
            raise ValueError(f'Metric not implemented yet: {metric_name}')
        total_evaluation.merge(evaluation)
    return total_evaluation


def evaluate_on_file(model: TrainedModel, file: Path,
                     evaluation_scenario_grid: EvaluationScenarioGrid = EvaluationScenarioGrid(),
                     full_tokens: bool = True) -> ShortEvaluation:
    suffix: str = file.suffix[1:]
    model.assert_extension_supported(suffix)

    token_loader = BatchedTokenLoader.from_file(file, prep_func=model.prep_text, return_file_structure=True)

    total_evaluation_summary = ShortEvaluation()
    for metric_name in evaluation_scenario_grid.metric_names:
        if metric_name == 'perplexity':
            evaluation_summary = perplexity(model, token_loader, evaluation_scenario_grid, full_tokens=full_tokens)
        else:
            raise ValueError(f'Metric not implemented yet: {metric_name}')
        total_evaluation_summary = total_evaluation_summary.merge(evaluation_summary)

    return total_evaluation_summary


def evaluate_on_path(model: TrainedModel, path: Path,
                     evaluation_scenario_grid: EvaluationScenarioGrid = EvaluationScenarioGrid(),
                     full_tokens: bool = True,
                     batch_size: int = 32) -> ShortEvaluation:

    token_loader = BatchedTokenLoader.from_path(path, model.prep_text, batch_size=batch_size,
                                                return_file_structure=False,
                                                context_modifier=evaluation_scenario_grid.context_modifier)

    total_evaluation_summary = ShortEvaluation()
    for metric_name in evaluation_scenario_grid.metric_names:
        if metric_name == 'perplexity':
            evaluation_summary = perplexity(model, token_loader, evaluation_scenario_grid, full_tokens=full_tokens)
        else:
            raise ValueError(f'Metric not implemented yet: {metric_name}')
        total_evaluation_summary = total_evaluation_summary.merge(evaluation_summary)

    return total_evaluation_summary
