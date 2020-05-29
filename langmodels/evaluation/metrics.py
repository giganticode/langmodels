import logging
from collections import defaultdict
from typing import Mapping

from tqdm import tqdm

from langmodels.evaluation.dataloader import BatchedTokenLoader
from langmodels.evaluation.result import ShortEvaluation, EvaluationScenarioGrid, EvaluationScenario, \
    ShortEvaluationResult
from langmodels.model._evaluator import calculate_losses
from langmodels.model.model import TrainedModel, retain_models_state
from langmodels.model.tokencategories import TokenCharacteristics, TokenCategory

logger = logging.getLogger(__name__)


@retain_models_state
def perplexity(model: TrainedModel, batched_token_loader: BatchedTokenLoader,
                evaluation_scenario_grid: EvaluationScenarioGrid,
                full_tokens: bool = True) -> ShortEvaluation:
    total_evaluation = ShortEvaluation()
    tqdmed_batched_losses = tqdm(calculate_losses(model, batched_token_loader), total=batched_token_loader.estimated_n_batches())
    for entropies_batch, file_structure_batch, prepped_tokens_batch in tqdmed_batched_losses:
        for entropies, prepped_tokens in zip(entropies_batch, prepped_tokens_batch):
            cur_batch_evaluation: ShortEvaluation = sequence_perplexity(entropies, prepped_tokens, evaluation_scenario_grid, full_tokens)
            total_evaluation = total_evaluation.merge(cur_batch_evaluation)

        update_progress_bar(tqdmed_batched_losses, batched_token_loader, total_evaluation, file_structure_batch)
    tqdmed_batched_losses.close()
    return total_evaluation


def sequence_perplexity(entropies, prepped_tokens, evaluation_scenario_grid, full_tokens):
    assert prepped_tokens.is_complete()
    current_eval_summary: Mapping[EvaluationScenario, ShortEvaluationResult] = defaultdict(ShortEvaluationResult)
    tokens = prepped_tokens.full_token_view(return_metadata=True) if full_tokens else prepped_tokens.sub_token_view(return_metadata=True)
    entropies_iterator = tokens.get_iterator(over=entropies, over_full_tokens=False, formatter=sum)
    for single_token_seq_elm, entropy in zip(tokens, entropies_iterator):
        for token_category in evaluation_scenario_grid.token_categories:
            if token_category.contains(TokenCharacteristics.from_metadata(single_token_seq_elm.metadata)):
                current_eval_summary[EvaluationScenario('perplexity', token_category)].add(single_token_seq_elm, entropy)
    return ShortEvaluation(current_eval_summary)


def update_progress_bar(tqdmed_batched_losses, batched_token_loader, total_evaluation, file_structure_batch):
    tqdmed_batched_losses.update(sum(map(lambda cs: len(cs.snippets) - 1, file_structure_batch)))
    try:
        evaluation_result_to_display = total_evaluation.scenarios[EvaluationScenario('perplexity', TokenCategory.full_set())]
    except KeyError:
        evaluation_result_to_display = next(iter(total_evaluation.scenarios.values()))
    estimated_n_batches = batched_token_loader.estimated_n_batches()
    current_iteration = batched_token_loader.current_iteration
    estimated_iterations_left = estimated_n_batches - current_iteration
    progress_bar_needs_to_be_updated = (estimated_iterations_left & (estimated_iterations_left - 1) == 0
                                        or current_iteration & (current_iteration - 1) == 0)
    tqdmed_batched_losses.set_postfix(ordered_dict={'perplexity': evaluation_result_to_display})
    if progress_bar_needs_to_be_updated:
        tqdmed_batched_losses.total = estimated_n_batches
        tqdmed_batched_losses.refresh()